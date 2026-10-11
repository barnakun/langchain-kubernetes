"""WebSocket exec and tar-based file transfer for raw Kubernetes Pods.

All functions require the ``kubernetes`` package (``pip install langchain-kubernetes[raw]``).
They are kept separate from the backend class so they can be unit-tested by
mocking the stream API.

Binary data (tar archives) is transported via base64 encoding to avoid
corruption through the UTF-8-decoded text channel of the Kubernetes exec API.
"""

from __future__ import annotations

import base64
import io
import logging
import tarfile
import time
from typing import TYPE_CHECKING

from deepagents.backends.protocol import FileDownloadResponse, FileUploadResponse

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Maximum output bytes captured from a single exec call before truncation.
_OUTPUT_LIMIT = 1024 * 1024  # 1 MiB


# Module-level shim for the kubernetes stream function.  Defined here so that
# tests can patch ``langchain_kubernetes.backends.raw_transport.stream`` without
# requiring the ``kubernetes`` package to be installed.
def stream(*args, **kwargs):
    """Lazy proxy for ``kubernetes.stream.stream``.

    All transport functions call this wrapper so tests can patch it as a
    module-level attribute.
    """
    from kubernetes.stream import stream as _k8s_stream

    return _k8s_stream(*args, **kwargs)


# ---------------------------------------------------------------------------
# Command execution
# ---------------------------------------------------------------------------


def exec_command(
    core_v1,
    pod_name: str,
    namespace: str,
    container: str,
    command: str,
    timeout: int = 60,
) -> tuple[str, int, bool]:
    """Execute a shell command inside a Pod via the Kubernetes exec API.

    Uses the WebSocket exec endpoint. stdout and stderr are captured and
    combined into a single string. If the combined output exceeds
    ``_OUTPUT_LIMIT`` bytes it is truncated and ``truncated=True`` is returned.

    Args:
        core_v1: ``kubernetes.client.CoreV1Api`` instance.
        pod_name: Name of the target Pod.
        namespace: Namespace containing the Pod.
        container: Container to exec into.
        command: Shell command string (executed via ``/bin/sh -c``).
        timeout: Maximum seconds to wait for the command to finish.

    Returns:
        Tuple of ``(combined_output, exit_code, truncated)``.
    """
    resp = stream(
        core_v1.connect_get_namespaced_pod_exec,
        name=pod_name,
        namespace=namespace,
        container=container,
        command=["/bin/sh", "-c", command],
        stderr=True,
        stdin=False,
        stdout=True,
        _preload_content=False,
    )

    stdout_parts: list[str] = []
    stderr_parts: list[str] = []
    deadline = time.monotonic() + timeout

    while resp.is_open():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            resp.close()
            raise TimeoutError(
                f"Command timed out after {timeout}s in pod {pod_name!r}: {command[:80]!r}"
            )
        resp.update(timeout=min(1.0, remaining))
        if resp.peek_stdout():
            stdout_parts.append(resp.read_stdout())
        if resp.peek_stderr():
            stderr_parts.append(resp.read_stderr())

    exit_code: int = resp.returncode if resp.returncode is not None else -1
    output = "".join(stdout_parts) + "".join(stderr_parts)

    truncated = len(output) > _OUTPUT_LIMIT
    if truncated:
        output = output[:_OUTPUT_LIMIT]
        logger.debug("exec output truncated for pod %s", pod_name)

    return output, exit_code, truncated


# ---------------------------------------------------------------------------
# Tar-based file upload
# ---------------------------------------------------------------------------


def upload_files_tar(
    core_v1,
    pod_name: str,
    namespace: str,
    container: str,
    files: list[tuple[str, bytes]],
    timeout: int = 60,
) -> list[FileUploadResponse]:
    """Upload files to a Pod by piping a base64-encoded tar archive via stdin.

    The container must have ``base64`` and ``tar`` available (standard in most
    images).

    Args:
        core_v1: ``kubernetes.client.CoreV1Api`` instance.
        pod_name: Name of the target Pod.
        namespace: Namespace containing the Pod.
        container: Container to exec into.
        files: List of ``(absolute_path, content_bytes)`` tuples.
        timeout: Maximum seconds for the exec call.

    Returns:
        List of :class:`~deepagents.backends.protocol.FileUploadResponse`,
        one per input file.

    Raises:
        RuntimeError: If the tar command exits non-zero.
    """
    # Build an in-memory tar archive with all files.
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for path, content in files:
            info = tarfile.TarInfo(name=path.lstrip("/"))
            info.size = len(content)
            info.mode = 0o644
            tar.addfile(info, io.BytesIO(content))

    b64_data = base64.b64encode(buf.getvalue()).decode("ascii")

    # Pipe the base64 data to `base64 -d | tar xf - -C /`. The exec stream cannot signal EOF on
    # stdin, so `head -c` reads exactly the payload and ends the pipeline; without it the
    # command waits for input until the timeout.
    resp = stream(
        core_v1.connect_get_namespaced_pod_exec,
        name=pod_name,
        namespace=namespace,
        container=container,
        command=["/bin/sh", "-c", f"head -c {len(b64_data)} | base64 -d | tar xf - -C /"],
        stderr=True,
        stdin=True,
        stdout=True,
        _preload_content=False,
    )

    resp.write_stdin(b64_data)

    stderr_parts: list[str] = []
    deadline = time.monotonic() + timeout
    while resp.is_open():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            resp.close()
            raise TimeoutError(f"Upload timed out after {timeout}s in pod {pod_name!r}")
        resp.update(timeout=min(1.0, remaining))
        if resp.peek_stdout():
            resp.read_stdout()  # discard stdout
        if resp.peek_stderr():
            stderr_parts.append(resp.read_stderr())

    exit_code = resp.returncode if resp.returncode is not None else 0

    if exit_code != 0:
        stderr_msg = "".join(stderr_parts).strip()
        raise RuntimeError(
            f"tar upload failed in pod {pod_name!r} (exit {exit_code}): {stderr_msg}"
        )

    logger.debug("uploaded %d files to pod %s", len(files), pod_name)
    return [FileUploadResponse(path=path, error=None) for path, _ in files]


# ---------------------------------------------------------------------------
# Tar-based file download
# ---------------------------------------------------------------------------


def download_files_tar(
    core_v1,
    pod_name: str,
    namespace: str,
    container: str,
    paths: list[str],
    timeout: int = 60,
) -> list[FileDownloadResponse]:
    """Download files from a Pod by streaming a base64-encoded tar archive.

    Uses ``tar cf - <paths> | base64 -w 0`` so binary content is safely
    transferred through the UTF-8-decoded text channel of the exec API.

    Args:
        core_v1: ``kubernetes.client.CoreV1Api`` instance.
        pod_name: Name of the target Pod.
        namespace: Namespace containing the Pod.
        container: Container to exec into.
        paths: Absolute paths to download.
        timeout: Maximum seconds for the exec call.

    Returns:
        List of :class:`~deepagents.backends.protocol.FileDownloadResponse`,
        one per input path.  Files not found in the archive get
        ``error="file_not_found"``.
    """
    quoted_paths = " ".join(f'"{p}"' for p in paths)
    cmd = f"tar cf - {quoted_paths} 2>/dev/null | base64 -w 0"

    resp = stream(
        core_v1.connect_get_namespaced_pod_exec,
        name=pod_name,
        namespace=namespace,
        container=container,
        command=["/bin/sh", "-c", cmd],
        stderr=True,
        stdin=False,
        stdout=True,
        _preload_content=False,
    )

    b64_parts: list[str] = []
    deadline = time.monotonic() + timeout
    while resp.is_open():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            resp.close()
            raise TimeoutError(f"Download timed out after {timeout}s in pod {pod_name!r}")
        resp.update(timeout=min(1.0, remaining))
        if resp.peek_stdout():
            b64_parts.append(resp.read_stdout())
        if resp.peek_stderr():
            resp.read_stderr()  # discard stderr (tar errors for missing files)

    b64_output = "".join(b64_parts).strip()
    if not b64_output:
        logger.debug("no tar output for pod %s — all files missing?", pod_name)
        return [
            FileDownloadResponse(path=p, content=None, error="file_not_found") for p in paths
        ]

    try:
        tar_bytes = base64.b64decode(b64_output)
    except Exception as exc:
        logger.error("base64 decode failed for pod %s: %s", pod_name, exc)
        return [
            FileDownloadResponse(path=p, content=None, error="file_not_found") for p in paths
        ]

    results: list[FileDownloadResponse] = []
    try:
        with tarfile.open(fileobj=io.BytesIO(tar_bytes)) as tar:
            for path in paths:
                member_name = path.lstrip("/")
                try:
                    member = tar.getmember(member_name)
                    fobj = tar.extractfile(member)
                    content = fobj.read() if fobj else b""
                    results.append(FileDownloadResponse(path=path, content=content, error=None))
                except KeyError:
                    results.append(
                        FileDownloadResponse(path=path, content=None, error="file_not_found")
                    )
    except Exception as exc:
        logger.error("tar parse failed for pod %s: %s", pod_name, exc)
        return [
            FileDownloadResponse(path=p, content=None, error="file_not_found") for p in paths
        ]

    return results
