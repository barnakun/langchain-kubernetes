"""KubernetesSandbox: a BaseSandbox backed by a KubernetesBackendProtocol."""

from __future__ import annotations

import base64
import binascii
import logging
import posixpath
import shlex
from typing import TYPE_CHECKING

from deepagents.backends.protocol import (
    ExecuteResponse,
    FileDownloadResponse,
    FileOperationError,
    FileUploadResponse,
)
from deepagents.backends.sandbox import BaseSandbox

if TYPE_CHECKING:
    from langchain_kubernetes.backends.protocol import KubernetesBackendProtocol

logger = logging.getLogger(__name__)

# Base64 characters per execute call in the upload fallback; well under Linux's 128 KiB per argument.
_UPLOAD_CHUNK = 60_000


def _error_code(output: str) -> FileOperationError:
    """Map a failed shell command's output to deepagents' file error codes."""
    lowered = (output or "").lower()
    if "no such file" in lowered:
        return "file_not_found"
    if "permission denied" in lowered or "read-only file system" in lowered:
        return "permission_denied"
    if "is a directory" in lowered:
        return "is_directory"
    return "invalid_path"


class KubernetesSandbox(BaseSandbox):
    """A DeepAgents sandbox backed by a Kubernetes execution environment.

    Delegates all operations to a
    :class:`~langchain_kubernetes.backends.protocol.KubernetesBackendProtocol`
    implementation — either
    :class:`~langchain_kubernetes.backends.agent_sandbox.AgentSandboxBackend`
    (when ``mode="agent-sandbox"``) or
    :class:`~langchain_kubernetes.backends.raw.RawK8sBackend`
    (when ``mode="raw"``).

    The sandbox does not know which mode is active. Use
    :class:`~langchain_kubernetes.provider.KubernetesProvider` to create
    instances with the right backend.

    All filesystem helper methods (``read``, ``write``, ``edit``, ``ls_info``,
    ``glob_info``, ``grep_raw``) are inherited from
    :class:`~deepagents.backends.sandbox.BaseSandbox` — they work by
    constructing shell commands and calling :meth:`execute`.

    File upload and download first try the backend's native transfer mechanism.
    If the backend raises, they fall back to the
    :class:`~deepagents.backends.sandbox.BaseSandbox` base64-via-execute path.

    Args:
        backend: An active backend instance.
    """

    def __init__(self, *, backend: "KubernetesBackendProtocol") -> None:
        self._backend = backend

    # ------------------------------------------------------------------
    # BaseSandbox: id
    # ------------------------------------------------------------------

    @property
    def id(self) -> str:
        """Unique sandbox identifier delegated to the backend.

        Returns:
            Sandbox ID string (Sandbox CR name for agent-sandbox mode,
            short hex ID for raw mode).
        """
        return self._backend.id

    # ------------------------------------------------------------------
    # BaseSandbox: execute (sync)
    # ------------------------------------------------------------------

    def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        """Execute *command* inside the sandbox and return the result.

        Args:
            command: Shell command string.
            timeout: Per-call timeout in seconds.

        Returns:
            :class:`~deepagents.backends.protocol.ExecuteResponse`.
        """
        return self._backend.execute(command, timeout=timeout)

    # ------------------------------------------------------------------
    # BaseSandbox: execute (async)
    # ------------------------------------------------------------------

    async def aexecute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        """Async variant of :meth:`execute`.

        Args:
            command: Shell command string.
            timeout: Per-call timeout in seconds.

        Returns:
            :class:`~deepagents.backends.protocol.ExecuteResponse`.
        """
        return await self._backend.aexecute(command, timeout=timeout)

    # ------------------------------------------------------------------
    # BaseSandbox: file transfer
    # ------------------------------------------------------------------

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """Upload files, falling back to base64-via-execute on error.

        Tries the backend's native upload mechanism first. If the backend raises
        an exception, each file is written through :meth:`execute` instead, in
        base64 chunks (deepagents' ``BaseSandbox.upload_files`` is abstract).

        Args:
            files: List of ``(absolute_path, content_bytes)`` tuples.

        Returns:
            List of :class:`~deepagents.backends.protocol.FileUploadResponse`.
        """
        try:
            return self._backend.upload_files(files)
        except Exception as exc:
            logger.warning(
                "Backend upload failed (%s), falling back to base64-via-execute", exc
            )
            return [FileUploadResponse(path=path, error=self._upload_via_execute(path, content))
                    for path, content in files]

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Download files, falling back to base64-via-execute on error.

        Tries the backend's native download mechanism first. If the backend
        raises, each file is read through :meth:`execute` as base64 instead
        (deepagents' ``BaseSandbox.download_files`` is abstract).

        Args:
            paths: Absolute paths to download from the sandbox.

        Returns:
            List of :class:`~deepagents.backends.protocol.FileDownloadResponse`.
        """
        try:
            return self._backend.download_files(paths)
        except Exception as exc:
            logger.warning(
                "Backend download failed (%s), falling back to base64-via-execute", exc
            )
            return [self._download_via_execute(path) for path in paths]

    def _upload_via_execute(self, path: str, content: bytes) -> FileOperationError | None:
        """Write one file through ``execute`` in base64 chunks; the error code, or None."""
        encoded = base64.b64encode(content).decode("ascii")
        chunks = [encoded[i : i + _UPLOAD_CHUNK] for i in range(0, len(encoded), _UPLOAD_CHUNK)] or [""]
        target = shlex.quote(path)
        for index, chunk in enumerate(chunks):
            prefix = f"mkdir -p {shlex.quote(posixpath.dirname(path) or '/')} && " if index == 0 else ""
            redirect = ">" if index == 0 else ">>"
            result = self.execute(f"{prefix}printf %s '{chunk}' | base64 -d {redirect} {target}")
            if result.exit_code != 0:
                return _error_code(result.output)
        return None

    def _download_via_execute(self, path: str) -> FileDownloadResponse:
        """Read one file through ``execute`` as base64.

        The checks run first because ``base64 … | tr`` exits with ``tr``'s status, which would
        hide a failed read; output that is not base64 is treated as an error, never decoded.
        """
        p = shlex.quote(path)
        result = self.execute(
            f"if [ ! -e {p} ]; then echo 'No such file or directory'; exit 1; fi; "
            f"if [ -d {p} ]; then echo 'Is a directory'; exit 1; fi; "
            f"if [ ! -r {p} ]; then echo 'Permission denied'; exit 1; fi; "
            f"base64 {p} | tr -d '\\n'"
        )
        if result.exit_code != 0:
            return FileDownloadResponse(path=path, content=None, error=_error_code(result.output))
        if result.truncated:
            logger.warning("Download of %s was truncated by the execute output limit", path)
            return FileDownloadResponse(path=path, content=None, error="invalid_path")
        try:
            content = base64.b64decode(result.output.strip(), validate=True)
        except binascii.Error:
            return FileDownloadResponse(path=path, content=None, error=_error_code(result.output))
        return FileDownloadResponse(path=path, content=content, error=None)
