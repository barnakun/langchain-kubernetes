"""Unit tests for raw_transport — exec and tar file transfer functions."""

from __future__ import annotations

import base64
import io
import tarfile
from unittest.mock import MagicMock, call, patch

import pytest

from deepagents.backends.protocol import FileDownloadResponse, FileUploadResponse
from langchain_kubernetes.backends.raw_transport import (
    download_files_tar,
    exec_command,
    upload_files_tar,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_stream_resp(
    stdout_chunks: list[str] | None = None,
    stderr_chunks: list[str] | None = None,
    returncode: int = 0,
) -> MagicMock:
    """Build a mock WSClient-like response object."""
    stdout_chunks = stdout_chunks or []
    stderr_chunks = stderr_chunks or []

    call_count = 0
    total_cycles = max(len(stdout_chunks), len(stderr_chunks), 1)

    resp = MagicMock()
    resp.returncode = returncode

    # Track open state: open for total_cycles, then closed
    open_states = [True] * total_cycles + [False]
    resp.is_open.side_effect = lambda: open_states.pop(0) if open_states else False

    # peek_stdout / peek_stderr return True once per chunk
    stdout_iter = iter(stdout_chunks)
    stderr_iter = iter(stderr_chunks)

    peek_stdout_states = [bool(c) for c in stdout_chunks] + [False]
    peek_stderr_states = [bool(c) for c in stderr_chunks] + [False]

    resp.peek_stdout.side_effect = lambda: (
        peek_stdout_states.pop(0) if peek_stdout_states else False
    )
    resp.peek_stderr.side_effect = lambda: (
        peek_stderr_states.pop(0) if peek_stderr_states else False
    )
    resp.read_stdout.side_effect = lambda: next(stdout_iter, "")
    resp.read_stderr.side_effect = lambda: next(stderr_iter, "")

    return resp


def _make_tar_b64(files: list[tuple[str, bytes]]) -> str:
    """Build a base64-encoded tar archive from a list of (path, content) tuples."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for path, content in files:
            info = tarfile.TarInfo(name=path.lstrip("/"))
            info.size = len(content)
            tar.addfile(info, io.BytesIO(content))
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ---------------------------------------------------------------------------
# exec_command
# ---------------------------------------------------------------------------


class TestExecCommand:
    @patch("langchain_kubernetes.backends.raw_transport.stream")
    def test_returns_stdout(self, mock_stream):
        resp = _make_stream_resp(stdout_chunks=["hello world\n"])
        mock_stream.return_value = resp

        output, exit_code, truncated = exec_command(
            MagicMock(), "pod1", "default", "sandbox", "echo hello", timeout=5
        )

        assert "hello world" in output
        assert exit_code == 0
        assert truncated is False

    @patch("langchain_kubernetes.backends.raw_transport.stream")
    def test_combines_stdout_and_stderr(self, mock_stream):
        resp = _make_stream_resp(
            stdout_chunks=["out\n"],
            stderr_chunks=["err\n"],
            returncode=1,
        )
        mock_stream.return_value = resp

        output, exit_code, _ = exec_command(
            MagicMock(), "pod1", "default", "sandbox", "cmd", timeout=5
        )

        assert "out" in output
        assert "err" in output
        assert exit_code == 1

    @patch("langchain_kubernetes.backends.raw_transport.stream")
    def test_nonzero_exit_code(self, mock_stream):
        resp = _make_stream_resp(returncode=127)
        mock_stream.return_value = resp

        _, exit_code, _ = exec_command(
            MagicMock(), "pod1", "default", "sandbox", "bad-cmd", timeout=5
        )

        assert exit_code == 127

    @patch("langchain_kubernetes.backends.raw_transport.stream")
    def test_truncates_long_output(self, mock_stream):
        big_output = "x" * (1024 * 1024 + 100)
        resp = _make_stream_resp(stdout_chunks=[big_output])
        mock_stream.return_value = resp

        output, _, truncated = exec_command(
            MagicMock(), "pod1", "default", "sandbox", "cmd", timeout=5
        )

        assert truncated is True
        assert len(output) == 1024 * 1024

    @patch("langchain_kubernetes.backends.raw_transport.stream")
    def test_stream_called_with_sh_c(self, mock_stream):
        resp = _make_stream_resp()
        mock_stream.return_value = resp

        exec_command(MagicMock(), "pod1", "ns", "sandbox", "echo hi", timeout=5)

        call_kwargs = mock_stream.call_args
        assert call_kwargs[1]["command"] == ["/bin/sh", "-c", "echo hi"]
        assert call_kwargs[1]["stdout"] is True
        assert call_kwargs[1]["stderr"] is True
        assert call_kwargs[1]["stdin"] is False

    @patch("langchain_kubernetes.backends.raw_transport.stream")
    def test_stream_passes_pod_and_namespace(self, mock_stream):
        resp = _make_stream_resp()
        mock_stream.return_value = resp

        exec_command(MagicMock(), "my-pod", "my-ns", "sandbox", "ls", timeout=5)

        call_kwargs = mock_stream.call_args
        assert call_kwargs[1]["name"] == "my-pod"
        assert call_kwargs[1]["namespace"] == "my-ns"


# ---------------------------------------------------------------------------
# upload_files_tar
# ---------------------------------------------------------------------------


class TestUploadFilesTar:
    def _make_upload_resp(self, returncode: int = 0) -> MagicMock:
        resp = MagicMock()
        resp.returncode = returncode
        resp.is_open.side_effect = [True, False]
        resp.peek_stdout.return_value = False
        resp.peek_stderr.return_value = False
        return resp

    @patch("langchain_kubernetes.backends.raw_transport.stream")
    def test_returns_success_response(self, mock_stream):
        mock_stream.return_value = self._make_upload_resp()

        results = upload_files_tar(
            MagicMock(), "pod1", "default", "sandbox",
            [("/tmp/a.txt", b"hello")],
            timeout=5,
        )

        assert len(results) == 1
        assert results[0].path == "/tmp/a.txt"
        assert results[0].error is None

    @patch("langchain_kubernetes.backends.raw_transport.stream")
    def test_multiple_files_in_one_call(self, mock_stream):
        mock_stream.return_value = self._make_upload_resp()

        files = [("/a.txt", b"a"), ("/b.txt", b"b"), ("/c.txt", b"c")]
        results = upload_files_tar(
            MagicMock(), "pod1", "default", "sandbox", files, timeout=5
        )

        # Only ONE exec call for all files (single tar archive)
        mock_stream.assert_called_once()
        assert len(results) == 3

    @patch("langchain_kubernetes.backends.raw_transport.stream")
    def test_uses_base64_decode_command(self, mock_stream):
        mock_stream.return_value = self._make_upload_resp()

        upload_files_tar(
            MagicMock(), "pod1", "default", "sandbox",
            [("/a.txt", b"data")], timeout=5
        )

        cmd = mock_stream.call_args[1]["command"]
        assert "base64 -d" in " ".join(cmd)
        assert "tar" in " ".join(cmd)

    @patch("langchain_kubernetes.backends.raw_transport.stream")
    def test_reads_exactly_the_payload_so_no_eof_is_needed(self, mock_stream):
        """The exec stream cannot signal EOF on stdin: `base64 -d` must stop after the payload, not wait."""
        resp = self._make_upload_resp()
        mock_stream.return_value = resp

        upload_files_tar(MagicMock(), "pod1", "default", "sandbox", [("/a.txt", b"data" * 100)], timeout=5)

        payload = resp.write_stdin.call_args[0][0]
        script = mock_stream.call_args[1]["command"][-1]
        assert script.startswith(f"head -c {len(payload)} |")

    @patch("langchain_kubernetes.backends.raw_transport.stream")
    def test_stdin_enabled(self, mock_stream):
        mock_stream.return_value = self._make_upload_resp()

        upload_files_tar(
            MagicMock(), "pod1", "default", "sandbox",
            [("/a.txt", b"data")], timeout=5
        )

        assert mock_stream.call_args[1]["stdin"] is True

    @patch("langchain_kubernetes.backends.raw_transport.stream")
    def test_raises_on_nonzero_exit(self, mock_stream):
        resp = self._make_upload_resp(returncode=1)
        resp.peek_stderr.side_effect = [True, False]
        resp.read_stderr.return_value = "tar: error"
        mock_stream.return_value = resp

        with pytest.raises(RuntimeError, match="tar upload failed"):
            upload_files_tar(
                MagicMock(), "pod1", "default", "sandbox",
                [("/a.txt", b"data")], timeout=5
            )

    @patch("langchain_kubernetes.backends.raw_transport.stream")
    def test_writes_base64_to_stdin(self, mock_stream):
        resp = self._make_upload_resp()
        mock_stream.return_value = resp

        upload_files_tar(
            MagicMock(), "pod1", "default", "sandbox",
            [("/a.txt", b"hello")], timeout=5
        )

        # write_stdin must be called with a non-empty string
        resp.write_stdin.assert_called_once()
        written = resp.write_stdin.call_args[0][0]
        assert isinstance(written, str)
        assert len(written) > 0


# ---------------------------------------------------------------------------
# download_files_tar
# ---------------------------------------------------------------------------


class TestDownloadFilesTar:
    def _make_download_resp(
        self, b64_content: str, returncode: int = 0
    ) -> MagicMock:
        resp = MagicMock()
        resp.returncode = returncode
        resp.is_open.side_effect = [True, False]
        resp.peek_stdout.side_effect = [True, False]
        resp.peek_stderr.return_value = False
        resp.read_stdout.return_value = b64_content
        return resp

    @patch("langchain_kubernetes.backends.raw_transport.stream")
    def test_returns_file_content(self, mock_stream):
        b64 = _make_tar_b64([("/workspace/a.txt", b"file content")])
        mock_stream.return_value = self._make_download_resp(b64)

        results = download_files_tar(
            MagicMock(), "pod1", "default", "sandbox",
            ["/workspace/a.txt"], timeout=5
        )

        assert len(results) == 1
        assert results[0].path == "/workspace/a.txt"
        assert results[0].content == b"file content"
        assert results[0].error is None

    @patch("langchain_kubernetes.backends.raw_transport.stream")
    def test_multiple_files_downloaded(self, mock_stream):
        b64 = _make_tar_b64([
            ("/a.txt", b"aaa"),
            ("/b.txt", b"bbb"),
        ])
        mock_stream.return_value = self._make_download_resp(b64)

        results = download_files_tar(
            MagicMock(), "pod1", "default", "sandbox",
            ["/a.txt", "/b.txt"], timeout=5
        )

        content_map = {r.path: r.content for r in results}
        assert content_map["/a.txt"] == b"aaa"
        assert content_map["/b.txt"] == b"bbb"

    @patch("langchain_kubernetes.backends.raw_transport.stream")
    def test_missing_file_returns_file_not_found(self, mock_stream):
        # Tar contains only a.txt, but b.txt is requested
        b64 = _make_tar_b64([("/a.txt", b"a")])
        mock_stream.return_value = self._make_download_resp(b64)

        results = download_files_tar(
            MagicMock(), "pod1", "default", "sandbox",
            ["/a.txt", "/b.txt"], timeout=5
        )

        errors = {r.path: r.error for r in results}
        assert errors["/a.txt"] is None
        assert errors["/b.txt"] == "file_not_found"

    @patch("langchain_kubernetes.backends.raw_transport.stream")
    def test_empty_output_returns_all_not_found(self, mock_stream):
        resp = MagicMock()
        resp.is_open.side_effect = [True, False]
        resp.peek_stdout.return_value = False
        resp.peek_stderr.return_value = False
        mock_stream.return_value = resp

        results = download_files_tar(
            MagicMock(), "pod1", "default", "sandbox",
            ["/missing.txt"], timeout=5
        )

        assert results[0].error == "file_not_found"

    @patch("langchain_kubernetes.backends.raw_transport.stream")
    def test_uses_base64_encoding_command(self, mock_stream):
        b64 = _make_tar_b64([("/a.txt", b"x")])
        mock_stream.return_value = self._make_download_resp(b64)

        download_files_tar(
            MagicMock(), "pod1", "default", "sandbox",
            ["/a.txt"], timeout=5
        )

        cmd = mock_stream.call_args[1]["command"]
        cmd_str = " ".join(cmd)
        assert "tar" in cmd_str
        assert "base64" in cmd_str

    @patch("langchain_kubernetes.backends.raw_transport.stream")
    def test_binary_content_preserved(self, mock_stream):
        binary_data = bytes(range(256))
        b64 = _make_tar_b64([("/bin/data", binary_data)])
        mock_stream.return_value = self._make_download_resp(b64)

        results = download_files_tar(
            MagicMock(), "pod1", "default", "sandbox",
            ["/bin/data"], timeout=5
        )

        assert results[0].content == binary_data
