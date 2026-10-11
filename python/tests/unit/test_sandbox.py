"""Unit tests for KubernetesSandbox — mocks the backend protocol."""

from __future__ import annotations

import base64

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from deepagents.backends.protocol import (
    ExecuteResponse,
    FileDownloadResponse,
    FileUploadResponse,
)
from langchain_kubernetes.sandbox import KubernetesSandbox


def _make_execute_response(output: str = "", exit_code: int = 0, truncated: bool = False):
    return ExecuteResponse(output=output, exit_code=exit_code, truncated=truncated)


def _make_sandbox(sandbox_id: str = "test-sandbox-abc123"):
    """Return (KubernetesSandbox, mock_backend)."""
    backend = MagicMock()
    backend.id = sandbox_id
    backend.execute.return_value = _make_execute_response()
    backend.aexecute = AsyncMock(return_value=_make_execute_response())
    backend.upload_files.return_value = []
    backend.download_files.return_value = []
    return KubernetesSandbox(backend=backend), backend


# ---------------------------------------------------------------------------
# id
# ---------------------------------------------------------------------------


class TestKubernetesSandboxId:
    def test_id_delegates_to_backend(self):
        sb, _ = _make_sandbox("my-sandbox-001")
        assert sb.id == "my-sandbox-001"


# ---------------------------------------------------------------------------
# execute
# ---------------------------------------------------------------------------


class TestKubernetesSandboxExecute:
    def test_execute_delegates_to_backend(self):
        sb, backend = _make_sandbox()
        backend.execute.return_value = _make_execute_response(output="hello\n", exit_code=0)

        result = sb.execute("echo hello")

        backend.execute.assert_called_once_with("echo hello", timeout=None)
        assert result.output == "hello\n"
        assert result.exit_code == 0
        assert result.truncated is False

    def test_execute_passes_timeout(self):
        sb, backend = _make_sandbox()
        backend.execute.return_value = _make_execute_response()

        sb.execute("cmd", timeout=42)

        backend.execute.assert_called_once_with("cmd", timeout=42)

    def test_execute_nonzero_exit_code(self):
        sb, backend = _make_sandbox()
        backend.execute.return_value = _make_execute_response(exit_code=127)

        result = sb.execute("bad-cmd")
        assert result.exit_code == 127

    def test_execute_truncated(self):
        sb, backend = _make_sandbox()
        backend.execute.return_value = _make_execute_response(
            output="x" * 100, truncated=True
        )

        result = sb.execute("big-cmd")
        assert result.truncated is True


# ---------------------------------------------------------------------------
# aexecute
# ---------------------------------------------------------------------------


class TestKubernetesSandboxAexecute:
    @pytest.mark.asyncio
    async def test_aexecute_delegates_to_backend(self):
        sb, backend = _make_sandbox()
        backend.aexecute = AsyncMock(
            return_value=_make_execute_response(output="async result", exit_code=0)
        )

        result = await sb.aexecute("echo async")

        backend.aexecute.assert_awaited_once_with("echo async", timeout=None)
        assert result.output == "async result"

    @pytest.mark.asyncio
    async def test_aexecute_passes_timeout(self):
        sb, backend = _make_sandbox()
        backend.aexecute = AsyncMock(return_value=_make_execute_response())

        await sb.aexecute("cmd", timeout=99)

        backend.aexecute.assert_awaited_once_with("cmd", timeout=99)


# ---------------------------------------------------------------------------
# upload_files
# ---------------------------------------------------------------------------


class TestKubernetesSandboxUploadFiles:
    def test_upload_delegates_to_backend(self):
        sb, backend = _make_sandbox()
        backend.upload_files.return_value = [
            FileUploadResponse(path="/tmp/a.txt", error=None)
        ]

        result = sb.upload_files([("/tmp/a.txt", b"hello")])

        backend.upload_files.assert_called_once_with([("/tmp/a.txt", b"hello")])
        assert result[0].path == "/tmp/a.txt"
        assert result[0].error is None

    def test_upload_falls_back_to_base64_via_execute(self):
        """deepagents' BaseSandbox.upload_files is abstract: the fallback must do the work itself."""
        sb, backend = _make_sandbox()
        backend.upload_files.side_effect = RuntimeError("backend failed")
        backend.execute.return_value = ExecuteResponse(output="", exit_code=0, truncated=False)

        result = sb.upload_files([("/tmp/dir/a.txt", b"data")])

        assert [(r.path, r.error) for r in result] == [("/tmp/dir/a.txt", None)]
        script = " ".join(c.args[0] for c in backend.execute.call_args_list)
        assert "mkdir -p /tmp/dir" in script and "base64 -d > /tmp/dir/a.txt" in script
        assert base64.b64encode(b"data").decode() in script

    def test_upload_fallback_writes_large_files_in_chunks(self):
        sb, backend = _make_sandbox()
        backend.upload_files.side_effect = RuntimeError("backend failed")
        backend.execute.return_value = ExecuteResponse(output="", exit_code=0, truncated=False)

        sb.upload_files([("/tmp/big.bin", b"x" * 200_000)])

        writes = [c.args[0] for c in backend.execute.call_args_list if "base64 -d" in c.args[0]]
        assert len(writes) > 1
        assert all(len(w) < 100_000 for w in writes)
        assert "> /tmp/big.bin" in writes[0] and all(">> /tmp/big.bin" in w for w in writes[1:])

    def test_upload_fallback_reports_failures_per_file(self):
        sb, backend = _make_sandbox()
        backend.upload_files.side_effect = RuntimeError("backend failed")
        backend.execute.side_effect = lambda cmd, **kw: ExecuteResponse(
            output="Permission denied" if "/root/" in cmd else "", exit_code=1 if "/root/" in cmd else 0,
            truncated=False,
        )

        result = sb.upload_files([("/root/x.txt", b"a"), ("/tmp/y.txt", b"b")])

        assert [(r.path, r.error) for r in result] == [("/root/x.txt", "permission_denied"), ("/tmp/y.txt", None)]

    def test_download_falls_back_to_base64_via_execute(self):
        sb, backend = _make_sandbox()
        backend.download_files.side_effect = RuntimeError("backend failed")
        backend.execute.side_effect = lambda cmd, **kw: (
            ExecuteResponse(output=base64.b64encode(b"hello").decode(), exit_code=0, truncated=False)
            if "/tmp/a.txt" in cmd
            else ExecuteResponse(output="No such file or directory", exit_code=1, truncated=False)
        )

        result = sb.download_files(["/tmp/a.txt", "/tmp/missing.txt"])

        assert [(r.path, r.content, r.error) for r in result] == [
            ("/tmp/a.txt", b"hello", None), ("/tmp/missing.txt", None, "file_not_found")
        ]

    def test_upload_multiple_files(self):
        sb, backend = _make_sandbox()
        files = [("/a.txt", b"a"), ("/b.txt", b"b")]
        backend.upload_files.return_value = [
            FileUploadResponse(path=p, error=None) for p, _ in files
        ]

        results = sb.upload_files(files)

        backend.upload_files.assert_called_once_with(files)
        assert len(results) == 2


# ---------------------------------------------------------------------------
# download_files
# ---------------------------------------------------------------------------


class TestKubernetesSandboxDownloadFiles:
    def test_download_fallback_treats_undecodable_output_as_an_error(self):
        """A pipe can hide a failed read (exit 0, error text on the output): never decode garbage."""
        sb, backend = _make_sandbox()
        backend.download_files.side_effect = RuntimeError("backend failed")
        backend.execute.return_value = ExecuteResponse(
            output="base64: /tmp/x: No such file or directory", exit_code=0, truncated=False
        )

        result = sb.download_files(["/tmp/x"])

        assert (result[0].content, result[0].error) == (None, "file_not_found")

    def test_download_fallback_checks_the_path_before_reading(self):
        sb, backend = _make_sandbox()
        backend.download_files.side_effect = RuntimeError("backend failed")
        backend.execute.return_value = ExecuteResponse(output="aGk=", exit_code=0, truncated=False)

        sb.download_files(["/tmp/a b.txt"])

        script = backend.execute.call_args.args[0]
        assert "[ ! -e '/tmp/a b.txt' ]" in script and "[ ! -r '/tmp/a b.txt' ]" in script

    def test_download_delegates_to_backend(self):
        sb, backend = _make_sandbox()
        backend.download_files.return_value = [
            FileDownloadResponse(path="/tmp/a.txt", content=b"content", error=None)
        ]

        result = sb.download_files(["/tmp/a.txt"])

        backend.download_files.assert_called_once_with(["/tmp/a.txt"])
        assert result[0].content == b"content"
        assert result[0].error is None

    def test_download_multiple_files(self):
        sb, backend = _make_sandbox()
        backend.download_files.return_value = [
            FileDownloadResponse(path="/a.txt", content=b"a", error=None),
            FileDownloadResponse(path="/b.txt", content=b"b", error=None),
        ]

        results = sb.download_files(["/a.txt", "/b.txt"])

        assert len(results) == 2
        assert results[0].content == b"a"
        assert results[1].content == b"b"
