"""Unit tests for KubernetesProvider — mocks backends."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from langchain_kubernetes._provider_base import SandboxNotFoundError
from langchain_kubernetes._types import SandboxInfo, SandboxListResponse
from langchain_kubernetes.config import KubernetesProviderConfig
from langchain_kubernetes.provider import (
    KubernetesProvider,
    _build_agent_sandbox_client,
    _import_sandbox_client,
)
from langchain_kubernetes.sandbox import KubernetesSandbox


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def _agent_sandbox_config(**kwargs) -> KubernetesProviderConfig:
    defaults = {"mode": "agent-sandbox", "template_name": "test-template"}
    defaults.update(kwargs)
    return KubernetesProviderConfig(**defaults)


def _raw_config(**kwargs) -> KubernetesProviderConfig:
    defaults = {"mode": "raw"}
    defaults.update(kwargs)
    return KubernetesProviderConfig(**defaults)


def _make_mock_sandbox_client(sandbox_name: str = "test-sandbox-abc") -> MagicMock:
    client = MagicMock()
    client.sandbox_name = sandbox_name
    client.claim_name = None
    client.__enter__ = MagicMock(return_value=client)
    client.__exit__ = MagicMock(return_value=False)
    return client


def _make_mock_backend(sandbox_id: str = "test-backend-001") -> MagicMock:
    backend = MagicMock()
    backend.id = sandbox_id
    return backend


# ---------------------------------------------------------------------------
# agent-sandbox mode: get_or_create
# ---------------------------------------------------------------------------


class TestGetOrCreateAgentSandbox:
    def test_creates_agent_sandbox_backend(self):
        config = _agent_sandbox_config()
        provider = KubernetesProvider(config)
        mock_client = _make_mock_sandbox_client("sandbox-xyz")

        with patch("langchain_kubernetes.provider._build_agent_sandbox_client", return_value=mock_client):
            sandbox = provider.get_or_create()

        mock_client.__enter__.assert_called_once()
        assert isinstance(sandbox, KubernetesSandbox)
        assert sandbox.id == "sandbox-xyz"

    def test_sandbox_id_from_client_sandbox_name(self):
        provider = KubernetesProvider(_agent_sandbox_config())
        mock_client = _make_mock_sandbox_client("my-sandbox-001")

        with patch("langchain_kubernetes.provider._build_agent_sandbox_client", return_value=mock_client):
            sandbox = provider.get_or_create()

        assert sandbox.id == "my-sandbox-001"

    def test_falls_back_to_claim_name(self):
        provider = KubernetesProvider(_agent_sandbox_config())
        mock_client = _make_mock_sandbox_client()
        mock_client.sandbox_name = None
        mock_client.claim_name = "claim-abc"

        with patch("langchain_kubernetes.provider._build_agent_sandbox_client", return_value=mock_client):
            sandbox = provider.get_or_create()

        assert sandbox.id == "claim-abc"

    def test_reconnect_to_active_sandbox(self):
        """get_or_create(sandbox_id=...) reconnects and returns sandbox with correct id."""
        provider = KubernetesProvider(_agent_sandbox_config())
        mock_client = _make_mock_sandbox_client("active-sandbox")

        with patch("langchain_kubernetes.provider._build_agent_sandbox_client", return_value=mock_client):
            sb = provider.get_or_create(sandbox_id="active-sandbox")

        assert sb.id == "active-sandbox"

    def test_reconnect_failure_falls_through_to_new_sandbox(self):
        """When reconnect raises SandboxNotFoundError, a new sandbox is provisioned."""
        provider = KubernetesProvider(_agent_sandbox_config())
        new_client = _make_mock_sandbox_client("new-sandbox")

        with patch.object(provider, "reconnect", side_effect=SandboxNotFoundError("gone")):
            with patch("langchain_kubernetes.provider._build_agent_sandbox_client", return_value=new_client):
                sb = provider.get_or_create(sandbox_id="stale-id")

        assert sb.id == "new-sandbox"


# ---------------------------------------------------------------------------
# raw mode: get_or_create
# ---------------------------------------------------------------------------


class TestGetOrCreateRaw:
    def test_creates_raw_backend(self):
        provider = KubernetesProvider(_raw_config())
        mock_backend = _make_mock_backend("rawid001")

        with patch("langchain_kubernetes.provider.KubernetesProvider._create_raw_backend", return_value=mock_backend):
            sandbox = provider.get_or_create()

        assert isinstance(sandbox, KubernetesSandbox)
        assert sandbox.id == "rawid001"


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------


class TestList:
    def test_list_empty_initially(self):
        provider = KubernetesProvider(_agent_sandbox_config())
        result = provider.list()
        assert result.sandboxes == []

    def test_list_delegates_to_agent_sandbox_listing(self):
        """list() returns what _list_agent_sandbox returns."""
        provider = KubernetesProvider(_agent_sandbox_config())
        mock_result = SandboxListResponse(sandboxes=[
            SandboxInfo(id="sb-001", namespace="default", status="running"),
        ])

        with patch.object(provider, "_list_agent_sandbox", return_value=mock_result):
            result = provider.list()

        assert len(result.sandboxes) == 1
        assert result.sandboxes[0].id == "sb-001"

    def test_list_returns_multiple_sandboxes(self):
        provider = KubernetesProvider(_agent_sandbox_config())
        sandboxes = [
            SandboxInfo(id=f"sb-{i:03d}", namespace="default", status="running")
            for i in range(3)
        ]
        mock_result = SandboxListResponse(sandboxes=sandboxes)

        with patch.object(provider, "_list_agent_sandbox", return_value=mock_result):
            result = provider.list()

        assert len(result.sandboxes) == 3


# ---------------------------------------------------------------------------
# delete
# ---------------------------------------------------------------------------


class TestDelete:
    def test_delete_calls_agent_sandbox_delete(self):
        """delete() calls _delete_agent_sandbox_claim in agent-sandbox mode."""
        provider = KubernetesProvider(_agent_sandbox_config())

        with patch.object(provider, "_delete_agent_sandbox_claim") as mock_del:
            provider.delete(sandbox_id="sb-del")
            mock_del.assert_called_once_with("sb-del")

    def test_delete_nonexistent_is_noop(self):
        provider = KubernetesProvider(_agent_sandbox_config())
        provider.delete(sandbox_id="does-not-exist")  # must not raise

    def test_delete_raw_calls_pod_delete(self):
        """delete() calls _delete_raw_pod in raw mode."""
        provider = KubernetesProvider(_raw_config())

        with patch.object(provider, "_delete_raw_pod") as mock_del:
            provider.delete(sandbox_id="rawid")
            mock_del.assert_called_once_with("rawid", "default")

    def test_delete_raw_triggers_warm_pool_replenish(self):
        """delete() schedules warm pool replenishment in raw mode when pool is enabled."""
        provider = KubernetesProvider(_raw_config(warm_pool_size=2))

        with patch.object(provider, "_delete_raw_pod"):
            with patch.object(provider, "_schedule_replenish") as mock_replenish:
                provider.delete(sandbox_id="any-pod")
                mock_replenish.assert_called_once()


# ---------------------------------------------------------------------------
# agent-sandbox client builder
# ---------------------------------------------------------------------------


class TestBuildAgentSandboxClient:
    def test_tunnel_mode_passes_correct_kwargs(self):
        config = _agent_sandbox_config(
            template_name="tpl",
            namespace="ns",
            connection_mode="tunnel",
            server_port=9000,
            startup_timeout_seconds=60,
        )

        with patch("langchain_kubernetes.provider._import_sandbox_client") as mock_import:
            mock_cls = MagicMock()
            mock_import.return_value = mock_cls
            _build_agent_sandbox_client(config)

        mock_cls.assert_called_once_with(
            template_name="tpl",
            namespace="ns",
            server_port=9000,
            sandbox_ready_timeout=60,
        )

    def test_gateway_mode_requires_gateway_name(self):
        config = _agent_sandbox_config(connection_mode="gateway", gateway_name=None)

        with patch("langchain_kubernetes.provider._import_sandbox_client"):
            with pytest.raises(ValueError, match="gateway_name"):
                _build_agent_sandbox_client(config)

    def test_direct_mode_requires_api_url(self):
        config = _agent_sandbox_config(connection_mode="direct", api_url=None)

        with patch("langchain_kubernetes.provider._import_sandbox_client"):
            with pytest.raises(ValueError, match="api_url"):
                _build_agent_sandbox_client(config)

    def test_missing_sdk_raises_import_error(self):
        provider = KubernetesProvider(_agent_sandbox_config())

        with patch(
            "langchain_kubernetes.provider._import_sandbox_client",
            side_effect=ImportError("k8s-agent-sandbox not installed"),
        ):
            with pytest.raises(ImportError, match="k8s-agent-sandbox"):
                provider.get_or_create()


# ---------------------------------------------------------------------------
# async interface
# ---------------------------------------------------------------------------


class TestAsync:
    @pytest.mark.asyncio
    async def test_aget_or_create(self):
        provider = KubernetesProvider(_agent_sandbox_config())
        mock_client = _make_mock_sandbox_client("async-sb")

        with patch("langchain_kubernetes.provider._build_agent_sandbox_client", return_value=mock_client):
            sandbox = await provider.aget_or_create()

        assert sandbox.id == "async-sb"

    @pytest.mark.asyncio
    async def test_adelete(self):
        """adelete() calls _delete_agent_sandbox_claim for agent-sandbox mode."""
        provider = KubernetesProvider(_agent_sandbox_config())

        with patch.object(provider, "_delete_agent_sandbox_claim") as mock_del:
            await provider.adelete(sandbox_id="async-del")
            mock_del.assert_called_once_with("async-del")

    @pytest.mark.asyncio
    async def test_alist(self):
        provider = KubernetesProvider(_agent_sandbox_config())
        mock_result = SandboxListResponse(sandboxes=[
            SandboxInfo(id="async-list-sb", namespace="default", status="running"),
        ])

        with patch.object(provider, "_list_agent_sandbox", return_value=mock_result):
            result = await provider.alist()

        assert len(result.sandboxes) == 1
        assert result.sandboxes[0].id == "async-list-sb"


# ---------------------------------------------------------------------------
# mode dispatch
# ---------------------------------------------------------------------------


class TestModeDispatch:
    def test_agent_sandbox_mode_calls_agent_sandbox_backend(self):
        provider = KubernetesProvider(_agent_sandbox_config())

        with patch.object(provider, "_create_agent_sandbox_backend") as mock_create:
            mock_backend = _make_mock_backend("asb-001")
            mock_create.return_value = mock_backend
            provider.get_or_create()

        mock_create.assert_called_once()

    def test_raw_mode_calls_raw_backend(self):
        provider = KubernetesProvider(_raw_config())

        with patch.object(provider, "_create_raw_backend") as mock_create:
            mock_backend = _make_mock_backend("raw-001")
            mock_create.return_value = mock_backend
            provider.get_or_create()

        mock_create.assert_called_once()


class TestDeleteRawPodRemovesNetworkPolicy:
    """provider.delete() in raw mode must not leave the per-sandbox deny-all policy behind."""

    def _clients(self):
        core_v1, networking_v1 = MagicMock(), MagicMock()
        return core_v1, networking_v1

    def test_deletes_pod_and_its_network_policy(self):
        provider = KubernetesProvider(_raw_config())
        core_v1, networking_v1 = self._clients()
        with patch("langchain_kubernetes.backends.raw.RawK8sBackend.load_k8s_clients",
                   return_value=(core_v1, networking_v1)):
            provider._delete_raw_pod("abc123", "default")

        core_v1.delete_namespaced_pod.assert_called_once_with(name="deepagents-abc123", namespace="default")
        networking_v1.delete_namespaced_network_policy.assert_called_once_with(
            name="deepagents-deny-all-abc123", namespace="default"
        )

    def test_missing_network_policy_is_ignored(self):
        from langchain_kubernetes.backends.raw import _ApiException

        provider = KubernetesProvider(_raw_config())
        core_v1, networking_v1 = self._clients()
        networking_v1.delete_namespaced_network_policy.side_effect = _ApiException(status=404)
        with patch("langchain_kubernetes.backends.raw.RawK8sBackend.load_k8s_clients",
                   return_value=(core_v1, networking_v1)):
            provider._delete_raw_pod("abc123", "default")  # no exception

        core_v1.delete_namespaced_pod.assert_called_once()
