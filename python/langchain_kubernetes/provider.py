"""KubernetesProvider: stateless sandbox lifecycle management.

The provider holds no per-sandbox state. Callers receive a ``sandbox.id``
on creation and are responsible for persisting it (e.g. in LangGraph graph
state) to reconnect on subsequent calls.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from datetime import datetime, timezone
from typing import Any

from langchain_kubernetes._labels import (
    ANN_CREATED_AT,
    ANN_LAST_ACTIVITY,
    ANN_TTL_IDLE_SECONDS,
    ANN_TTL_SECONDS,
    LABEL_POOL_STATUS,
    LABEL_PREFIX,
    MANAGED_SELECTOR,
    POOL_STATUS_ACTIVE,
    POOL_STATUS_WARM,
    build_labels,
    build_ttl_annotations,
    warm_pool_selector,
)
from langchain_kubernetes._provider_base import (
    SandboxNotFoundError,
    SandboxProvider,
)
from langchain_kubernetes._types import (
    CleanupResult,
    ProviderStats,
    SandboxInfo,
    SandboxListResponse,
    WarmPoolStatus,
)
from langchain_kubernetes.backends.agent_sandbox import AgentSandboxBackend
from langchain_kubernetes.backends.protocol import KubernetesBackendProtocol
from langchain_kubernetes.config import KubernetesProviderConfig
from langchain_kubernetes.sandbox import KubernetesSandbox

logger = logging.getLogger(__name__)

# CRD constants for agent-sandbox SandboxClaim resources
_CLAIM_API_GROUP = "extensions.agents.x-k8s.io"
_CLAIM_API_VERSION = "v1alpha1"
_CLAIM_PLURAL = "sandboxclaims"


class KubernetesProvider(SandboxProvider):
    """Stateless lifecycle manager for Kubernetes-based sandboxes.

    Supports two backend modes selected via ``config.mode``:

    - ``"agent-sandbox"`` — provisions Sandbox CRs via the
      ``kubernetes-sigs/agent-sandbox`` controller.
    - ``"raw"`` — directly creates ephemeral Pods. Works on any cluster with
      no additional infrastructure.

    **The provider holds no per-sandbox state.** Every :meth:`get_or_create`
    call returns a :class:`~langchain_kubernetes.sandbox.KubernetesSandbox`
    whose ``.id`` attribute is the durable sandbox identifier. Persist this ID
    in your application state (e.g. LangGraph graph state) and pass it back on
    the next call to reconnect to the same sandbox rather than creating a new one.

    The recommended integration pattern for LangGraph applications is to
    store ``sandbox_id`` as a field in your graph state so that LangGraph's
    checkpointer (in-memory, Postgres, Redis, …) handles cross-run persistence
    automatically — no Kubernetes label writes or direct cluster API access
    required.

    Args:
        config: Provider configuration. For agent-sandbox mode, ``template_name``
            is required. For raw mode, image and resource fields are used.
    """

    def __init__(self, config: KubernetesProviderConfig) -> None:
        self._config = config
        self._warm_pool_initialised = False
        self._warm_pool_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public: get_or_create / reconnect
    # ------------------------------------------------------------------

    def get_or_create(
        self,
        *,
        sandbox_id: str | None = None,
        labels: dict[str, str] | None = None,
        ttl_seconds: int | None = None,
        ttl_idle_seconds: int | None = None,
        **kwargs: Any,
    ) -> KubernetesSandbox:
        """Return an existing sandbox or create a new one.

        When *sandbox_id* is provided the provider first attempts to reconnect
        to that sandbox. If it no longer exists (deleted, TTL expired) a new
        sandbox is provisioned transparently. Check ``sandbox.id`` on the
        returned object to detect whether a new sandbox was created — its value
        will differ from the *sandbox_id* argument if a new one was provisioned.

        Persist ``sandbox.id`` in your application (e.g. LangGraph graph state)
        and pass it back on the next call to reuse the same sandbox across runs.

        Args:
            sandbox_id: ID returned by a previous call. Used to reconnect to an
                existing sandbox. Pass ``None`` (or omit) for the first call.
            labels: Labels applied only to *newly created* sandboxes. Keys are
                auto-prefixed with the provider label namespace. Ignored on
                reconnect.
            ttl_seconds: Absolute TTL from creation applied to new sandboxes.
                Defaults to ``config.ttl_seconds``.
            ttl_idle_seconds: Idle TTL from last execute(), applied to new
                sandboxes. Defaults to ``config.ttl_idle_seconds``.
            **kwargs: Ignored; present for interface compatibility.

        Returns:
            :class:`~langchain_kubernetes.sandbox.KubernetesSandbox`.

        Raises:
            ImportError: When the required backend package is not installed.
            TimeoutError: If a new sandbox does not become ready in time.
            RuntimeError: On sandbox creation failure.
        """
        if self._config.mode == "raw" and self._config.warm_pool_size > 0:
            self._ensure_warm_pool()

        # Attempt reconnect if we have an existing sandbox ID
        if sandbox_id is not None:
            try:
                return self.reconnect(sandbox_id)
            except SandboxNotFoundError:
                logger.info(
                    "Sandbox %s no longer exists — provisioning a new one", sandbox_id
                )

        # Build labels/annotations for the new sandbox
        eff_ttl = ttl_seconds if ttl_seconds is not None else self._config.ttl_seconds
        eff_idle = (
            ttl_idle_seconds if ttl_idle_seconds is not None else self._config.ttl_idle_seconds
        )
        extra_labels, extra_annotations = build_labels(
            default_labels=self._config.default_labels,
            call_labels=labels,
        )
        extra_annotations.update(
            build_ttl_annotations(ttl_seconds=eff_ttl, ttl_idle_seconds=eff_idle)
        )

        # Claim from warm pool (raw mode only)
        if self._config.mode == "raw" and self._config.warm_pool_size > 0:
            from langchain_kubernetes.backends.raw import RawK8sBackend

            backend = RawK8sBackend.claim_warm_pod(
                self._config,
                extra_labels=extra_labels,
                extra_annotations=extra_annotations,
                ttl_idle_seconds=eff_idle,
            )
            if backend is not None:
                logger.info("Claimed warm Pod %s", backend.id)
                return KubernetesSandbox(backend=backend)

        # Create a new sandbox
        backend = self._create_backend(
            extra_labels=extra_labels,
            extra_annotations=extra_annotations,
            ttl_idle_seconds=eff_idle,
        )
        logger.info(
            "Created sandbox %s (mode=%s, namespace=%s)",
            backend.id,
            self._config.mode,
            self._config.namespace,
        )
        return KubernetesSandbox(backend=backend)

    def reconnect(self, sandbox_id: str) -> KubernetesSandbox:
        """Reconnect to an existing sandbox by its ID.

        For raw mode the Pod must still be running; raises
        :exc:`~langchain_kubernetes._provider_base.SandboxNotFoundError`
        otherwise.

        For agent-sandbox mode the SandboxClaim must still exist. When the
        Kubernetes API is reachable (``kube_api_url`` configured or running
        in-cluster) the claim is verified before returning; otherwise the
        backend is returned optimistically and a missing claim will surface on
        the first ``execute()`` call.

        Args:
            sandbox_id: The ``sandbox.id`` from a previous
                :meth:`get_or_create` call.

        Returns:
            :class:`~langchain_kubernetes.sandbox.KubernetesSandbox`.

        Raises:
            SandboxNotFoundError: When the sandbox is confirmed gone.
            ImportError: When the required backend package is not installed.
        """
        backend = self._reconnect_backend(sandbox_id)
        logger.info("Reconnected to sandbox %s (mode=%s)", sandbox_id, self._config.mode)
        return KubernetesSandbox(backend=backend)

    async def areconnect(self, sandbox_id: str) -> KubernetesSandbox:
        """Async wrapper around :meth:`reconnect`.

        Args:
            sandbox_id: The sandbox ID to reconnect to.

        Returns:
            :class:`~langchain_kubernetes.sandbox.KubernetesSandbox`.

        Raises:
            SandboxNotFoundError: When the sandbox is confirmed gone.
        """
        return await asyncio.to_thread(self.reconnect, sandbox_id)

    # ------------------------------------------------------------------
    # Public: list / delete / cleanup / stats / pool_status
    # ------------------------------------------------------------------

    def list(
        self,
        cursor: str | None = None,
        *,
        labels: dict[str, str] | None = None,
        status: str | None = None,
        **kwargs: Any,
    ) -> SandboxListResponse:
        """List sandboxes from the Kubernetes API with optional filtering.

        Args:
            cursor: Kubernetes ``continue`` token for pagination.
            labels: Filter by labels (keys auto-prefixed with our namespace
                prefix). ``None`` returns all managed sandboxes.
            status: Filter by status string: ``"running"``, ``"warm"``,
                ``"terminated"``.
            **kwargs: Unused.

        Returns:
            :class:`~langchain_kubernetes._types.SandboxListResponse`.
        """
        if self._config.mode == "raw":
            return self._list_raw(cursor=cursor, labels=labels, status=status)
        return self._list_agent_sandbox(cursor=cursor, labels=labels, status=status)

    def delete(self, *, sandbox_id: str, **kwargs: Any) -> None:
        """Delete a sandbox by reaching directly into the Kubernetes API.

        Idempotent — silently no-ops if the sandbox is already gone.

        Args:
            sandbox_id: Sandbox ID returned by :meth:`get_or_create`.
            **kwargs: Unused.
        """
        try:
            if self._config.mode == "raw":
                self._delete_raw_pod(sandbox_id, self._config.namespace)
            else:
                self._delete_agent_sandbox_claim(sandbox_id)
            logger.info("Deleted sandbox %s", sandbox_id)
        except Exception as exc:
            logger.warning("Error deleting sandbox %s: %s", sandbox_id, exc)

        if self._config.mode == "raw" and self._config.warm_pool_size > 0:
            self._schedule_replenish()

    def cleanup(self, max_idle_seconds: int | None = None) -> CleanupResult:
        """Delete sandboxes that have exceeded their TTL or idle threshold.

        Lists all managed sandboxes via the Kubernetes API, checks TTL and
        idle-time annotations, and deletes any that have expired.

        Args:
            max_idle_seconds: Override idle threshold. Sandboxes whose
                last-activity annotation is older than this are deleted.

        Returns:
            :class:`~langchain_kubernetes._types.CleanupResult`.
        """
        result = CleanupResult()
        now = datetime.now(timezone.utc)
        response = self.list()

        for info in response.sandboxes:
            should_delete = False
            ann = info.annotations

            ttl_str = ann.get(ANN_TTL_SECONDS)
            created_str = ann.get(ANN_CREATED_AT)
            if ttl_str and created_str:
                try:
                    ttl = int(ttl_str)
                    created = datetime.fromisoformat(created_str)
                    if (now - created).total_seconds() > ttl:
                        should_delete = True
                        logger.info("Sandbox %s exceeded TTL (%ss)", info.id, ttl_str)
                except (ValueError, TypeError):
                    pass

            idle_threshold = max_idle_seconds
            if idle_threshold is None:
                idle_str = ann.get(ANN_TTL_IDLE_SECONDS)
                if idle_str:
                    try:
                        idle_threshold = int(idle_str)
                    except (ValueError, TypeError):
                        pass

            if idle_threshold is not None:
                last_str = ann.get(ANN_LAST_ACTIVITY) or ann.get(ANN_CREATED_AT)
                if last_str:
                    try:
                        last = datetime.fromisoformat(last_str)
                        if (now - last).total_seconds() > idle_threshold:
                            should_delete = True
                            logger.info(
                                "Sandbox %s exceeded idle threshold (%ss)",
                                info.id,
                                idle_threshold,
                            )
                    except (ValueError, TypeError):
                        pass

            if should_delete:
                try:
                    if self._config.mode == "raw":
                        self._delete_raw_pod(info.id, info.namespace)
                    else:
                        self._delete_agent_sandbox_claim(info.id)
                    result.deleted.append(info.id)
                except Exception as exc:
                    logger.warning("Failed to delete sandbox %s: %s", info.id, exc)
            else:
                result.kept += 1

        return result

    def stats(self, idle_threshold_seconds: int = 300) -> ProviderStats:
        """Return aggregate statistics for all managed sandboxes.

        Args:
            idle_threshold_seconds: Seconds since last execute() before a
                sandbox is considered idle. Default: 300.

        Returns:
            :class:`~langchain_kubernetes._types.ProviderStats`.
        """
        response = self.list()
        now = datetime.now(timezone.utc)
        running = warm = idle = 0

        for info in response.sandboxes:
            if info.status == "running":
                running += 1
            elif info.status == "warm":
                warm += 1

            last_str = (
                info.annotations.get(ANN_LAST_ACTIVITY)
                or info.annotations.get(ANN_CREATED_AT)
            )
            if last_str and info.status == "running":
                try:
                    last = datetime.fromisoformat(last_str)
                    if (now - last).total_seconds() > idle_threshold_seconds:
                        idle += 1
                except (ValueError, TypeError):
                    pass

        return ProviderStats(
            total=len(response.sandboxes),
            running=running,
            warm=warm,
            idle=idle,
            thread_ids=0,
        )

    def pool_status(self) -> WarmPoolStatus:
        """Return the current warm-pool status.

        For raw mode, counts Pods by their ``pool-status`` label. For
        agent-sandbox mode, returns a basic count from :meth:`list`.

        Returns:
            :class:`~langchain_kubernetes._types.WarmPoolStatus`.
        """
        if self._config.mode != "raw":
            response = self.list()
            active = sum(1 for s in response.sandboxes if s.status == "running")
            return WarmPoolStatus(available=0, active=active, total=active, target=0)

        try:
            from langchain_kubernetes.backends.raw import RawK8sBackend

            core_v1, _ = RawK8sBackend.load_k8s_clients()
            warm_list = core_v1.list_namespaced_pod(
                namespace=self._config.namespace,
                label_selector=warm_pool_selector(),
            )
            available = len(
                [
                    p
                    for p in (warm_list.items or [])
                    if p.status and p.status.phase in ("Running", "Pending")
                ]
            )
            active_list = core_v1.list_namespaced_pod(
                namespace=self._config.namespace,
                label_selector=f"{LABEL_POOL_STATUS}={POOL_STATUS_ACTIVE}",
            )
            active = len(active_list.items or [])
        except Exception as exc:
            logger.warning("Failed to query warm pool status: %s", exc)
            return WarmPoolStatus(
                available=0, active=0, total=0, target=self._config.warm_pool_size
            )

        return WarmPoolStatus(
            available=available,
            active=active,
            total=available + active,
            target=self._config.warm_pool_size,
        )

    # ------------------------------------------------------------------
    # Public: async wrappers
    # ------------------------------------------------------------------

    async def aget_or_create(
        self,
        *,
        sandbox_id: str | None = None,
        labels: dict[str, str] | None = None,
        ttl_seconds: int | None = None,
        ttl_idle_seconds: int | None = None,
        **kwargs: Any,
    ) -> KubernetesSandbox:
        """Async wrapper around :meth:`get_or_create`.

        Args:
            sandbox_id: Existing sandbox ID to reconnect to, or ``None``.
            labels: Per-call labels for new sandboxes.
            ttl_seconds: Absolute TTL override.
            ttl_idle_seconds: Idle TTL override.
            **kwargs: Forwarded.

        Returns:
            :class:`~langchain_kubernetes.sandbox.KubernetesSandbox`.
        """
        return await asyncio.to_thread(
            self.get_or_create,
            sandbox_id=sandbox_id,
            labels=labels,
            ttl_seconds=ttl_seconds,
            ttl_idle_seconds=ttl_idle_seconds,
            **kwargs,
        )

    async def alist(
        self,
        cursor: str | None = None,
        *,
        labels: dict[str, str] | None = None,
        status: str | None = None,
        **kwargs: Any,
    ) -> SandboxListResponse:
        """Async wrapper around :meth:`list`."""
        return await asyncio.to_thread(
            self.list, cursor, labels=labels, status=status, **kwargs
        )

    async def adelete(self, *, sandbox_id: str, **kwargs: Any) -> None:
        """Async wrapper around :meth:`delete`."""
        await asyncio.to_thread(self.delete, sandbox_id=sandbox_id, **kwargs)

    async def acleanup(self, max_idle_seconds: int | None = None) -> CleanupResult:
        """Async wrapper around :meth:`cleanup`."""
        return await asyncio.to_thread(self.cleanup, max_idle_seconds)

    async def astats(self, idle_threshold_seconds: int = 300) -> ProviderStats:
        """Async wrapper around :meth:`stats`."""
        return await asyncio.to_thread(self.stats, idle_threshold_seconds)

    # ------------------------------------------------------------------
    # Internal: reconnect
    # ------------------------------------------------------------------

    def _reconnect_backend(self, sandbox_id: str) -> KubernetesBackendProtocol:
        if self._config.mode == "agent-sandbox":
            return self._reconnect_agent_sandbox_backend(sandbox_id)
        if self._config.mode == "raw":
            return self._reconnect_raw_backend(sandbox_id)
        raise ValueError(f"Unknown mode: {self._config.mode!r}")

    def _reconnect_agent_sandbox_backend(self, sandbox_id: str) -> AgentSandboxBackend:
        """Attach to an existing SandboxClaim without provisioning a new one.

        When the Kubernetes API is reachable the claim is verified first.
        When the API is not configured (e.g. accessing through the gateway
        from outside the cluster) the backend is returned optimistically —
        a missing claim surfaces as an error on the first execute() call.
        """
        from langchain_kubernetes._k8s_http import is_k8s_api_configured

        if is_k8s_api_configured(self._config.kube_api_url, self._config.kube_token):
            try:
                items = self._list_sandbox_claims()
                claim_names = {
                    item.get("metadata", {}).get("name") for item in items
                }
                if sandbox_id not in claim_names:
                    raise SandboxNotFoundError(
                        f"SandboxClaim '{sandbox_id}' not found in namespace "
                        f"'{self._config.namespace}'"
                    )
            except SandboxNotFoundError:
                raise
            except Exception as exc:
                logger.debug(
                    "Cannot verify SandboxClaim %s via K8s API (%s) — proceeding optimistically",
                    sandbox_id,
                    exc,
                )

        client = _build_agent_sandbox_client(self._config)
        # Set sandbox_name on the client so run() routes to the correct claim
        # without calling __enter__() (which would provision a new claim).
        try:
            client.sandbox_name = sandbox_id
            client.claim_name = sandbox_id
        except AttributeError:
            pass  # SDK may not expose these as writable; handled by sandbox_name arg

        return AgentSandboxBackend(client=client, sandbox_name=sandbox_id)

    def _reconnect_raw_backend(self, sandbox_id: str) -> KubernetesBackendProtocol:
        from langchain_kubernetes.backends.raw import RawK8sBackend

        backend = RawK8sBackend.reconnect(self._config, sandbox_id)
        if backend is None:
            raise SandboxNotFoundError(
                f"Pod for sandbox '{sandbox_id}' not found or not running in "
                f"namespace '{self._config.namespace}'"
            )
        return backend

    # ------------------------------------------------------------------
    # Internal: create
    # ------------------------------------------------------------------

    def _create_backend(
        self,
        extra_labels: dict[str, str] | None = None,
        extra_annotations: dict[str, str] | None = None,
        ttl_idle_seconds: int | None = None,
    ) -> KubernetesBackendProtocol:
        if self._config.mode == "agent-sandbox":
            return self._create_agent_sandbox_backend(
                extra_labels=extra_labels,
                extra_annotations=extra_annotations,
            )
        if self._config.mode == "raw":
            return self._create_raw_backend(
                extra_labels=extra_labels,
                extra_annotations=extra_annotations,
                ttl_idle_seconds=ttl_idle_seconds,
            )
        raise ValueError(f"Unknown mode: {self._config.mode!r}")

    def _create_agent_sandbox_backend(
        self,
        extra_labels: dict[str, str] | None = None,
        extra_annotations: dict[str, str] | None = None,
    ) -> AgentSandboxBackend:
        """Build and enter a SandboxClient, return an AgentSandboxBackend."""
        client = _build_agent_sandbox_client(self._config)

        try:
            client.__enter__()
        except Exception as exc:
            _raise_clear_agent_sandbox_error(exc, self._config)

        sandbox_name: str = client.sandbox_name or client.claim_name or "unknown"
        logger.info(
            "agent-sandbox: provisioned %s (template=%s)",
            sandbox_name,
            self._config.template_name,
        )
        return AgentSandboxBackend(client=client, sandbox_name=sandbox_name)

    def _create_raw_backend(
        self,
        extra_labels: dict[str, str] | None = None,
        extra_annotations: dict[str, str] | None = None,
        ttl_idle_seconds: int | None = None,
    ) -> KubernetesBackendProtocol:
        from langchain_kubernetes.backends.raw import RawK8sBackend

        return RawK8sBackend.create(
            self._config,
            extra_labels=extra_labels,
            extra_annotations=extra_annotations,
            ttl_idle_seconds=ttl_idle_seconds,
        )

    # ------------------------------------------------------------------
    # Internal: list
    # ------------------------------------------------------------------

    def _list_raw(
        self,
        cursor: str | None = None,
        labels: dict[str, str] | None = None,
        status: str | None = None,
    ) -> SandboxListResponse:
        try:
            from langchain_kubernetes.backends.raw import RawK8sBackend

            core_v1, _ = RawK8sBackend.load_k8s_clients()
        except ImportError:
            return SandboxListResponse(sandboxes=[])

        selector = MANAGED_SELECTOR
        if labels:
            for k, v in labels.items():
                selector += f",{LABEL_PREFIX}{k}={v}"

        kwargs_k8s: dict[str, Any] = {
            "namespace": self._config.namespace,
            "label_selector": selector,
        }
        if cursor:
            kwargs_k8s["_continue"] = cursor

        try:
            pod_list = core_v1.list_namespaced_pod(**kwargs_k8s)
        except Exception as exc:
            logger.warning("Failed to list Pods: %s", exc)
            return SandboxListResponse(sandboxes=[])

        sandboxes = []
        for pod in pod_list.items or []:
            info = _pod_to_sandbox_info(pod)
            if status and info.status != status:
                continue
            sandboxes.append(info)

        next_cursor = getattr(pod_list.metadata, "_continue", None) or None
        return SandboxListResponse(sandboxes=sandboxes, cursor=next_cursor)

    def _list_agent_sandbox(
        self,
        cursor: str | None = None,
        labels: dict[str, str] | None = None,
        status: str | None = None,
    ) -> SandboxListResponse:
        from langchain_kubernetes._labels import LABEL_MANAGED_BY, LABEL_MANAGED_BY_VALUE

        selector_parts = [f"{LABEL_MANAGED_BY}={LABEL_MANAGED_BY_VALUE}"]
        if labels:
            for k, v in labels.items():
                selector_parts.append(f"{LABEL_PREFIX}{k}={v}")

        try:
            items = self._list_sandbox_claims(
                label_selector=",".join(selector_parts),
                continuation=cursor,
            )
        except Exception as exc:
            logger.warning("Failed to list SandboxClaims: %s", exc)
            return SandboxListResponse(sandboxes=[])

        sandboxes = []
        for item in items:
            info = _claim_to_sandbox_info(item, self._config.namespace)
            if status and info.status != status:
                continue
            sandboxes.append(info)

        return SandboxListResponse(sandboxes=sandboxes)

    def _list_sandbox_claims(
        self,
        label_selector: str | None = None,
        continuation: str | None = None,
    ) -> list[dict]:
        """List SandboxClaims from the Kubernetes API."""
        from langchain_kubernetes._k8s_http import k8s_get

        path = (
            f"/apis/{_CLAIM_API_GROUP}/{_CLAIM_API_VERSION}"
            f"/namespaces/{self._config.namespace}/{_CLAIM_PLURAL}"
        )
        resp = k8s_get(
            api_url=self._config.kube_api_url,
            token_override=self._config.kube_token,
            path=path,
            label_selector=label_selector,
        )
        return resp.get("items", [])

    # ------------------------------------------------------------------
    # Internal: delete
    # ------------------------------------------------------------------

    def _delete_agent_sandbox_claim(self, claim_name: str) -> None:
        """Delete a SandboxClaim via the Kubernetes API."""
        import urllib.error
        import urllib.request

        from langchain_kubernetes._k8s_http import (
            _IN_CLUSTER_API_URL,
            _build_ssl_context,
            _make_headers,
            _read_token,
        )

        path = (
            f"/apis/{_CLAIM_API_GROUP}/{_CLAIM_API_VERSION}"
            f"/namespaces/{self._config.namespace}/{_CLAIM_PLURAL}/{claim_name}"
        )
        base = (self._config.kube_api_url or _IN_CLUSTER_API_URL).rstrip("/")
        url = f"{base}{path}"
        token = _read_token(self._config.kube_token)
        headers = _make_headers(token)
        ctx = _build_ssl_context()

        req = urllib.request.Request(url, headers=headers, method="DELETE")
        try:
            with urllib.request.urlopen(req, context=ctx, timeout=10):
                pass
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return
            raise

    def _delete_raw_pod(self, sandbox_id: str, namespace: str) -> None:
        """Delete a raw-mode Pod and its deny-all NetworkPolicy by sandbox ID.

        The policy is created alongside the Pod when ``block_network=True``;
        deleting only the Pod left one orphaned policy behind per sandbox.
        Missing objects (404) are ignored.
        """
        try:
            from langchain_kubernetes.backends.raw import RawK8sBackend, _ApiException

            core_v1, networking_v1 = RawK8sBackend.load_k8s_clients()
            deletions = (
                (core_v1.delete_namespaced_pod, f"deepagents-{sandbox_id}"),
                (networking_v1.delete_namespaced_network_policy, f"deepagents-deny-all-{sandbox_id}"),
            )
            for delete, name in deletions:
                try:
                    delete(name=name, namespace=namespace)
                except _ApiException as exc:
                    if exc.status != 404:
                        raise
        except ImportError:
            pass

    # ------------------------------------------------------------------
    # Internal: warm pool
    # ------------------------------------------------------------------

    def _ensure_warm_pool(self) -> None:
        """Initialise the warm pool on first call (lazy, thread-safe)."""
        with self._warm_pool_lock:
            if self._warm_pool_initialised:
                return
            self._warm_pool_initialised = True
        self._replenish_warm_pool()

    def _replenish_warm_pool(self) -> None:
        """Create warm Pods until the pool reaches the target size."""
        if self._config.mode != "raw" or self._config.warm_pool_size <= 0:
            return

        try:
            from langchain_kubernetes.backends.raw import RawK8sBackend

            core_v1, _ = RawK8sBackend.load_k8s_clients()
        except ImportError:
            return

        try:
            warm_list = core_v1.list_namespaced_pod(
                namespace=self._config.namespace,
                label_selector=warm_pool_selector(),
            )
            current_count = len(
                [
                    p
                    for p in (warm_list.items or [])
                    if p.status
                    and p.status.phase not in ("Failed", "Unknown", "Succeeded")
                ]
            )
        except Exception as exc:
            logger.warning("Failed to count warm Pods: %s", exc)
            return

        needed = self._config.warm_pool_size - current_count
        if needed <= 0:
            return

        from langchain_kubernetes._labels import build_labels as _bl

        pool_labels, _ = _bl()
        pool_labels[LABEL_POOL_STATUS] = POOL_STATUS_WARM

        for _ in range(needed):
            try:
                from langchain_kubernetes.backends.raw import RawK8sBackend

                backend = RawK8sBackend.create(self._config, extra_labels=pool_labels)
                logger.info("Created warm Pod %s", backend.id)
            except Exception as exc:
                logger.warning("Failed to create warm Pod: %s", exc)

    def _schedule_replenish(self) -> None:
        """Schedule warm pool replenishment in a background thread."""
        t = threading.Thread(target=self._replenish_warm_pool, daemon=True)
        t.start()


# ---------------------------------------------------------------------------
# agent-sandbox helpers
# ---------------------------------------------------------------------------


def _import_sandbox_client():
    """Import ``SandboxClient`` with a friendly error on missing package."""
    try:
        from k8s_agent_sandbox import SandboxClient

        return SandboxClient
    except ImportError as exc:
        raise ImportError(
            "agent-sandbox mode requires the 'k8s-agent-sandbox' package. "
            "Install with: pip install langchain-kubernetes[agent-sandbox]"
        ) from exc


def _build_agent_sandbox_client(config: KubernetesProviderConfig):
    """Instantiate a ``SandboxClient`` from *config* (not yet entered)."""
    SandboxClient = _import_sandbox_client()

    kwargs: dict[str, Any] = {
        "template_name": config.template_name,
        "namespace": config.namespace,
        "server_port": config.server_port,
        "sandbox_ready_timeout": config.startup_timeout_seconds,
    }

    if config.connection_mode == "gateway":
        if config.gateway_name is None:
            raise ValueError(
                "connection_mode='gateway' requires gateway_name to be set in KubernetesProviderConfig"
            )
        kwargs["gateway_name"] = config.gateway_name
        kwargs["gateway_namespace"] = config.gateway_namespace
    elif config.connection_mode == "direct":
        if config.api_url is None:
            raise ValueError(
                "connection_mode='direct' requires api_url to be set in KubernetesProviderConfig"
            )
        kwargs["api_url"] = config.api_url

    if config.warm_pool_name:
        kwargs["warm_pool_name"] = config.warm_pool_name

    return SandboxClient(**kwargs)


def _raise_clear_agent_sandbox_error(
    exc: Exception, config: KubernetesProviderConfig
) -> None:
    """Re-raise *exc* with a human-readable message for common failure modes."""
    msg = str(exc).lower()

    # Guard: re-raise dev-server enforcement errors (e.g. blockbuster.BlockingError)
    # before any keyword matching — their messages contain "connect" which would
    # otherwise trigger the connectivity hint below.
    if "blocking call" in msg:
        raise exc

    if "timeout" in msg or isinstance(exc, TimeoutError):
        raise TimeoutError(
            f"Sandbox did not become ready within {config.startup_timeout_seconds}s. "
            "Check controller logs: kubectl logs -n agent-sandbox-system "
            "-l app=agent-sandbox-controller"
        ) from exc

    if "not found" in msg and config.template_name and config.template_name in str(exc):
        raise RuntimeError(
            f"SandboxTemplate '{config.template_name}' not found in namespace "
            f"'{config.namespace}'. Create it first — see README."
        ) from exc

    if any(kw in msg for kw in ("connection refused", "unreachable", "connect", "router")):
        mode_hint = {
            "tunnel": "Ensure kubectl is available and the sandbox-router Service exists.",
            "gateway": f"Ensure the Gateway '{config.gateway_name}' is deployed and reachable.",
            "direct": f"Ensure the sandbox-router is accessible at '{config.api_url}'.",
        }.get(config.connection_mode, "")
        raise RuntimeError(
            f"Cannot reach the sandbox-router (connection_mode='{config.connection_mode}'). "
            f"{mode_hint}"
        ) from exc

    raise exc


# ---------------------------------------------------------------------------
# Info helpers
# ---------------------------------------------------------------------------


def _pod_to_sandbox_info(pod: Any) -> SandboxInfo:
    """Convert a Kubernetes Pod object to SandboxInfo."""
    from langchain_kubernetes.backends.raw_manifests import LABEL_SANDBOX_ID
    from langchain_kubernetes._labels import LABEL_POOL_STATUS

    meta = pod.metadata or object()
    name = getattr(meta, "name", "unknown") or "unknown"
    namespace = getattr(meta, "namespace", "default") or "default"
    labels: dict[str, str] = dict(getattr(meta, "labels", {}) or {})
    annotations: dict[str, str] = dict(getattr(meta, "annotations", {}) or {})

    sandbox_id = labels.get(LABEL_SANDBOX_ID, name.removeprefix("deepagents-"))
    pool_status = labels.get(LABEL_POOL_STATUS)

    phase = None
    if pod.status:
        phase = pod.status.phase

    if pool_status == POOL_STATUS_WARM:
        status = "warm"
    elif phase == "Running":
        status = "running"
    elif phase in ("Succeeded", "Failed"):
        status = "terminated"
    else:
        status = phase.lower() if phase else "pending"

    return SandboxInfo(
        id=sandbox_id,
        namespace=namespace,
        labels=labels,
        annotations=annotations,
        created_at=annotations.get(ANN_CREATED_AT),
        last_activity=annotations.get(ANN_LAST_ACTIVITY),
        status=status,
    )


def _claim_to_sandbox_info(item: dict, default_namespace: str) -> SandboxInfo:
    """Convert a raw SandboxClaim dict to SandboxInfo."""
    meta = item.get("metadata", {})
    name = meta.get("name", "unknown")
    namespace = meta.get("namespace", default_namespace)
    labels: dict[str, str] = meta.get("labels") or {}
    annotations: dict[str, str] = meta.get("annotations") or {}

    status_obj = item.get("status", {})
    conditions = status_obj.get("conditions", []) if isinstance(status_obj, dict) else []
    ready = any(
        c.get("type") == "Ready" and c.get("status") == "True"
        for c in conditions
        if isinstance(c, dict)
    )
    status = "running" if ready else "pending"

    return SandboxInfo(
        id=name,
        namespace=namespace,
        labels=labels,
        annotations=annotations,
        created_at=annotations.get(ANN_CREATED_AT),
        last_activity=annotations.get(ANN_LAST_ACTIVITY),
        status=status,
    )
