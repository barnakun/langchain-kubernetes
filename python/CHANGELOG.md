# Changelog — langchain-kubernetes (Python)

All notable changes to this project are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versions follow [Semantic Versioning](https://semver.org/).

---

## [Unreleased]

### Fixed

- **Raw mode: deleting a sandbox left its NetworkPolicy behind** — `KubernetesProvider.delete()` (also used by
  `cleanup()` for expired sandboxes) removed only the Pod, so with `block_network=True` every sandbox leaked
  its `deepagents-deny-all-<id>` NetworkPolicy. The policy is now deleted together with the Pod; a missing
  object (404) is ignored.

---

## [0.4.0] — 2026-04-01

### Changed

#### KubernetesSandboxManager — lazy sandbox acquisition, top-level deepagent

- `create_agent()` now returns the deepagent graph directly (via `create_deep_agent()`) instead of wrapping it in a `StateGraph(setup → agent)`. All deepagent steps (todos, tool calls, LLM tokens) are emitted as top-level graph events — visible in the Deep Agent UI and LangGraph Platform streaming.
- `_make_backend_factory()` now lazily acquires the sandbox on first tool call using a `ThreadPoolExecutor`, instead of requiring a dedicated setup node. The K8s I/O runs in a worker thread so the ASGI event loop thread stays free (avoids `blockbuster.BlockingError`).
- Module-level `_sandbox_acquire_executor` (`ThreadPoolExecutor`, 4 workers) added for the lazy acquisition path.
- `_sandbox_by_thread` instance cache is still populated, but now by the backend factory on first call rather than by a setup node.
- `create_setup_node()` and `create_agent_node()` are unchanged and kept for backward compatibility / custom graph builds.

---

## [0.3.0] — 2026-03-21

### Added

#### KubernetesSandboxManager

- New `KubernetesSandboxManager` class: stateless high-level wrapper for building DeepAgents-powered applications. Holds no sandbox cache and no locks — all sandbox-to-conversation binding lives in LangGraph graph state.
- `create_agent(model, *, checkpointer=None)`: returns a compiled LangGraph+DeepAgents graph. Stores `sandbox_id` in graph state; the checkpointer (or LangGraph Platform) persists it between runs so each thread reconnects to the same sandbox automatically.
- `create_agent_node(model)`: returns a single async node function for embedding in a larger `StateGraph`.
- `get_or_reconnect(sandbox_id)`: lower-level helper; delegates to `provider.aget_or_create(sandbox_id=...)`.
- `shutdown()`: deletes all sandboxes created by this provider instance; errors are logged but do not raise.
- `ashutdown()`: async cleanup variant.
- `langchain_kubernetes/manager.py`: new module containing `KubernetesSandboxManager`.

#### TTL and auto-cleanup

- New `ttl_seconds` and `ttl_idle_seconds` `KubernetesProviderConfig` fields: set `langchain-kubernetes.bitkaio.com/ttl-seconds`, `created-at`, and `ttl-idle-seconds` annotations at creation time.
- `execute()` updates `langchain-kubernetes.bitkaio.com/last-activity` annotation after each call (fire-and-forget, never blocks or raises).
- New `cleanup(max_idle_seconds=None)` method on `KubernetesProvider`: deletes sandboxes that have exceeded their absolute or idle TTL.
- New `stats(idle_threshold_seconds=300)` method: returns aggregate `ProviderStats` (total, running, warm, idle).
- New `list()` method: queries the Kubernetes API; supports filtering by `labels` and `status`; returns `SandboxListResponse`.
- New `pool_status()` method: returns `WarmPoolStatus` (available, active, total, target).
- Async variants for all provider methods: `aget_or_create()`, `alist()`, `adelete()`, `acleanup()`, `astats()`.

#### Warm pool integration

- **agent-sandbox mode**: new `warm_pool_name` config field — claims sandboxes from a named `SandboxWarmPool` CRD instead of cold-creating them.
- **raw mode**: new `warm_pool_size` config field — provider pre-creates idle Pods labelled `pool-status=warm`; `get_or_create()` claims a warm Pod (patches label to `pool-status=active`) before falling back to cold creation; `delete()` triggers background replenishment.

#### Optional K8s API access in agent-sandbox mode

- `is_k8s_api_configured()` in `langchain_kubernetes/_k8s_http.py`: returns `True` when direct K8s API access is available (explicit `kube_api_url` / `kube_token` config, or in-cluster service account token). When configured, `reconnect()` verifies sandbox existence before returning; otherwise returns optimistically.

#### New config fields (`KubernetesProviderConfig`)

- `default_labels` — applied to every sandbox.
- `ttl_seconds` — default absolute TTL.
- `ttl_idle_seconds` — default idle TTL.
- `warm_pool_size` — raw mode warm pool target size.
- `warm_pool_name` — agent-sandbox mode warm pool CRD name.
- `kube_api_url` — Kubernetes API URL for optional direct K8s API access in agent-sandbox mode.
- `kube_token` — Bearer token for Kubernetes API (auto-reads in-cluster service account token if unset).

#### New exported types

- `SandboxInfo` — metadata entry returned by `list()`.
- `SandboxListResponse` — list result with `.sandboxes` and optional `.cursor`.
- `CleanupResult` — result of `cleanup()` (`.deleted` IDs + `.kept` count).
- `WarmPoolStatus` — pool state snapshot (`.available`, `.active`, `.total`, `.target`).
- `ProviderStats` — aggregate statistics.

#### New internal modules

- `langchain_kubernetes/_labels.py`: label/annotation constants, `sanitize_label_value()`, `build_labels()`, `build_ttl_annotations()`, `warm_pool_selector()`.
- `langchain_kubernetes/_types.py`: `SandboxInfo`, `SandboxListResponse`, `CleanupResult`, `WarmPoolStatus`, `ProviderStats` dataclasses.
- `langchain_kubernetes/_k8s_http.py`: stdlib `urllib`-based Kubernetes HTTP client for agent-sandbox mode.
- `langchain_kubernetes/provider.py`: `get_or_create()` with reconnect support, `list()`, `delete()` with warm-pool replenishment, `cleanup()`, `stats()`, `pool_status()`.

### Changed

- `get_or_create()` now accepts keyword arguments including `sandbox_id` for reconnection, `labels`, `ttl_seconds`, `ttl_idle_seconds` — fully backward-compatible.
- `list()` return type is now `SandboxListResponse` with a `.sandboxes` field. **Breaking change** for direct callers of `provider.list()`.
- `RawK8sBackend.create()` signature extended with optional `sandbox_id`, `extra_labels`, `extra_annotations`, `ttl_idle_seconds` parameters — fully backward-compatible.

### Fixed

- `_raise_clear_agent_sandbox_error` no longer misidentifies `blockbuster.BlockingError` as a connectivity failure — the original exception is re-raised immediately when `"blocking call"` is detected in the message.

### CI

- Added `python:dev-publish` manual GitLab CI job in a new `dev-publish` stage; triggered via the GitLab UI or API from any branch, never run automatically.
- Version scheme: `{base}.dev{CI_PIPELINE_IID}` (PEP 440 pre-release).
- Publishes to the GitLab Package Registry only.

---

## [0.2.0] — 2026-01 (initial public release)

### Added

- `KubernetesProvider` with `agent-sandbox` and `raw` backend modes.
- `KubernetesSandbox` extending `BaseSandbox` (execute, upload, download).
- `AgentSandboxBackend`: wraps `k8s-agent-sandbox` `SandboxClient`.
- `RawK8sBackend`: direct Pod management via the `kubernetes` Python client.
- Deny-all `NetworkPolicy` per sandbox (raw mode, opt-out via `block_network=False`).
- Per-sandbox namespace isolation (`namespace_per_sandbox`).
- Hardened Pod security context defaults (`runAsNonRoot`, `allowPrivilegeEscalation: false`, `capabilities.drop: [ALL]`, `seccompProfile: RuntimeDefault`).
- Full file I/O: `upload_files` / `download_files` via tar-over-exec (raw) and `SandboxClient` file methods (agent-sandbox).
- 181 unit tests.

[0.4.0]: https://github.com/bitkaio/langchain-kubernetes/compare/python/v0.3.0...python/v0.4.0
[0.3.0]: https://github.com/bitkaio/langchain-kubernetes/compare/python/v0.2.0...python/v0.3.0
[0.2.0]: https://github.com/bitkaio/langchain-kubernetes/releases/tag/python/v0.2.0
