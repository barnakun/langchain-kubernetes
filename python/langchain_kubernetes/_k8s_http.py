"""Minimal stdlib-based Kubernetes HTTP client for agent-sandbox mode.

Used to list and patch CRD resources (SandboxClaims) when the ``kubernetes``
package is not available. Uses only the Python standard library
(``urllib.request``, ``ssl``). Handles both in-cluster and out-of-cluster
access via an explicit API URL + token.
"""

from __future__ import annotations

import json
import logging
import ssl
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_IN_CLUSTER_TOKEN_PATH = "/var/run/secrets/kubernetes.io/serviceaccount/token"
_IN_CLUSTER_CA_PATH = "/var/run/secrets/kubernetes.io/serviceaccount/ca.crt"
_IN_CLUSTER_API_URL = "https://kubernetes.default.svc.cluster.local"


def _build_ssl_context() -> ssl.SSLContext:
    """Build an SSL context that works for both in-cluster and dev setups."""
    ctx = ssl.create_default_context()
    ca = Path(_IN_CLUSTER_CA_PATH)
    if ca.exists():
        try:
            ctx.load_verify_locations(str(ca))
        except Exception:
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
    else:
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
    return ctx


def _read_token(override: str | None) -> str | None:
    if override:
        return override
    try:
        return Path(_IN_CLUSTER_TOKEN_PATH).read_text().strip()
    except Exception:
        return None


def _make_headers(token: str | None, content_type: str = "application/json") -> dict[str, str]:
    headers: dict[str, str] = {
        "Accept": "application/json",
        "Content-Type": content_type,
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _in_cluster() -> bool:
    """Whether Kubernetes mounted a service-account token into this process's pod."""
    return Path(_IN_CLUSTER_TOKEN_PATH).exists()


def _kubeconfig() -> Any | None:
    """The local kubeconfig resolved into a ``kubernetes.client.Configuration``, or ``None``.

    A kubeconfig may authenticate with client certificates or an exec plugin (minikube, kind,
    cloud CLIs), which a bare bearer-token request cannot reproduce. Only the stable
    ``Configuration`` fields are used (host, CA, client certificate, token), not the generated API
    client, whose call signature differs between ``kubernetes`` releases. ``None`` when the
    package or a kubeconfig is missing.
    """
    try:
        from kubernetes import client, config

        configuration = client.Configuration()
        config.load_kube_config(client_configuration=configuration)
        return configuration
    except Exception:
        return None


def _kubeconfig_transport(configuration: Any) -> tuple[str, ssl.SSLContext, str | None]:
    """Host, TLS context (CA, client certificate) and bearer header value of a Configuration."""
    ca = getattr(configuration, "ssl_ca_cert", None)
    ctx = ssl.create_default_context(cafile=ca) if ca else ssl.create_default_context()
    if not getattr(configuration, "verify_ssl", True):
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
    cert = getattr(configuration, "cert_file", None)
    if cert:
        ctx.load_cert_chain(cert, getattr(configuration, "key_file", None) or None)
    auth = None
    getter = getattr(configuration, "get_api_key_with_prefix", None)
    if callable(getter):
        auth = getter("authorization") or getter("BearerToken")
    return str(getattr(configuration, "host", "") or ""), ctx, auth


def is_k8s_api_configured(api_url: str | None, token_override: str | None) -> bool:
    """Return True if Kubernetes API access is configured: an explicit URL or token, a
    service-account token (in a cluster), or a usable kubeconfig (outside one).

    When none applies (e.g. a developer reaching an externally hosted sandbox-router) there is no
    reason to attempt K8s API calls, and failures should not produce warnings.

    Args:
        api_url: Explicit Kubernetes API base URL from config (``kube_api_url``).
        token_override: Explicit bearer token from config (``kube_token``).

    Returns:
        ``True`` if K8s API access is configured or expected to work.
    """
    if api_url or token_override or _in_cluster():
        return True
    return _kubeconfig() is not None


def _request(
    method: str,
    api_url: str | None,
    token_override: str | None,
    path: str,
    *,
    query: dict[str, str] | None = None,
    body: dict[str, Any] | None = None,
    content_type: str = "application/json",
    timeout: int = 10,
) -> dict[str, Any]:
    """One Kubernetes API call over HTTPS.

    Credentials: an explicit token, else the in-cluster service-account token, else the local
    kubeconfig (client certificate or token; ``api_url``, if given, replaces its host).
    """
    base = (api_url or _IN_CLUSTER_API_URL).rstrip("/")
    ctx = None
    auth = None
    if not token_override and not _in_cluster():
        configuration = _kubeconfig()
        if configuration is not None:
            host, ctx, auth = _kubeconfig_transport(configuration)
            base = (api_url or host).rstrip("/")
    url = f"{base}{path}"
    if query:
        url += "?" + urllib.parse.urlencode(query)
    headers = _make_headers(_read_token(token_override), content_type=content_type)
    if auth:
        headers["Authorization"] = auth if " " in auth else f"Bearer {auth}"
    data_bytes = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data_bytes, headers=headers, method=method)
    try:
        context = ctx or _build_ssl_context()
        with urllib.request.urlopen(req, context=context, timeout=timeout) as resp:
            raw = resp.read()
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as exc:
        logger.debug("k8s %s %s → %s", method, path, exc.code)
        raise


def k8s_get(
    api_url: str | None,
    token_override: str | None,
    path: str,
    label_selector: str | None = None,
    *,
    timeout: int = 10,
) -> dict[str, Any]:
    """Perform a GET request against the Kubernetes API.

    Args:
        api_url: Base URL of the Kubernetes API server.
        token_override: Explicit bearer token (``None``: in-cluster token, else the kubeconfig).
        path: API path (e.g. ``"/apis/extensions.agents.x-k8s.io/v1alpha1/...``).
        label_selector: Optional label selector query string.
        timeout: HTTP timeout in seconds.

    Returns:
        Parsed JSON response dict.

    Raises:
        urllib.error.HTTPError: On non-2xx responses.
    """
    query = {"labelSelector": label_selector} if label_selector else None
    return _request("GET", api_url, token_override, path, query=query, timeout=timeout)


def k8s_patch(
    api_url: str | None,
    token_override: str | None,
    path: str,
    patch: dict[str, Any],
    *,
    timeout: int = 10,
) -> dict[str, Any]:
    """Perform a merge PATCH request against the Kubernetes API.

    Args:
        api_url: Base URL of the Kubernetes API server.
        token_override: Explicit bearer token.
        path: API path of the resource to patch.
        patch: Merge-patch body dict.
        timeout: HTTP timeout in seconds.

    Returns:
        Parsed JSON response dict of the patched resource.

    Raises:
        urllib.error.HTTPError: On non-2xx responses.
    """
    return _request(
        "PATCH", api_url, token_override, path, body=patch,
        content_type="application/merge-patch+json", timeout=timeout,
    )


def k8s_delete(
    api_url: str | None,
    token_override: str | None,
    path: str,
    *,
    timeout: int = 10,
) -> None:
    """Delete a Kubernetes object; a missing one (404) is not an error.

    Raises:
        urllib.error.HTTPError: On other non-2xx responses.
    """
    try:
        _request("DELETE", api_url, token_override, path, timeout=timeout)
    except urllib.error.HTTPError as exc:
        if exc.code != 404:
            raise
