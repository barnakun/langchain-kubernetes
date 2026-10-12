"""Outside a cluster, Kubernetes API calls use the kubeconfig credentials, certificates too."""

from __future__ import annotations

import urllib.error
from unittest.mock import MagicMock, patch

import pytest

from langchain_kubernetes import _k8s_http


class _Configuration:
    host = "https://kubeconfig-host:6443"
    ssl_ca_cert = None
    verify_ssl = False
    cert_file = None
    key_file = None

    def get_api_key_with_prefix(self, identifier):
        return "Bearer from-kubeconfig" if identifier == "authorization" else None


def _response(payload: bytes = b'{"items": [{"metadata": {"name": "c1"}}]}'):
    response = MagicMock()
    response.__enter__.return_value.read.return_value = payload
    return response


class TestKubeconfigFallback:
    def test_get_without_token_or_cluster_uses_the_kubeconfig(self) -> None:
        with patch.object(_k8s_http, "_in_cluster", return_value=False), \
             patch.object(_k8s_http, "_kubeconfig", return_value=_Configuration()), \
             patch("urllib.request.urlopen", return_value=_response()) as urlopen:
            result = _k8s_http.k8s_get(
                None, None, "/apis/x/v1/namespaces/default/claims", label_selector="a=b"
            )

        assert result == {"items": [{"metadata": {"name": "c1"}}]}
        request = urlopen.call_args.args[0]
        assert request.full_url == "https://kubeconfig-host:6443/apis/x/v1/namespaces/default/claims?labelSelector=a%3Db"
        assert request.get_header("Authorization") == "Bearer from-kubeconfig"

    def test_an_explicit_api_url_overrides_the_kubeconfig_host(self) -> None:
        with patch.object(_k8s_http, "_in_cluster", return_value=False), \
             patch.object(_k8s_http, "_kubeconfig", return_value=_Configuration()), \
             patch("urllib.request.urlopen", return_value=_response(b"{}")) as urlopen:
            _k8s_http.k8s_get("https://127.0.0.1:57346/", None, "/api/v1/x")

        assert urlopen.call_args.args[0].full_url == "https://127.0.0.1:57346/api/v1/x"

    def test_a_client_certificate_is_loaded_into_the_tls_context(self) -> None:
        configuration = _Configuration()
        configuration.cert_file, configuration.key_file = "/c.crt", "/c.key"
        with patch("ssl.SSLContext.load_cert_chain") as load_cert_chain:
            _k8s_http._kubeconfig_transport(configuration)

        load_cert_chain.assert_called_once_with("/c.crt", "/c.key")

    def test_a_token_keeps_the_plain_request(self) -> None:
        with patch.object(_k8s_http, "_kubeconfig") as kubeconfig, \
             patch("urllib.request.urlopen", return_value=_response(b'{"ok": true}')) as urlopen:
            assert _k8s_http.k8s_get("https://api", "tok", "/api/v1/x") == {"ok": True}

        kubeconfig.assert_not_called()
        assert urlopen.call_args.args[0].get_header("Authorization") == "Bearer tok"

    def test_delete_ignores_a_missing_object_and_raises_other_errors(self) -> None:
        def fail(code):
            return urllib.error.HTTPError("u", code, "x", None, None)

        with patch.object(_k8s_http, "_in_cluster", return_value=False), \
             patch.object(_k8s_http, "_kubeconfig", return_value=_Configuration()), \
             patch("urllib.request.urlopen", side_effect=fail(404)) as urlopen:
            _k8s_http.k8s_delete(None, None, "/apis/x/v1/namespaces/default/claims/c1")
            assert urlopen.call_args.args[0].get_method() == "DELETE"
            urlopen.side_effect = fail(403)
            with pytest.raises(urllib.error.HTTPError) as exc:
                _k8s_http.k8s_delete(None, None, "/apis/x/v1/namespaces/default/claims/c1")

        assert exc.value.code == 403

    def test_api_counts_as_configured_when_a_kubeconfig_loads(self) -> None:
        with patch.object(_k8s_http, "_in_cluster", return_value=False), \
             patch.object(_k8s_http, "_kubeconfig", return_value=_Configuration()):
            assert _k8s_http.is_k8s_api_configured(None, None)
        with patch.object(_k8s_http, "_in_cluster", return_value=False), \
             patch.object(_k8s_http, "_kubeconfig", return_value=None):
            assert not _k8s_http.is_k8s_api_configured(None, None)


def test_provider_deletes_claims_through_the_shared_helper() -> None:
    from langchain_kubernetes.config import KubernetesProviderConfig
    from langchain_kubernetes.provider import KubernetesProvider

    provider = KubernetesProvider(KubernetesProviderConfig(template_name="t", namespace="ns1"))
    with patch("langchain_kubernetes._k8s_http.k8s_delete") as delete:
        provider._delete_agent_sandbox_claim("claim-1")

    path = delete.call_args.args[2]
    assert path.endswith("/namespaces/ns1/sandboxclaims/claim-1")
