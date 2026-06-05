"""Tests for the minimax quota poller guard in create_app()."""

from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock

from shrimp_router.app import create_app




def _build_config(backends: dict) -> dict:
    return {"backends": backends}



def _backend(name: str, auth_env: str | None = None, quota: dict | None = None) -> dict:
    return {
        "name": name,
        "base_url": f"https://example.com/{name}",
        "models": [f"{name}-model"],
        "auth_env": auth_env,
        "quota": quota,
    }


class TestMinimaxQuotaPollerGuard:
    """Verify MinimaxQuotaPoller only starts for the 'minimax' backend."""

    @patch("shrimp_router.app.MinimaxQuotaPoller")
    @patch("shrimp_router.app.BackendManager")
    def test_minimax_backend_starts_poller(self, mock_mgr_cls, mock_poller_cls):
        """'minimax' backend with auth_env+quota should start a poller."""
        mock_mgr_cls.return_value = MagicMock()
        mock_poller_cls.return_value = MagicMock()

        config = _build_config({
            "minimax": _backend("minimax", auth_env="MINIMAX_API_KEY", quota={"limit": 1000}),
        })

        with patch.dict("os.environ", {"MINIMAX_API_KEY": "fake-key"}):
            app = create_app(config)

        assert len(app.state.quota_pollers) == 1
        mock_poller_cls.assert_called_once()

    @patch("shrimp_router.app.MinimaxQuotaPoller")
    @patch("shrimp_router.app.BackendManager")
    def test_non_minimax_backend_with_auth_env_and_quota_no_poller(self, mock_mgr_cls, mock_poller_cls):
        """Non-MiniMax backend with auth_env+quota must NOT start a poller."""
        mock_mgr_cls.return_value = MagicMock()
        mock_poller_cls.return_value = MagicMock()

        config = _build_config({
            "openai": _backend("openai", auth_env="OPENAI_API_KEY", quota={"limit": 5000}),
        })

        with patch.dict("os.environ", {"OPENAI_API_KEY": "fake-key"}):
            app = create_app(config)

        assert len(app.state.quota_pollers) == 0
        mock_poller_cls.assert_not_called()


    @patch("shrimp_router.app.MinimaxQuotaPoller")
    @patch("shrimp_router.app.BackendManager")
    def test_other_backend_starts_no_poller(self, mock_mgr_cls, mock_poller_cls):
        """'anthropic' backend with auth_env+quota must NOT start a poller."""
        mock_mgr_cls.return_value = MagicMock()
        mock_poller_cls.return_value = MagicMock()

        config = _build_config({
            "anthropic": _backend("anthropic", auth_env="ANTHROPIC_API_KEY", quota={"limit": 2000}),
        })

        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "fake-key"}):
            app = create_app(config)

        assert len(app.state.quota_pollers) == 0
        mock_poller_cls.assert_not_called()


    @patch("shrimp_router.app.MinimaxQuotaPoller")
    @patch("shrimp_router.app.BackendManager")
    def test_minimax_without_auth_env_no_poller(self, mock_mgr_cls, mock_poller_cls):
        """'minimax' backend without auth_env must NOT start a poller."""
        mock_mgr_cls.return_value = MagicMock()
        mock_poller_cls.return_value = MagicMock()

        config = _build_config({
            "minimax": _backend("minimax", auth_env=None, quota={"limit": 1000}),
        })

        app = create_app(config)
        assert len(app.state.quota_pollers) == 0
        mock_poller_cls.assert_not_called()

    @patch("shrimp_router.app.MinimaxQuotaPoller")
    @patch("shrimp_router.app.BackendManager")
    def test_minimax_without_quota_no_poller(self, mock_mgr_cls, mock_poller_cls):
        """'minimax' backend without quota config must NOT start a poller."""
        mock_mgr_cls.return_value = MagicMock()
        mock_poller_cls.return_value = MagicMock()

        config = _build_config({
            "minimax": _backend("minimax", auth_env="MINIMAX_API_KEY", quota=None),
        })

        with patch.dict("os.environ", {"MINIMAX_API_KEY": "fake-key"}):
            app = create_app(config)

        assert len(app.state.quota_pollers) == 0
        mock_poller_cls.assert_not_called()

    @patch("shrimp_router.app.MinimaxQuotaPoller")
    @patch("shrimp_router.app.BackendManager")
    def test_multiple_backends_only_minimax_starts_poller(self, mock_mgr_cls, mock_poller_cls):
        """Among many backends, only 'minimax' should start a poller."""
        mock_mgr_cls.return_value = MagicMock()
        mock_poller_cls.return_value = MagicMock()

        config = _build_config({
            "openai": _backend("openai", auth_env="OPENAI_API_KEY", quota={"limit": 5000}),
            "anthropic": _backend("anthropic", auth_env="ANTHROPIC_API_KEY", quota={"limit": 2000}),
            "minimax": _backend("minimax", auth_env="MINIMAX_API_KEY", quota={"limit": 1000}),
            "groq": _backend("groq", auth_env="GROQ_API_KEY", quota={"limit": 3000}),
        })

        with patch.dict("os.environ", {
            "OPENAI_API_KEY": "fake-key",
            "ANTHROPIC_API_KEY": "fake-key",
            "MINIMAX_API_KEY": "fake-key",
            "GROQ_API_KEY": "fake-key",
        }):
            app = create_app(config)

        assert len(app.state.quota_pollers) == 1
        mock_poller_cls.assert_called_once()
        _, kwargs = mock_poller_cls.call_args
        assert kwargs["backend_name"] == "minimax"
