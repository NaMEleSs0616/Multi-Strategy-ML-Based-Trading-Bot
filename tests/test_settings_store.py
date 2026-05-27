"""Settings store round-trip tests."""

from config.settings_store import DEFAULTS, load_settings, save_settings


def test_load_settings_has_defaults(tmp_path):
    path = tmp_path / "settings.yaml"
    settings = load_settings(path)
    assert settings["rl"]["embedding_dim"] == DEFAULTS["rl"]["embedding_dim"]
    assert "walk_forward" in settings


def test_save_and_reload(tmp_path):
    path = tmp_path / "settings.yaml"
    settings = load_settings(path)
    settings["rl"]["turnover_penalty_lambda"] = 0.25
    save_settings(settings, path)
    reloaded = load_settings(path)
    assert reloaded["rl"]["turnover_penalty_lambda"] == 0.25
