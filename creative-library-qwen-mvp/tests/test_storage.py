import json
from pathlib import Path

from app.schema import ModelConfigInput
from app.storage import ConfigStore, safe_filename


def test_config_never_persists_api_key(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    store = ConfigStore(path)
    view = store.update(ModelConfigInput(
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model_id="qwen3.5-omni-plus",
        api_key="sk-ws-secret-value",
    ))

    assert view.configured is True
    assert view.masked_key.endswith("alue")
    persisted = json.loads(path.read_text(encoding="utf-8"))
    assert "api_key" not in persisted
    assert "secret" not in path.read_text(encoding="utf-8")


def test_safe_filename_removes_path_and_symbols() -> None:
    assert safe_filename("../../一部 剧?.mp4") == "一部_剧_.mp4"


def test_legacy_default_model_is_migrated_to_pinned_version(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(json.dumps({
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model_id": "qwen3.5-omni-plus",
    }), encoding="utf-8")

    store = ConfigStore(path)

    assert store.model_id == "qwen3.5-omni-plus-2026-03-15"
