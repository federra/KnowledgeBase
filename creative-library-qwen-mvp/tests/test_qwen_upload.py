import json
from pathlib import Path
from types import SimpleNamespace

from app import qwen
from app.qwen import QwenClient, build_analysis_prompt, build_world_prompt


def test_analysis_prompt_treats_upload_genre_as_hint_and_curve_as_content_prediction() -> None:
    prompt = build_analysis_prompt(
        file_name="episode.mp4",
        genre="女性成长",
        platform="本地导入",
        subtitle="",
        comments="一条评论",
        series_title="测试剧",
        video_type_hint="预告片",
    )

    assert "题材提示=女性成长" in prompt
    assert "必须根据视频内容独立识别题材" in prompt
    assert '"genre":"根据视频内容识别"' in prompt
    assert "视频类型提示=预告片" in prompt
    assert '"video_type":"正片/预告片/周边"' in prompt
    assert "情绪曲线不得使用评论" in prompt
    assert "不是播放、留存或评论统计指标" in prompt


def test_world_prompt_requires_aliases_for_unnamed_visible_people_and_no_invention() -> None:
    prompt = build_world_prompt("episode.mp4", "测试剧", "第 1 集")

    assert "每个实际出场人物都必须单独建立角色档案" in prompt
    assert "女A、男B" in prompt
    assert "背景信息不全" in prompt
    assert "禁止为了完整而虚构" in prompt


def test_video_is_uploaded_to_temporary_oss_before_model_request(monkeypatch, tmp_path: Path) -> None:
    video_path = tmp_path / "episode.mp4"
    video_path.write_bytes(b"video-bytes")
    calls: dict[str, object] = {}

    policy = {
        "upload_dir": "dashscope-instant/test",
        "oss_access_key_id": "access-id",
        "signature": "signature",
        "policy": "policy",
        "x_oss_object_acl": "private",
        "x_oss_forbid_overwrite": "true",
        "upload_host": "https://upload.example.invalid",
    }

    class FakeResponse:
        status_code = 200
        text = ""

        def __init__(self, payload=None) -> None:
            self.payload = payload or {}

        def json(self):
            return self.payload

    class FakeHttpClient:
        def __init__(self, **kwargs) -> None:
            calls["http_options"] = kwargs

        def __enter__(self):
            return self

        def __exit__(self, *args) -> None:
            return None

        def get(self, url, *, headers, params):
            calls["policy_request"] = (url, headers, params)
            return FakeResponse({"data": policy})

        def post(self, url, *, files):
            calls["upload_url"] = url
            calls["uploaded_bytes"] = files["file"][1].read()
            return FakeResponse()

    monkeypatch.setattr(qwen, "httpx", SimpleNamespace(Client=FakeHttpClient), raising=False)

    model_payload = json.dumps({
        "video": {"title": "测试", "duration": "10 秒", "goal": "目标", "change": "变化", "focus": "重点"},
        "fields": [],
        "storyboard": [],
        "emotion_curve": {"labels": ["0s", "10s"], "values": [30, 60], "reasons": ["开始", "结束"]},
    }, ensure_ascii=False)

    class FakeCompletions:
        def create(self, **kwargs):
            calls["model_request"] = kwargs
            return [SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=model_payload))])]

    client = QwenClient.__new__(QwenClient)
    client.api_key = "sk-test"
    client.model_id = "qwen3.5-omni-plus"
    client.client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))

    client.analyze_video(
        video_path=str(video_path),
        file_name=video_path.name,
        genre="女性成长",
        platform="本地导入",
        subtitle_path=None,
        comments_path=None,
    )

    policy_request = calls["policy_request"]
    assert policy_request[2] == {"action": "getPolicy", "model": "qwen3.5-omni-plus"}
    assert calls["uploaded_bytes"] == b"video-bytes"
    model_request = calls["model_request"]
    video_url = model_request["messages"][0]["content"][0]["video_url"]["url"]
    assert video_url == "oss://dashscope-instant/test/episode.mp4"
    assert model_request["extra_headers"] == {"X-DashScope-OssResourceResolve": "enable"}
