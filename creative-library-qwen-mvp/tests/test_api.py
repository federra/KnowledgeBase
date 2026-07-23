import os

os.environ["QWEN_MOCK_MODE"] = "1"

from fastapi.testclient import TestClient

from app.main import app, normalize_analysis
from app.schema import EmotionCurve, JobRecord, ModelAnalysis, VideoOverview


client = TestClient(app)


def test_health_and_masked_config() -> None:
    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["ok"] is True

    response = client.put("/api/config", json={
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model_id": "qwen3.5-omni-plus",
        "api_key": "sk-ws-test-secret",
    })
    assert response.status_code == 200
    assert "test-secret" not in response.text
    assert response.json()["masked_key"].endswith("cret")


def test_rejects_token_plan_credentials_for_video_backend() -> None:
    response = client.put("/api/config", json={
        "base_url": "https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
        "model_id": "qwen3.5-omni-plus",
        "api_key": "sk-sp-not-for-app-backends",
    })

    assert response.status_code == 400
    assert "Token Plan" in response.json()["detail"]
    assert "sk-" in response.json()["detail"]


def test_mock_upload_returns_all_fields_without_invented_audience_data() -> None:
    client.put("/api/config", json={
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model_id": "qwen3.5-omni-plus",
        "api_key": "sk-ws-test-secret",
    })
    response = client.post(
        "/api/jobs",
        files={"video": ("测试短剧.mp4", b"not-a-real-video", "video/mp4")},
        data={"genre": "女性成长", "platform": "红果", "series_id": "s1"},
    )
    assert response.status_code == 202
    job_id = response.json()["id"]

    job = client.get(f"/api/jobs/{job_id}").json()
    assert job["status"] == "completed"
    assert job["progress"] == 100

    analysis = client.get(f"/api/analyses/{job_id}")
    assert analysis.status_code == 200
    payload = analysis.json()
    assert len(payload["fields"]) == 57
    assert len(payload["storyboard"]) >= 6
    assert len(payload["emotion_curve"]["labels"]) >= 6
    audience = [field for field in payload["fields"] if field["category"] == "audience"]
    assert len(audience) == 13
    assert all(field["status"] == "unknown" for field in audience)


def test_upload_episode_hint_is_kept_in_the_final_analysis() -> None:
    response = client.post(
        "/api/jobs",
        files={"video": ("第七集.mp4", b"not-a-real-video", "video/mp4")},
        data={
            "genre": "AI 自动识别",
            "platform": "本地导入",
            "series_id": "s-episode",
            "series_title": "荒岛求生",
            "episode": "7",
        },
    )

    assert response.status_code == 202
    job = response.json()
    assert job["episode_hint"] == 7
    analysis = client.get(f"/api/analyses/{job['id']}").json()
    assert analysis["video"]["episode"] == 7


def test_upload_video_type_hint_is_kept_in_the_final_analysis() -> None:
    response = client.post(
        "/api/jobs",
        files={"video": ("预告.mp4", b"not-a-real-video", "video/mp4")},
        data={
            "platform": "本地导入",
            "series_id": "s-video-type",
            "series_title": "荒岛求生",
            "video_type": "预告片",
        },
    )

    assert response.status_code == 202
    job = response.json()
    assert job["video_type_hint"] == "预告片"
    analysis = client.get(f"/api/analyses/{job['id']}").json()
    assert analysis["video"]["video_type"] == "预告片"


def test_rejects_unsupported_video_format() -> None:
    response = client.post(
        "/api/jobs",
        files={"video": ("bad.exe", b"bad", "application/octet-stream")},
        data={"genre": "现实职场", "platform": "本地导入", "series_id": "s1"},
    )
    assert response.status_code == 415


def test_normalization_does_not_overwrite_ai_genre_with_upload_hint() -> None:
    job = JobRecord(
        id="genre-test",
        file_name="episode.mp4",
        video_path="/tmp/episode.mp4",
        genre="女性成长",
        platform="本地导入",
    )
    analysis = ModelAnalysis(
        video=VideoOverview(
            title="测试集",
            duration="10 秒",
            goal="",
            change="",
            focus="",
            genre="校园奇幻",
        ),
        fields=[],
        storyboard=[],
        emotion_curve=EmotionCurve(labels=["0s", "10s"], values=[40, 60], reasons=["开始", "结束"]),
    )

    result = normalize_analysis(job, analysis, "test-model")

    assert result.video.genre == "校园奇幻"


def test_existing_job_can_request_focused_world_reanalysis() -> None:
    client.put("/api/config", json={
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model_id": "qwen3.5-omni-plus",
        "api_key": "sk-ws-test-secret",
    })
    created = client.post(
        "/api/jobs",
        files={"video": ("人物测试.mp4", b"not-a-real-video", "video/mp4")},
        data={"genre": "AI 自动识别", "platform": "本地导入", "series_id": "s-world", "series_title": "人物测试"},
    )
    job_id = created.json()["id"]

    response = client.post(f"/api/jobs/{job_id}/reanalyze-world")

    assert response.status_code == 202
    analysis = client.get(f"/api/analyses/{job_id}").json()
    assert analysis["world"]["characters"]
    assert analysis["world"]["characters"][0]["name"]
