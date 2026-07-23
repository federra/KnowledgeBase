import json

import pytest

from app.qwen import parse_model_analysis


def valid_payload() -> dict:
    return {
        "video": {
            "title": "测试集",
            "duration": "96 秒",
            "goal": "完成目标",
            "change": "人物状态发生变化",
            "focus": "冲突与反转",
        },
        "fields": [],
        "storyboard": [],
        "emotion_curve": {
            "labels": ["0s", "96s"],
            "values": [30, 70],
            "reasons": ["冲突建立", "结果落地"],
        },
    }


def test_parse_model_analysis_accepts_markdown_fenced_json() -> None:
    text = f"```json\n{json.dumps(valid_payload(), ensure_ascii=False)}\n```"

    result = parse_model_analysis(text)

    assert result.video.title == "测试集"


def test_parse_model_analysis_repairs_trailing_commas() -> None:
    text = json.dumps(valid_payload(), ensure_ascii=False)
    text = text.replace('"结果落地"]', '"结果落地",]')

    result = parse_model_analysis(text)

    assert result.emotion_curve.values == [30, 70]


def test_parse_model_analysis_accepts_numeric_story_intensity() -> None:
    payload = valid_payload()
    payload["storyboard"] = [{
        "timestamp": "00:00-00:03",
        "shot": "特写",
        "audio": "提示音",
        "visual": "冲突出现",
        "dialogue": "你是谁？",
        "story_intensity": 80,
        "audience_emotion": 72,
    }]

    result = parse_model_analysis(json.dumps(payload, ensure_ascii=False))

    assert result.storyboard[0].story_intensity == "80"
    assert result.storyboard[0].audience_emotion == "72"


def test_parse_model_analysis_normalizes_flat_meta_notes() -> None:
    payload = valid_payload()
    payload["fields"] = [{
        "key": "comments",
        "category": "audiovisual",
        "title": "代表评论",
        "summary": "暂无真实评论数据可供分析。",
        "evidence": "未提供评论和行为数据。",
        "time": "N/A",
        "confidence": 0,
        "status": "unknown",
        "meta": ["缺少评论区抓取权限", "无弹幕或互动记录"],
    }]

    result = parse_model_analysis(json.dumps(payload, ensure_ascii=False))

    assert result.fields[0].meta == [
        ["补充", "缺少评论区抓取权限"],
        ["补充", "无弹幕或互动记录"],
    ]


def test_parse_model_analysis_distinguishes_schema_errors_from_invalid_json() -> None:
    payload = valid_payload()
    payload["emotion_curve"]["values"] = "高"

    with pytest.raises(RuntimeError, match="字段格式不符合要求"):
        parse_model_analysis(json.dumps(payload, ensure_ascii=False))


def test_parse_model_analysis_reports_a_short_actionable_error() -> None:
    with pytest.raises(RuntimeError, match="模型返回的 JSON 无法自动修复") as exc_info:
        parse_model_analysis("```json\n{完全不是 JSON}\n```")

    assert len(str(exc_info.value)) < 500
