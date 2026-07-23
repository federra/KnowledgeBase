from app.schema import (
    VIDEO_FIELD_SPECS,
    CharacterProfile,
    FactionProfile,
    ModelAnalysis,
    RelationshipProfile,
    VideoOverview,
    WorldAnalysis,
    WorldElementProfile,
)


def test_video_field_contract_has_every_page_field() -> None:
    keys = [(item.category, item.key) for item in VIDEO_FIELD_SPECS]
    assert len(keys) == 54
    assert len(set(keys)) == 54
    assert {item.category for item in VIDEO_FIELD_SPECS} == {
        "responsibility", "timing", "drama", "audiovisual", "audience", "knowledge"
    }


def test_audience_fields_require_real_comments() -> None:
    audience = [item for item in VIDEO_FIELD_SPECS if item.category == "audience"]
    assert len(audience) == 10
    assert all(item.requires_comments for item in audience)


def test_video_overview_keeps_series_and_episode_titles_separate() -> None:
    overview = VideoOverview(
        title="第一个转折",
        series_title="她的第二人生",
        duration="96 秒",
        goal="拿回证据",
        change="从隐忍转向公开举证",
        focus="集尾反转",
    )

    assert overview.series_title == "她的第二人生"
    assert overview.episode_title == "第一个转折"
    assert overview.title == "第一个转折"


def test_world_analysis_keeps_each_visible_character_as_a_separate_profile() -> None:
    world = WorldAnalysis(
        background_completeness="信息有限",
        notice="仅基于当前一集，跨集背景仍待补充。",
        characters=[
            CharacterProfile(
                name="女A",
                gender="女",
                role="主要人物",
                first_seen="00:02",
                evidence="00:02 出现在教室并主动发言",
            ),
            CharacterProfile(
                name="男B",
                gender="男",
                role="关系人物",
                first_seen="00:08",
                evidence="00:08 与女A发生对话",
            ),
        ],
    )

    assert [item.name for item in world.characters] == ["女A", "男B"]
    assert world.characters[0].surface_goal == "信息不足"
    assert world.characters[0].status == "ai"


def test_old_analysis_payload_remains_valid_without_world_data() -> None:
    analysis = ModelAnalysis.model_validate({
        "video": {"title": "旧结果", "duration": "10 秒", "goal": "", "change": "", "focus": ""},
        "fields": [],
        "storyboard": [],
        "emotion_curve": {"labels": ["0s", "10s"], "values": [40, 60], "reasons": ["开始", "结束"]},
    })

    assert analysis.world.characters == []
    assert analysis.world.background_completeness == "待补充解析"


def test_world_confidence_accepts_probability_and_fractional_percentages() -> None:
    world = WorldAnalysis(
        characters=[CharacterProfile(name="林枫", confidence=0.95)],
        relationships=[RelationshipProfile(from_name="林枫", to_name="女A", confidence=0.9)],
        factions=[FactionProfile(name="幸存者", confidence=1)],
        elements=[WorldElementProfile(title="荒岛", confidence=87.5)],
    )

    assert world.characters[0].confidence == 95
    assert world.relationships[0].confidence == 90
    assert world.factions[0].confidence == 100
    assert world.elements[0].confidence == 88


def test_video_type_is_normalized_to_supported_options() -> None:
    base = {"title": "测试", "duration": "10 秒", "goal": "", "change": "", "focus": ""}

    assert VideoOverview(**base, video_type="正片单集").video_type == "正片"
    assert VideoOverview(**base, video_type="先导预告").video_type == "预告片"
    assert VideoOverview(**base, video_type="幕后花絮").video_type == "周边"
