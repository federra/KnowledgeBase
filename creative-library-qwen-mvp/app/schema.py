from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated, Any, Literal

from pydantic import BaseModel, BeforeValidator, Field, field_validator, model_validator


FieldStatus = Literal["known", "ai", "unknown", "confirmed", "edited", "rejected"]
JobStatus = Literal["queued", "processing", "completed", "failed"]
DEFAULT_MODEL_ID = "qwen3.5-omni-plus-2026-03-15"


def _normalize_confidence(value: Any) -> Any:
    if value is None or isinstance(value, bool):
        return value
    try:
        raw = value.strip() if isinstance(value, str) else value
        percentage = isinstance(raw, str) and raw.endswith("%")
        number = float(raw[:-1] if percentage else raw)
    except (TypeError, ValueError):
        return value
    if not percentage and 0 <= number <= 1:
        number *= 100
    return round(number)


ConfidenceScore = Annotated[int, BeforeValidator(_normalize_confidence)]


class FieldSpec(BaseModel):
    key: str
    category: str
    title: str
    requires_comments: bool = False


def _specs(category: str, rows: list[tuple[str, str]], requires_comments: bool = False) -> list[FieldSpec]:
    return [FieldSpec(key=key, category=category, title=title, requires_comments=requires_comments) for key, title in rows]


VIDEO_FIELD_SPECS: list[FieldSpec] = [
    *_specs("responsibility", [
        ("type", "视频类型"), ("episode", "集号"), ("previous", "前集承接"),
        ("next", "后集开启"), ("goal", "本集目标"), ("information", "信息增量"),
        ("change", "状态变化"),
    ]),
    *_specs("timing", [
        ("duration", "时长"), ("scenes", "场景"), ("beats", "情节节拍"),
        ("shots", "镜头"), ("firstframe", "首帧"), ("first3", "前 3 秒"),
        ("first30", "前 30 秒"), ("ending", "集尾"),
    ]),
    *_specs("drama", [
        ("goal", "人物目标"), ("obstacle", "阻碍"), ("stakes", "赌注"),
        ("conflict", "核心冲突"), ("hook", "钩子"), ("reversal", "反转"),
        ("payoff", "爽点"), ("suspense", "悬念"), ("result", "阶段结果"),
    ]),
    *_specs("audiovisual", [
        ("dialogue", "台词"), ("action", "动作"), ("composition", "构图"),
        ("camera", "运镜"), ("visual", "视觉风格"), ("sound", "音效语言"),
        ("bgm", "BGM"), ("retention", "留存表现"), ("comments", "代表评论"),
        ("human", "人工判断"),
    ]),
    *_specs("audience", [
        ("source", "来源与上下文"), ("anchor", "内容锚点"), ("emotion", "情绪"),
        ("stance", "立场"), ("intent", "评论类型"), ("signal", "观众信号簇"),
        ("creative", "创作价值"), ("represent", "群体代表性"),
        ("behavior", "行为印证"), ("quality", "质量风险"),
    ], requires_comments=True),
    *_specs("knowledge", [
        ("fact", "发生了什么"), ("why", "为什么有效"), ("reuse", "如何迁移"),
        ("asset", "资产类型"), ("audience", "适用受众"), ("tasks", "适用任务"),
        ("conditions", "适用条件"), ("boundary", "失效边界"),
        ("variables", "改写变量"), ("result", "版本与使用结果"),
    ]),
]


class AnalysisField(BaseModel):
    id: str = ""
    key: str
    category: str
    title: str
    summary: str
    evidence: str
    time: str
    confidence: ConfidenceScore = Field(default=0, ge=0, le=100)
    status: FieldStatus = "ai"
    applicability: str = "同类内容的结构设计与复核"
    boundary: str = "迁移时必须重新核验人物动机、题材语境与证据链。"
    meta: list[list[str]] = Field(default_factory=list)
    discussion: list[dict[str, str]] = Field(default_factory=list)
    entityType: str | None = None
    details: dict[str, Any] | None = None

    @field_validator("summary", "evidence", "time", "applicability", "boundary", mode="before")
    @classmethod
    def stringify(cls, value: Any) -> str:
        if value is None:
            return ""
        return str(value)

    @field_validator("meta", mode="before")
    @classmethod
    def normalize_meta(cls, value: Any) -> list[list[str]]:
        """Keep optional display metadata from invalidating a full model response."""
        if value in (None, ""):
            return []

        def text(item: Any) -> str:
            if item is None:
                return ""
            if isinstance(item, (dict, list, tuple)):
                return str(item)
            return str(item)

        if isinstance(value, dict):
            return [[text(key), text(item)] for key, item in value.items()]
        if not isinstance(value, (list, tuple)):
            return [["补充", text(value)]]

        rows: list[list[str]] = []
        for item in value:
            if isinstance(item, dict):
                label = item.get("label") or item.get("key") or item.get("title") or item.get("name")
                content = item.get("value") or item.get("content") or item.get("summary")
                if label is not None or content is not None:
                    rows.append([text(label or "补充"), text(content)])
                else:
                    rows.extend([[text(key), text(entry)] for key, entry in item.items()])
            elif isinstance(item, (list, tuple)):
                if not item:
                    continue
                rows.append([text(item[0]), text(item[1]) if len(item) > 1 else ""])
            else:
                rows.append(["补充", text(item)])
        return rows


class StoryboardRow(BaseModel):
    timestamp: str
    shot: str
    audio: str
    visual: str
    dialogue: str
    story_intensity: str
    audience_emotion: str

    @field_validator("story_intensity", "audience_emotion", mode="before")
    @classmethod
    def stringify_score(cls, value: Any) -> str:
        if value is None:
            return ""
        return str(value)


class EmotionCurve(BaseModel):
    labels: list[str]
    values: list[int]
    reasons: list[str]

    @field_validator("values")
    @classmethod
    def validate_values(cls, values: list[int]) -> list[int]:
        return [max(0, min(100, int(value))) for value in values]

    def model_post_init(self, __context: Any) -> None:
        size = min(len(self.labels), len(self.values), len(self.reasons))
        if size < 2:
            raise ValueError("情绪曲线至少需要两个数据点")
        self.labels = self.labels[:size]
        self.values = self.values[:size]
        self.reasons = self.reasons[:size]


class CharacterProfile(BaseModel):
    id: str = ""
    name: str
    aliases: list[str] = Field(default_factory=list)
    gender: str = "待确认"
    role: str = "出场人物"
    faction: str = "信息不足"
    arc: str = "信息不足"
    surface_goal: str = "信息不足"
    deep_goal: str = "信息不足"
    contrast: str = "信息不足"
    weakness: str = "信息不足"
    visual_anchor: str = "信息不足"
    first_seen: str = "时间待确认"
    evidence: str = "缺少可核验证据"
    confidence: ConfidenceScore = Field(default=0, ge=0, le=100)
    status: FieldStatus = "ai"
    possible_matches: list[str] = Field(default_factory=list)

    @field_validator(
        "name", "gender", "role", "faction", "arc", "surface_goal", "deep_goal",
        "contrast", "weakness", "visual_anchor", "first_seen", "evidence", mode="before",
    )
    @classmethod
    def stringify_character_value(cls, value: Any) -> str:
        return "" if value is None else str(value)


class RelationshipProfile(BaseModel):
    id: str = ""
    from_name: str
    to_name: str
    relation: str = "关系待确认"
    evidence: str = "缺少可核验证据"
    confidence: ConfidenceScore = Field(default=0, ge=0, le=100)
    status: FieldStatus = "ai"


class FactionProfile(BaseModel):
    id: str = ""
    name: str
    position: str = "信息不足"
    traits: str = "信息不足"
    goal: str = "信息不足"
    resources: str = "信息不足"
    constraints: str = "信息不足"
    evidence: str = "缺少可核验证据"
    confidence: ConfidenceScore = Field(default=0, ge=0, le=100)
    status: FieldStatus = "ai"


class WorldElementProfile(BaseModel):
    id: str = ""
    category: str = "其他"
    title: str
    description: str = "信息不足"
    evidence: str = "缺少可核验证据"
    confidence: ConfidenceScore = Field(default=0, ge=0, le=100)
    status: FieldStatus = "ai"


class WorldAnalysis(BaseModel):
    background_completeness: str = "待补充解析"
    notice: str = "尚未进行人物与世界专项解析。"
    characters: list[CharacterProfile] = Field(default_factory=list)
    relationships: list[RelationshipProfile] = Field(default_factory=list)
    factions: list[FactionProfile] = Field(default_factory=list)
    elements: list[WorldElementProfile] = Field(default_factory=list)


class VideoOverview(BaseModel):
    title: str = ""
    series_title: str = ""
    episode_title: str = ""
    duration: str
    goal: str
    change: str
    focus: str
    episode: int = 1
    video_type: str = "待确认"
    genre: str = "待确认"
    platform: str = "本地导入"

    @field_validator("video_type", mode="before")
    @classmethod
    def normalize_video_type(cls, value: Any) -> str:
        text = "" if value is None else str(value).strip()
        if "预告" in text:
            return "预告片"
        if any(keyword in text for keyword in ("周边", "花絮", "幕后", "采访", "衍生")):
            return "周边"
        if any(keyword in text for keyword in ("正片", "单集", "剧情")):
            return "正片"
        return "待确认"

    @model_validator(mode="after")
    def align_legacy_title(self) -> "VideoOverview":
        if not self.episode_title:
            self.episode_title = self.title
        if not self.title:
            self.title = self.episode_title
        return self


class ModelAnalysis(BaseModel):
    video: VideoOverview
    fields: list[AnalysisField]
    storyboard: list[StoryboardRow]
    emotion_curve: EmotionCurve
    world: WorldAnalysis = Field(default_factory=WorldAnalysis)


class AnalysisResult(ModelAnalysis):
    job_id: str
    created_at: str
    model_id: str
    source_file: str


class JobRecord(BaseModel):
    id: str
    status: JobStatus = "queued"
    progress: int = Field(default=0, ge=0, le=100)
    stage: str = "等待处理"
    file_name: str
    video_path: str
    subtitle_path: str | None = None
    comments_path: str | None = None
    genre: str = "待确认"
    platform: str = "本地导入"
    series_id: str = "s1"
    series_title: str = ""
    episode_hint: int | None = Field(default=None, ge=1)
    video_type_hint: str = "AI 自动识别"
    error: str | None = None
    analysis_path: str | None = None
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class ModelConfigInput(BaseModel):
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    model_id: str = DEFAULT_MODEL_ID
    api_key: str | None = None

    @field_validator("base_url")
    @classmethod
    def validate_url(cls, value: str) -> str:
        value = value.strip().rstrip("/")
        if not value.startswith(("https://", "http://127.0.0.1", "http://localhost")):
            raise ValueError("Base URL 必须使用 HTTPS；仅本机地址允许 HTTP")
        return value


class ModelConfigView(BaseModel):
    base_url: str
    model_id: str
    configured: bool
    masked_key: str
    mock_mode: bool = False


class ConfigTestResult(BaseModel):
    ok: bool
    message: str
    model_id: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
