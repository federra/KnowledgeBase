from __future__ import annotations

import json
import mimetypes
import os
from pathlib import Path
from typing import Any

import httpx
from json_repair import repair_json
from openai import OpenAI
from pydantic import ValidationError

from .schema import (
    EmotionCurve,
    ModelAnalysis,
    StoryboardRow,
    VideoOverview,
    WorldAnalysis,
    VIDEO_FIELD_SPECS,
)


UPLOAD_POLICY_URL = "https://dashscope.aliyuncs.com/api/v1/uploads"


def _extract_json_candidate(text: str) -> str:
    candidate = text.strip()
    if candidate.startswith("```"):
        first_newline = candidate.find("\n")
        candidate = candidate[first_newline + 1:] if first_newline >= 0 else candidate[3:]
    if candidate.endswith("```"):
        candidate = candidate[:-3].rstrip()
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start >= 0 and end > start:
        candidate = candidate[start:end + 1]
    return candidate


def parse_model_analysis(text: str) -> ModelAnalysis:
    candidate = _extract_json_candidate(text)
    try:
        payload = json.loads(candidate)
        repaired_syntax = False
    except json.JSONDecodeError:
        try:
            payload = repair_json(candidate, return_objects=True)
            repaired_syntax = True
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                "模型返回的 JSON 无法自动修复。请重新解析；如果重复出现，"
                "可更换模型或缩小单次分析字段范围。"
            ) from exc
    try:
        return ModelAnalysis.model_validate(payload)
    except ValidationError as exc:
        if repaired_syntax:
            raise RuntimeError(
                "模型返回的 JSON 无法自动修复。请重新解析；如果重复出现，"
                "可更换模型或缩小单次分析字段范围。"
            ) from exc
        first_error = exc.errors(include_url=False)[0]
        location = ".".join(str(part) for part in first_error["loc"])
        raise RuntimeError(
            f"模型返回的字段格式不符合要求：{location}，{first_error['msg']}。"
        ) from exc


def parse_world_analysis(text: str) -> WorldAnalysis:
    candidate = _extract_json_candidate(text)
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        try:
            payload = repair_json(candidate, return_objects=True)
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            raise RuntimeError("人物与世界专项解析返回的 JSON 无法自动修复，请重试。") from exc
    try:
        return WorldAnalysis.model_validate(payload)
    except ValidationError as exc:
        first_error = exc.errors(include_url=False)[0]
        location = ".".join(str(part) for part in first_error["loc"])
        raise RuntimeError(f"人物与世界字段格式不符合要求：{location}，{first_error['msg']}。") from exc


def _save_failed_model_response(video_path: str, text: str) -> Path | None:
    try:
        source = Path(video_path)
        output_dir = source.parent.parent / "raw-responses"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{source.stem}.txt"
        output_path.write_text(text, encoding="utf-8")
        return output_path
    except OSError:
        return None


def _read_optional(path: str | None, limit: int = 60_000) -> str:
    if not path:
        return ""
    try:
        return Path(path).read_text(encoding="utf-8", errors="ignore")[:limit]
    except OSError:
        return ""


def build_analysis_prompt(
    file_name: str,
    genre: str,
    platform: str,
    subtitle: str,
    comments: str,
    series_title: str = "",
    episode_hint: int | None = None,
    video_type_hint: str = "AI 自动识别",
) -> str:
    field_contract = [spec.model_dump() for spec in VIDEO_FIELD_SPECS]
    audience_rule = (
        "已提供评论文本。观众字段只能依据评论原文分析，禁止生成播放量、留存率、增长率等未提供指标。"
        if comments
        else "未提供评论和行为数据。category=audience 的全部字段必须 status=unknown、confidence=0，并明确缺少何种数据；禁止编造评论和指标。"
    )
    return f"""
你是AI漫剧内容解剖专家。请完整观看视频画面并理解声音，输出可供人工审核的JSON。不要输出Markdown。

素材信息：文件名={file_name}；当前归属剧名={series_title or '待识别'}；人工集数提示={episode_hint or '未提供'}；视频类型提示={video_type_hint or 'AI 自动识别'}；题材提示={genre or '未提供'}；平台={platform}。
题材提示只供参考，你必须根据视频内容独立识别题材；可输出“主类型 / 次类型”，不得照抄错误提示。
人工集数提示已提供时必须写入video.episode；未提供时根据文件名、片头字幕和画面内容识别，无法确认时再暂填1，解析后由人工修改。
视频类型提示为正片、预告片或周边时必须写入video.video_type；为AI自动识别时，依据是否呈现连续剧情、宣发剪辑或幕后衍生内容判断。
{audience_rule}

必须返回以下顶层结构：
{{
  "video": {{"title":"集标题（兼容字段）", "series_title":"剧名", "episode_title":"集标题", "duration":"", "goal":"", "change":"", "focus":"", "episode":1, "video_type":"正片/预告片/周边", "genre":"根据视频内容识别", "platform":"{platform}"}},
  "fields": [{{"key":"", "category":"", "title":"", "summary":"", "evidence":"", "time":"", "confidence":0, "status":"ai", "applicability":"", "boundary":"", "meta":[]}}],
  "storyboard": [{{"timestamp":"", "shot":"", "audio":"", "visual":"", "dialogue":"", "story_intensity":"", "audience_emotion":""}}],
  "emotion_curve": {{"labels":[], "values":[], "reasons":[]}}
}}

字段规则：
1. fields必须逐项覆盖下面清单，key+category不可改变，不可增加或遗漏。
2. 每个结论都要给出可回看视频的时间范围和具体证据；不确定就标记unknown，不能猜测。
3. video.video_type只能为正片、预告片或周边；status只能为ai或unknown；confidence为0到100整数。
4. storyboard按连续时间段覆盖全片，至少6行；台词无法辨认时写“未辨认”，不要杜撰；story_intensity与audience_emotion都用字符串输出。
5. emotion_curve给出6到10个按时间排序的数据点；values是普通观众随剧情推进可能产生的0到100定性情绪值，只要求相对起伏合理。每个reason必须说明对应时刻发生了什么，以及它为何可能改变观众情绪。情绪曲线不得使用评论、弹幕或平台数据作为绘制依据，它不是播放、留存或评论统计指标。
6. 视听分析要区分画面事实、声音事实和专业解释；留存表现没有真实行为数据时只能写AI定性判断。
7. “人工判断”字段必须status=unknown，留给人工填写。
8. meta必须是二维字符串数组，每一项严格写成["标签","值"]；没有补充信息时写[]。禁止输出字符串列表或对象，例如["说明一","说明二"]是不合法格式。
9. 输出必须是严格合法 JSON：不要使用代码围栏，不要写尾逗号，字符串内部的引号、换行和反斜杠必须正确转义。

字段清单：{json.dumps(field_contract, ensure_ascii=False)}

可选字幕：
{subtitle or '未提供'}

可选评论：
{comments or '未提供'}
""".strip()


def build_world_prompt(file_name: str, series_title: str, episode_title: str) -> str:
    return f"""
你是AI漫剧人物与世界观分析专家。请完整观看视频画面并理解声音，只依据当前素材输出严格合法JSON，不要输出Markdown。

素材：文件名={file_name}；当前归属剧名={series_title or '待识别'}；集标题={episode_title or '待识别'}。

必须返回：
{{
  "background_completeness":"充分/部分/信息有限",
  "notice":"说明当前材料足以确认什么、仍缺少什么",
  "characters":[{{"name":"", "aliases":[], "gender":"", "role":"", "faction":"", "arc":"", "surface_goal":"", "deep_goal":"", "contrast":"", "weakness":"", "visual_anchor":"", "first_seen":"", "evidence":"", "confidence":0, "status":"ai", "possible_matches":[]}}],
  "relationships":[{{"from_name":"", "to_name":"", "relation":"", "evidence":"", "confidence":0, "status":"ai"}}],
  "factions":[{{"name":"", "position":"", "traits":"", "goal":"", "resources":"", "constraints":"", "evidence":"", "confidence":0, "status":"ai"}}],
  "elements":[{{"category":"阶层/规则/资源/禁忌/基本矛盾/时空背景/其他", "title":"", "description":"", "evidence":"", "confidence":0, "status":"ai"}}]
}}

规则：
1. 每个实际出场人物都必须单独建立角色档案，不得把多人挤在一个字段中。
2. 无法确认姓名时使用稳定代号，如女A、男B；同类多人再按女B、男A、人物A等顺序扩展。不要因没有姓名而遗漏角色。
3. 能从行动、对白、服装、空间位置判断的信息尽量填写；无法判断的属性写“信息不足”，并降低confidence。
4. 只有画面或声音中有证据时才建立阵营和世界观要素；素材较少时必须在notice明确写“背景信息不全”。
5. 禁止为了完整而虚构未出现的人物、关系、阵营、规则或跨集剧情。
6. possible_matches只填写当前人物可能对应的已知姓名或代号；没有候选时返回空数组。
7. confidence必须输出0到100的整数，例如90；禁止输出0.9、0.95等0到1的小数概率。
8. status只能为ai或unknown；所有证据需给出时间点或时间段。
""".strip()


class QwenClient:
    def __init__(self, api_key: str, base_url: str, model_id: str) -> None:
        if not api_key:
            raise ValueError("尚未配置百炼 API Key")
        self.api_key = api_key
        self.model_id = model_id
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=900.0, max_retries=1)

    def _upload_to_temporary_oss(self, video_path: str, file_name: str) -> str:
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        params = {"action": "getPolicy", "model": self.model_id}
        with httpx.Client(timeout=900.0, follow_redirects=True) as client:
            policy_response = client.get(UPLOAD_POLICY_URL, headers=headers, params=params)
            if policy_response.status_code != 200:
                if policy_response.status_code in {401, 403}:
                    raise RuntimeError(
                        "百炼按量付费 API Key 无效或无临时文件权限。"
                        "请使用 sk- 开头的按量付费 Key，不要使用 sk-sp- Token Plan Key。"
                    )
                raise RuntimeError(f"获取百炼临时上传凭证失败：{policy_response.text[:500]}")
            try:
                policy = policy_response.json()["data"]
                key = f"{policy['upload_dir']}/{file_name}"
                mime_type = mimetypes.guess_type(file_name)[0] or "video/mp4"
                with Path(video_path).open("rb") as source:
                    files = {
                        "OSSAccessKeyId": (None, policy["oss_access_key_id"]),
                        "Signature": (None, policy["signature"]),
                        "policy": (None, policy["policy"]),
                        "x-oss-object-acl": (None, policy["x_oss_object_acl"]),
                        "x-oss-forbid-overwrite": (None, policy["x_oss_forbid_overwrite"]),
                        "key": (None, key),
                        "success_action_status": (None, "200"),
                        "file": (file_name, source, mime_type),
                    }
                    upload_response = client.post(policy["upload_host"], files=files)
            except (KeyError, TypeError, ValueError) as exc:
                raise RuntimeError("百炼返回了无效的临时上传凭证") from exc
            if upload_response.status_code != 200:
                raise RuntimeError(f"视频上传至百炼临时存储失败：{upload_response.text[:500]}")
        return f"oss://{key}"

    def _collect_stream(self, stream: Any) -> str:
        chunks: list[str] = []
        for chunk in stream:
            if not getattr(chunk, "choices", None):
                continue
            content = getattr(chunk.choices[0].delta, "content", None)
            if isinstance(content, str):
                chunks.append(content)
            elif isinstance(content, list):
                for item in content:
                    text = item.get("text") if isinstance(item, dict) else getattr(item, "text", None)
                    if text:
                        chunks.append(text)
        return "".join(chunks).strip()

    def test_connection(self) -> str:
        stream = self.client.chat.completions.create(
            model=self.model_id,
            messages=[{"role": "user", "content": "只回复：连接成功"}],
            stream=True,
            stream_options={"include_usage": True},
            modalities=["text"],
            extra_body={"enable_thinking": False},
        )
        return self._collect_stream(stream) or "连接成功"

    def analyze_video(
        self,
        video_path: str,
        file_name: str,
        genre: str,
        platform: str,
        subtitle_path: str | None,
        comments_path: str | None,
        series_title: str = "",
        episode_hint: int | None = None,
        video_type_hint: str = "AI 自动识别",
    ) -> ModelAnalysis:
        video_url = self._upload_to_temporary_oss(video_path, file_name)
        prompt = build_analysis_prompt(
            file_name=file_name,
            genre=genre,
            platform=platform,
            subtitle=_read_optional(subtitle_path),
            comments=_read_optional(comments_path),
            series_title=series_title,
            episode_hint=episode_hint,
            video_type_hint=video_type_hint,
        )
        stream = self.client.chat.completions.create(
            model=self.model_id,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "video_url", "video_url": {"url": video_url}},
                    {"type": "text", "text": prompt},
                ],
            }],
            response_format={"type": "json_object"},
            stream=True,
            stream_options={"include_usage": True},
            modalities=["text"],
            extra_body={"enable_thinking": False},
            extra_headers={"X-DashScope-OssResourceResolve": "enable"},
        )
        text = self._collect_stream(stream)
        try:
            return parse_model_analysis(text)
        except RuntimeError as exc:
            debug_path = _save_failed_model_response(video_path, text)
            suffix = f" 原始响应已保存到：{debug_path}" if debug_path else ""
            raise RuntimeError(f"{exc}{suffix}") from exc

    def analyze_world(
        self,
        video_path: str,
        file_name: str,
        series_title: str = "",
        episode_title: str = "",
    ) -> WorldAnalysis:
        video_url = self._upload_to_temporary_oss(video_path, file_name)
        prompt = build_world_prompt(file_name, series_title, episode_title)
        stream = self.client.chat.completions.create(
            model=self.model_id,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "video_url", "video_url": {"url": video_url}},
                    {"type": "text", "text": prompt},
                ],
            }],
            response_format={"type": "json_object"},
            stream=True,
            stream_options={"include_usage": True},
            modalities=["text"],
            extra_body={"enable_thinking": False},
            extra_headers={"X-DashScope-OssResourceResolve": "enable"},
        )
        text = self._collect_stream(stream)
        try:
            return parse_world_analysis(text)
        except RuntimeError as exc:
            debug_path = _save_failed_model_response(video_path, text)
            suffix = f" 原始响应已保存到：{debug_path}" if debug_path else ""
            raise RuntimeError(f"{exc}{suffix}") from exc


class MockQwenClient:
    model_id = "mock-qwen3.5-omni-plus-2026-03-15"

    def test_connection(self) -> str:
        return "Mock连接成功"

    def analyze_video(
        self,
        video_path: str,
        file_name: str,
        genre: str,
        platform: str,
        subtitle_path: str | None,
        comments_path: str | None,
        series_title: str = "",
        episode_hint: int | None = None,
        video_type_hint: str = "AI 自动识别",
    ) -> ModelAnalysis:
        has_comments = bool(comments_path)
        fields = []
        for index, spec in enumerate(VIDEO_FIELD_SPECS):
            unknown = spec.requires_comments and not has_comments
            if spec.key == "human" and spec.category == "audiovisual":
                unknown = True
            fields.append({
                "key": spec.key,
                "category": spec.category,
                "title": spec.title,
                "summary": "缺少真实评论或行为数据，等待补充后分析。" if unknown else f"{spec.title}的Mock解析结论，依据视频中的可见行动与声音变化。",
                "evidence": "需要接入评论原文和采集记录。" if unknown else "00:00–集尾的画面、对白与声音证据。",
                "time": "数据待接入" if unknown else "全片",
                "confidence": 0 if unknown else 82 + index % 14,
                "status": "unknown" if unknown else "ai",
                "applicability": "同类短剧的视频拆解与创作复核",
                "boundary": "仅基于当前视频，跨集和真实观众结论需要额外数据。",
                "meta": [],
            })
        return ModelAnalysis(
            video=VideoOverview(
                title=Path(file_name).stem,
                series_title=series_title,
                episode_title=Path(file_name).stem,
                duration="约 96 秒",
                goal="主角必须用可见行动改变当前关系位置",
                change="从被动承受转为主动提出选择",
                focus="结果前置、关系阻碍、证据变化与集尾新问题",
                episode=episode_hint or 1,
                video_type=video_type_hint if video_type_hint in {"正片", "预告片", "周边"} else "正片",
                genre=genre,
                platform=platform,
            ),
            fields=fields,
            storyboard=[
                StoryboardRow(timestamp="00:00–00:03", shot="特写 / 推近", audio="环境声突停", visual="关键结果先出现", dialogue="未辨认", story_intensity="中 → 高", audience_emotion="疑惑 → 紧张"),
                StoryboardRow(timestamp="00:03–00:18", shot="中景 / 横移", audio="对白进入", visual="主角进入冲突空间", dialogue="先确认发生了什么。", story_intensity="高 → 中", audience_emotion="紧张 → 压抑"),
                StoryboardRow(timestamp="00:18–00:36", shot="双人近景", audio="BGM收弱", visual="对手重新定义问题", dialogue="这不属于你。", story_intensity="中 → 高", audience_emotion="压抑 → 愤怒"),
                StoryboardRow(timestamp="00:36–00:54", shot="固定镜头", audio="低频渐入", visual="主角发现信息差", dialogue="把原始记录拿出来。", story_intensity="高 → 中", audience_emotion="愤怒 → 期待"),
                StoryboardRow(timestamp="00:54–01:12", shot="交叉特写", audio="提示音", visual="新证据改变解释权", dialogue="记录还在。", story_intensity="中 → 峰值", audience_emotion="期待 → 解气"),
                StoryboardRow(timestamp="01:12–集尾", shot="近景 / 拉远", audio="BGM短停", visual="结果产生下一集问题", dialogue="我要一个明确答案。", story_intensity="峰值 → 高", audience_emotion="解气 → 笃定"),
            ],
            emotion_curve=EmotionCurve(
                labels=["0s", "3s", "18s", "36s", "54s", "72s", "集尾"],
                values=[42, 66, 34, 31, 49, 92, 80],
                reasons=["异常结果前置", "目标出现", "规则压制成立", "冲突尚未释放", "主角开始反击", "证据改变权力位置", "结果落地并留下新问题"],
            ),
        )

    def analyze_world(
        self,
        video_path: str,
        file_name: str,
        series_title: str = "",
        episode_title: str = "",
    ) -> WorldAnalysis:
        return WorldAnalysis.model_validate({
            "background_completeness": "信息有限",
            "notice": "背景信息不全；当前仅依据本集出场、对白与可见行动建立档案。",
            "characters": [
                {"name": "女A", "gender": "女", "role": "主要人物", "faction": "待确认", "surface_goal": "完成当前行动", "first_seen": "00:02", "evidence": "00:02 首次出场并主动推动事件", "confidence": 82, "status": "ai"},
                {"name": "男A", "gender": "男", "role": "关系人物", "faction": "待确认", "surface_goal": "阻止女A行动", "first_seen": "00:08", "evidence": "00:08 与女A发生直接对话", "confidence": 76, "status": "ai"},
            ],
            "relationships": [
                {"from_name": "女A", "to_name": "男A", "relation": "当前目标冲突", "evidence": "00:08–00:18 对话立场相反", "confidence": 75, "status": "ai"}
            ],
            "factions": [],
            "elements": [
                {"category": "基本矛盾", "title": "当前关系冲突", "description": "人物围绕当前行动目标产生对抗", "evidence": "00:08–00:18", "confidence": 70, "status": "ai"}
            ],
        })


def create_qwen_client(api_key: str, base_url: str, model_id: str) -> QwenClient | MockQwenClient:
    if os.getenv("QWEN_MOCK_MODE") == "1":
        return MockQwenClient()
    return QwenClient(api_key=api_key, base_url=base_url, model_id=model_id)
