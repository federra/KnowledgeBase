from __future__ import annotations

import os
import shutil
import uuid
from pathlib import Path
from typing import Annotated

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .qwen import create_qwen_client
from .schema import (
    AnalysisField,
    AnalysisResult,
    ConfigTestResult,
    JobRecord,
    ModelAnalysis,
    ModelConfigInput,
    ModelConfigView,
    WorldAnalysis,
    VIDEO_FIELD_SPECS,
    utc_now,
)
from .storage import (
    ANALYSIS_DIR,
    ROOT,
    UPLOAD_DIR,
    ConfigStore,
    JobStore,
    ensure_runtime_dirs,
    read_json,
    safe_filename,
    write_json,
)


MAX_VIDEO_BYTES = 80 * 1024 * 1024
MAX_AUX_BYTES = 8 * 1024 * 1024
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".flv", ".wmv"}
AUX_EXTENSIONS = {".srt", ".vtt", ".txt", ".csv"}

ensure_runtime_dirs()
config_store = ConfigStore()
job_store = JobStore()

app = FastAPI(title="创意库 Qwen MVP", version="0.1.0")


TOKEN_PLAN_MESSAGE = (
    "Token Plan/Coding Plan 凭证不能用于本应用的视频分析后端。"
    "请在百炼控制台创建按量付费 API Key（sk- 开头），并使用 "
    "https://dashscope.aliyuncs.com/compatible-mode/v1"
)


def _validate_app_credentials(base_url: str, api_key: str) -> None:
    normalized_url = base_url.lower()
    normalized_key = api_key.strip().lower()
    if "token-plan" in normalized_url or "coding.dashscope" in normalized_url or normalized_key.startswith("sk-sp-"):
        raise HTTPException(status_code=400, detail=TOKEN_PLAN_MESSAGE)


def _sanitized_error(exc: Exception) -> str:
    message = str(exc) or exc.__class__.__name__
    if config_store.api_key:
        message = message.replace(config_store.api_key, "***")
    return message[:800]


def _platform_fields(job_id: str, platform: str, has_comments: bool) -> list[AnalysisField]:
    platforms = ["红果", "抖音", "快手"]
    rows: list[AnalysisField] = []
    for name in platforms:
        available = has_comments and platform == name
        rows.append(AnalysisField(
            id=f"{job_id}-audience-platform-{name}",
            key=f"platform-{name}",
            category="audience",
            title=name,
            summary=("已提供该平台评论文件，可结合下方观众字段进行人工核验。" if available else "尚未接入可核验的平台数据，暂不生成量化指标与观众结论。"),
            evidence=("本次上传的评论文件" if available else "需要接入平台评论、弹幕、留存及采集批次记录后分析"),
            time=("评论文件" if available else "数据待接入"),
            confidence=(70 if available else 0),
            status=("ai" if available else "unknown"),
            applicability="分平台观众洞察与内容优化",
            boundary="平台人群差异不可直接外推。",
            entityType="platform",
            details={
                "platform": name,
                "source": "本次上传评论文件" if available else "待接入真实平台数据",
                "note": "仅依据用户上传文本，不生成未提供的播放、增长和留存指标。" if available else "当前仅保留分析位置，不展示无依据的反馈量、共鸣度、增长率或代表评论。",
                "fields": ["有效反馈规模", "高频情绪", "内容锚点", "行为变化", "代表评论"],
            },
        ))
    return rows


def normalize_analysis(job: JobRecord, analysis: ModelAnalysis, model_id: str) -> AnalysisResult:
    provided = {(field.category, field.key): field for field in analysis.fields}
    normalized: list[AnalysisField] = []
    has_comments = bool(job.comments_path)
    for spec in VIDEO_FIELD_SPECS:
        field = provided.get((spec.category, spec.key))
        forced_unknown = (spec.requires_comments and not has_comments) or (
            spec.category == "audiovisual" and spec.key == "human"
        )
        if field is None:
            field = AnalysisField(
                key=spec.key,
                category=spec.category,
                title=spec.title,
                summary="模型未返回该字段，等待人工补充或重新解析。",
                evidence="缺少可核验输出。",
                time="未解析",
                confidence=0,
                status="unknown",
            )
        field.id = f"{job.id}-{spec.category}-{spec.key}"
        field.key = spec.key
        field.category = spec.category
        field.title = spec.title
        if forced_unknown:
            field.summary = (
                "等待人工填写。" if spec.key == "human" else "未提供真实评论或行为数据，等待补充后分析。"
            )
            field.evidence = "人工复核入口。" if spec.key == "human" else "需要平台评论、弹幕、留存或采集记录。"
            field.time = "人工证据" if spec.key == "human" else "数据待接入"
            field.confidence = 0
            field.status = "unknown"
        elif field.status not in {"ai", "unknown"}:
            field.status = "ai"
        normalized.append(field)

    normalized.extend(_platform_fields(job.id, job.platform, has_comments))
    if not analysis.video.series_title:
        analysis.video.series_title = job.series_title
    if job.episode_hint is not None:
        analysis.video.episode = job.episode_hint
    if job.video_type_hint in {"正片", "预告片", "周边"}:
        analysis.video.video_type = job.video_type_hint
    if analysis.video.genre.strip() in {"", "待确认", "AI 自动识别"} and job.genre not in {"", "待确认", "AI 自动识别"}:
        analysis.video.genre = job.genre
    analysis.video.platform = job.platform
    for index, character in enumerate(analysis.world.characters):
        character.id = character.id or f"{job.id}-character-{index + 1}"
    for index, relationship in enumerate(analysis.world.relationships):
        relationship.id = relationship.id or f"{job.id}-relationship-{index + 1}"
    for index, faction in enumerate(analysis.world.factions):
        faction.id = faction.id or f"{job.id}-faction-{index + 1}"
    for index, element in enumerate(analysis.world.elements):
        element.id = element.id or f"{job.id}-world-element-{index + 1}"
    return AnalysisResult(
        job_id=job.id,
        created_at=utc_now(),
        model_id=model_id,
        source_file=job.file_name,
        video=analysis.video,
        fields=normalized,
        storyboard=analysis.storyboard,
        emotion_curve=analysis.emotion_curve,
        world=analysis.world,
    )


def process_job(job_id: str) -> None:
    job = job_store.get(job_id)
    if not job:
        return
    try:
        job_store.update(job_id, status="processing", progress=18, stage="校验模型配置", error=None)
        view = config_store.view()
        if not view.configured:
            raise ValueError("尚未配置百炼 API Key，请先在页面完成模型配置")
        client = create_qwen_client(config_store.api_key, config_store.base_url, config_store.model_id)
        job_store.update(job_id, progress=28, stage="正在上传至百炼临时存储并准备分析")
        analysis = client.analyze_video(
            video_path=job.video_path,
            file_name=job.file_name,
            genre=job.genre,
            platform=job.platform,
            subtitle_path=job.subtitle_path,
            comments_path=job.comments_path,
            series_title=job.series_title,
            episode_hint=job.episode_hint,
            video_type_hint=job.video_type_hint,
        )
        job_store.update(job_id, progress=72, stage="专项解析人物、关系与世界观")
        try:
            analysis.world = client.analyze_world(
                video_path=job.video_path,
                file_name=job.file_name,
                series_title=analysis.video.series_title or job.series_title,
                episode_title=analysis.video.episode_title,
            )
        except Exception as world_exc:
            analysis.world = WorldAnalysis(
                background_completeness="解析失败",
                notice=f"人物与世界专项解析未完成，可单独重试：{_sanitized_error(world_exc)}",
            )
        job_store.update(job_id, progress=86, stage="校验并补齐元数据字段")
        result = normalize_analysis(job, analysis, getattr(client, "model_id", config_store.model_id))
        analysis_path = ANALYSIS_DIR / f"{job_id}.json"
        write_json(analysis_path, result.model_dump())
        job_store.update(
            job_id,
            status="completed",
            progress=100,
            stage="等待人工审核",
            analysis_path=str(analysis_path),
            error=None,
        )
    except Exception as exc:  # Model/network failures need to remain inspectable and retryable.
        job_store.update(job_id, status="failed", stage="解析失败", error=_sanitized_error(exc))


def process_world_job(job_id: str) -> None:
    job = job_store.get(job_id)
    if not job:
        return
    analysis_path = ANALYSIS_DIR / f"{job_id}.json"
    payload = read_json(analysis_path)
    if not payload:
        job_store.update(job_id, status="failed", stage="人物与世界补充解析失败", error="原分析结果不存在")
        return
    try:
        current = AnalysisResult.model_validate(payload)
        client = create_qwen_client(config_store.api_key, config_store.base_url, config_store.model_id)
        job_store.update(job_id, status="processing", progress=45, stage="正在补充解析人物与世界", error=None)
        current.world = client.analyze_world(
            video_path=job.video_path,
            file_name=job.file_name,
            series_title=current.video.series_title or job.series_title,
            episode_title=current.video.episode_title,
        )
        normalized = normalize_analysis(job, current, getattr(client, "model_id", config_store.model_id))
        write_json(analysis_path, normalized.model_dump())
        job_store.update(job_id, status="completed", progress=100, stage="人物与世界已补充，等待人工审核", error=None)
    except Exception as exc:
        job_store.update(job_id, status="failed", stage="人物与世界补充解析失败", error=_sanitized_error(exc))


async def _save_upload(upload: UploadFile, target: Path, limit: int) -> int:
    size = 0
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target.open("wb") as handle:
            while chunk := await upload.read(1024 * 1024):
                size += len(chunk)
                if size > limit:
                    raise HTTPException(status_code=413, detail=f"文件超过 {limit // 1024 // 1024}MB 限制")
                handle.write(chunk)
    except Exception:
        target.unlink(missing_ok=True)
        raise
    finally:
        await upload.close()
    return size


@app.get("/api/health")
def health() -> dict[str, object]:
    return {"ok": True, "service": "creative-library-qwen-mvp", "configured": config_store.view().configured}


@app.get("/api/config", response_model=ModelConfigView)
def get_config() -> ModelConfigView:
    return config_store.view()


@app.put("/api/config", response_model=ModelConfigView)
def update_config(config: ModelConfigInput) -> ModelConfigView:
    effective_key = (config.api_key or config_store.api_key).strip()
    _validate_app_credentials(config.base_url, effective_key)
    return config_store.update(config)


@app.post("/api/config/test", response_model=ConfigTestResult)
def test_config() -> ConfigTestResult:
    view = config_store.view()
    if not view.configured:
        raise HTTPException(status_code=400, detail="请先填写并保存 API Key")
    _validate_app_credentials(config_store.base_url, config_store.api_key)
    try:
        client = create_qwen_client(config_store.api_key, config_store.base_url, config_store.model_id)
        message = client.test_connection()
        return ConfigTestResult(ok=True, message=message, model_id=getattr(client, "model_id", config_store.model_id))
    except Exception as exc:
        raise HTTPException(status_code=502, detail=_sanitized_error(exc)) from exc


@app.post("/api/jobs", response_model=JobRecord, status_code=202)
async def create_job(
    background_tasks: BackgroundTasks,
    video: Annotated[UploadFile, File(...)],
    subtitle: Annotated[UploadFile | None, File()] = None,
    comments: Annotated[UploadFile | None, File()] = None,
    genre: Annotated[str, Form()] = "待确认",
    platform: Annotated[str, Form()] = "本地导入",
    series_id: Annotated[str, Form()] = "s1",
    series_title: Annotated[str, Form()] = "",
    episode: Annotated[int | None, Form(ge=1)] = None,
    video_type: Annotated[str, Form()] = "AI 自动识别",
) -> JobRecord:
    if not config_store.view().configured:
        raise HTTPException(status_code=400, detail="请先完成模型配置并测试连接")
    _validate_app_credentials(config_store.base_url, config_store.api_key)
    suffix = Path(video.filename or "").suffix.lower()
    if suffix not in VIDEO_EXTENSIONS:
        raise HTTPException(status_code=415, detail="不支持该视频格式")
    job_id = uuid.uuid4().hex[:16]
    video_name = safe_filename(video.filename or f"video{suffix}")
    video_path = UPLOAD_DIR / f"{job_id}-{video_name}"
    await _save_upload(video, video_path, MAX_VIDEO_BYTES)

    subtitle_path: Path | None = None
    comments_path: Path | None = None
    if subtitle and subtitle.filename:
        if Path(subtitle.filename).suffix.lower() not in AUX_EXTENSIONS:
            video_path.unlink(missing_ok=True)
            raise HTTPException(status_code=415, detail="字幕文件仅支持 SRT、VTT、TXT")
        subtitle_path = UPLOAD_DIR / f"{job_id}-subtitle-{safe_filename(subtitle.filename)}"
        await _save_upload(subtitle, subtitle_path, MAX_AUX_BYTES)
    if comments and comments.filename:
        if Path(comments.filename).suffix.lower() not in AUX_EXTENSIONS:
            video_path.unlink(missing_ok=True)
            if subtitle_path:
                subtitle_path.unlink(missing_ok=True)
            raise HTTPException(status_code=415, detail="评论文件仅支持 CSV、TXT")
        comments_path = UPLOAD_DIR / f"{job_id}-comments-{safe_filename(comments.filename)}"
        await _save_upload(comments, comments_path, MAX_AUX_BYTES)

    job = JobRecord(
        id=job_id,
        file_name=video_name,
        video_path=str(video_path),
        subtitle_path=str(subtitle_path) if subtitle_path else None,
        comments_path=str(comments_path) if comments_path else None,
        genre=genre.strip() or "待确认",
        platform=platform.strip() or "本地导入",
        series_id=series_id,
        series_title=series_title.strip(),
        episode_hint=episode,
        video_type_hint=video_type if video_type in {"正片", "预告片", "周边"} else "AI 自动识别",
        progress=10,
        stage="文件已保存，等待模型处理",
    )
    job_store.save(job)
    background_tasks.add_task(process_job, job.id)
    return job


@app.get("/api/jobs/{job_id}", response_model=JobRecord)
def get_job(job_id: str) -> JobRecord:
    job = job_store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="任务不存在")
    return job


@app.post("/api/jobs/{job_id}/retry", response_model=JobRecord, status_code=202)
def retry_job(job_id: str, background_tasks: BackgroundTasks) -> JobRecord:
    job = job_store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="任务不存在")
    if not Path(job.video_path).exists():
        raise HTTPException(status_code=410, detail="原视频已不存在，无法重试")
    _validate_app_credentials(config_store.base_url, config_store.api_key)
    job = job_store.update(job_id, status="queued", progress=10, stage="等待重试", error=None)
    background_tasks.add_task(process_job, job.id)
    return job


@app.post("/api/jobs/{job_id}/reanalyze-world", response_model=JobRecord, status_code=202)
def reanalyze_world(job_id: str, background_tasks: BackgroundTasks) -> JobRecord:
    job = job_store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="任务不存在")
    if not Path(job.video_path).exists():
        raise HTTPException(status_code=410, detail="原视频已不存在，无法补充解析")
    if not (ANALYSIS_DIR / f"{job_id}.json").exists():
        raise HTTPException(status_code=409, detail="基础解析尚未完成")
    _validate_app_credentials(config_store.base_url, config_store.api_key)
    job = job_store.update(job_id, status="queued", progress=20, stage="等待人物与世界补充解析", error=None)
    background_tasks.add_task(process_world_job, job.id)
    return job


@app.get("/api/analyses/{job_id}", response_model=AnalysisResult)
def get_analysis(job_id: str) -> AnalysisResult:
    path = ANALYSIS_DIR / f"{job_id}.json"
    payload = read_json(path)
    if not payload:
        raise HTTPException(status_code=404, detail="解析结果不存在")
    return AnalysisResult.model_validate(payload)


@app.put("/api/analyses/{job_id}", response_model=AnalysisResult)
def update_analysis(job_id: str, result: AnalysisResult) -> AnalysisResult:
    if result.job_id != job_id:
        raise HTTPException(status_code=400, detail="任务ID不一致")
    path = ANALYSIS_DIR / f"{job_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="解析结果不存在")
    write_json(path, result.model_dump())
    return result


@app.get("/api/analyses/{job_id}/download")
def download_analysis(job_id: str) -> FileResponse:
    path = ANALYSIS_DIR / f"{job_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="解析结果不存在")
    return FileResponse(path, media_type="application/json", filename=f"analysis-{job_id}.json")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(ROOT / "web" / "index.html")


app.mount("/assets", StaticFiles(directory=ROOT / "web"), name="assets")
