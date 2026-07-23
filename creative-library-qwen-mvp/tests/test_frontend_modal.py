from pathlib import Path


HTML_PATH = Path(__file__).parents[1] / "web" / "index.html"


def test_modal_only_closes_when_the_backdrop_itself_is_clicked():
    html = HTML_PATH.read_text(encoding="utf-8")

    assert 'class="modal-backdrop" data-action="backdrop-close"' not in html
    assert "e.target.classList.contains('modal-backdrop')" in html


def test_every_modal_shell_has_an_explicit_close_button():
    html = HTML_PATH.read_text(encoding="utf-8")

    assert 'data-action="close-modal" title="关闭"' in html
    assert "foot||`<button class=\"btn\" data-action=\"close-modal\">关闭</button>`" in html


def test_video_overview_can_be_edited_and_reassigned_to_a_series():
    html = HTML_PATH.read_text(encoding="utf-8")

    assert 'data-action="edit-video-overview"' in html
    assert "function openVideoOverviewEdit()" in html
    assert "action==='save-video-overview'" in html
    assert 'value="__new__"' in html


def test_soft_light_theme_is_the_final_theme():
    html = HTML_PATH.read_text(encoding="utf-8")

    assert "/* Soft daylight workbench theme */" in html
    assert "html { color-scheme: light; }" in html
    assert "--bg: #e9ece8" in html


def test_left_content_tree_displays_the_full_episode_label():
    html = HTML_PATH.read_text(encoding="utf-8")

    assert 'class="video-thumb">${episodeLabel(item)}</span>' in html


def test_uploaded_series_analysis_is_derived_only_from_uploaded_videos():
    html = HTML_PATH.read_text(encoding="utf-8")

    assert "function uploadedSeriesVideos(series)" in html
    assert "function buildUploadedSeriesFields(series,videos)" in html
    assert "buildUploadedSeriesFields(series,uploaded)" in html
    assert "未上传部分不推断" in html


def test_uploaded_series_emotion_curve_uses_real_video_curves():
    html = HTML_PATH.read_text(encoding="utf-8")

    assert "function uploadedSeriesEmotionCurve(series)" in html
    assert "uploadedSeriesEmotionCurve(getSeries())" in html


def test_content_object_panel_has_independent_nested_scroll_regions():
    html = HTML_PATH.read_text(encoding="utf-8")

    assert 'class="panel sticky content-object-panel"' in html
    assert ".content-object-body { overflow-y: auto;" in html
    assert ".video-children {" in html
    assert "resize: vertical" in html


def test_selected_series_episode_list_supports_persisted_vertical_height():
    html = HTML_PATH.read_text(encoding="utf-8")

    assert "seriesHeights" in html
    assert "document.addEventListener('pointerup'" in html
    assert "resize: vertical" in html


def test_series_manager_can_hide_restore_and_create_series():
    html = HTML_PATH.read_text(encoding="utf-8")

    assert 'data-action="open-series-manager"' in html
    assert "function openSeriesManager()" in html
    assert "action==='toggle-series-visibility'" in html
    assert "action==='create-empty-series'" in html
    assert "hiddenSeriesIds" in html


def test_genre_defaults_to_ai_inference_and_has_an_expanded_editable_list():
    html = HTML_PATH.read_text(encoding="utf-8")

    assert "const GENRE_OPTIONS=" in html
    assert html.count("校园青春") >= 1
    assert html.count("无限流") >= 1
    assert html.count("年代传奇") >= 1
    assert 'id="upload-genre" list="upload-genre-options" value="AI 自动识别"' in html
    assert 'id="video-genre" list="video-genre-options"' in html


def test_upload_allows_an_optional_human_episode_hint() -> None:
    html = HTML_PATH.read_text(encoding="utf-8")

    assert 'id="upload-episode"' in html
    assert 'placeholder="留空由 AI 识别"' in html
    assert "form.append('episode',episodeHint)" in html


def test_video_type_can_be_ai_inferred_and_manually_adjusted() -> None:
    html = HTML_PATH.read_text(encoding="utf-8")

    assert 'id="upload-video-type"' in html
    assert '<option value="AI 自动识别">AI 自动识别</option>' in html
    assert '<option value="正片">正片</option>' in html
    assert '<option value="预告片">预告片</option>' in html
    assert '<option value="周边">周边</option>' in html
    assert "form.append('video_type'" in html
    assert 'id="video-type"' in html


def test_world_analysis_can_be_retried_and_entities_can_be_added_or_removed():
    html = HTML_PATH.read_text(encoding="utf-8")

    assert "function applyWorldAnalysis(" in html
    assert "action==='reanalyze-world'" in html
    assert "/reanalyze-world" in html
    assert "action==='add-world-entity'" in html
    assert "action==='delete-world-entity'" in html
    assert "背景信息不全" in html


def test_left_panel_renders_only_the_selected_series_and_six_section_anchors():
    html = HTML_PATH.read_text(encoding="utf-8")

    assert "function renderSectionDirectory()" in html
    assert 'class="section-directory"' in html
    assert "metadataSectionDefinitions().map" in html
    assert "renderSeriesGroup(series)" in html
    assert "shownSeries.map(renderSeriesGroup)" not in html


def test_episode_list_is_vertical_only_and_episode_rows_are_compact():
    html = HTML_PATH.read_text(encoding="utf-8")

    assert "overflow-y: auto; overflow-x: hidden" in html
    assert ".video-node .item-title" in html
    assert "font-size: 11px" in html


def test_emotion_curve_is_described_as_content_based_qualitative_prediction():
    html = HTML_PATH.read_text(encoding="utf-8")

    assert "依据剧情内容的 AI 定性预测" in html
    assert "不读取评论来绘制" in html
    assert "相对高" in html
    assert "相对低" in html
