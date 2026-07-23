# 创意库 Qwen MVP

这是一个不使用数据库的本地应用：上传短剧视频后，通过阿里云百炼 Qwen3.5-Omni 生成创意库元数据，并保存为本地 JSON。

视频会先上传至百炼的临时存储，再以 `oss://` 地址交给模型分析，避免 Base64 请求体的 10MB 限制。临时文件由百炼保存 48 小时；本机原始视频仍保存在 `runtime-data/uploads`。

## 启动

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

打开 `http://127.0.0.1:8000`，进入“模型配置”填写百炼 API Key、Base URL 和模型 ID，测试连接后即可上传视频。

默认模型：`qwen3.5-omni-plus-2026-03-15`。默认 Base URL：
`https://dashscope.aliyuncs.com/compatible-mode/v1`

必须使用百炼控制台创建的**按量付费 API Key**（`sk-` 开头）。Token Plan / Coding Plan 的 `sk-sp-` Key 仅面向指定的交互式编程工具，不能用于本应用后端，也无法获取视频临时上传凭证。
## 数据与密钥

- API Key只保存在服务进程内存，重启后需重新输入；也可通过环境变量 `DASHSCOPE_API_KEY` 提供。
- 非密钥配置写入 `runtime-data/config.json`。
- 上传文件、任务状态和解析结果分别保存在 `runtime-data/uploads`、`runtime-data/jobs`、`runtime-data/analyses`。
- 这些运行数据默认被 Git 忽略。

## 无网络自测

```bash
QWEN_MOCK_MODE=1 uvicorn app.main:app --port 8000
```

Mock 模式会生成完整的54项元数据、分镜和情绪曲线，不调用外部模型。
