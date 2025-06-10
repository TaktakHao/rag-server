# Enterprise RAG API

一个基于 FastAPI 和 LangChain 的企业级 RAG（检索增强生成）API，支持多种文件格式的知识库管理。

## 功能特点

- 支持多种文件格式（PDF、TXT、DOC、DOCX）
- 文件上传和管理界面
- 实时聊天界面
- 知识库管理功能
- 支持文件删除和知识库清空

## 技术栈

- FastAPI: Web 框架
- LangChain: RAG 实现
- ChromaDB: 向量数据库
- DeepSeek: 大语言模型
- HuggingFace: 文本嵌入模型

## 安装

1. 确保已安装 Python 3.10 或更高版本
2. 安装 Poetry（如果尚未安装）：
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```
3. 克隆项目并安装依赖：
   ```bash
   git clone <repository-url>
   cd my-lang-server
   poetry install
   ```

## 依赖项

主要依赖包括：

- fastapi: Web 框架
- uvicorn: ASGI 服务器
- python-multipart: 文件上传支持
- pydantic-settings: 配置管理
- langchain: RAG 框架
- langchain-core: LangChain 核心功能
- langchain-community: LangChain 社区组件
- langchain-deepseek: DeepSeek 模型集成
- langchain-chroma: ChromaDB 集成
- langchain-text-splitters: 文本分割
- langchain-huggingface: HuggingFace 模型集成
- chromadb: 向量数据库
- python-docx: DOCX 文件处理
- docx2txt: DOCX 文本提取
- pdfminer.six: PDF 文件处理
- unstructured: 非结构化文档处理

## 配置

1. 创建 `.env` 文件并设置必要的环境变量：
   ```
   DEEPSEEK_API_KEY=your_api_key
   EMBEDDING_MODEL_PATH=your_model_path
   ```

2. 确保知识库目录存在：
   ```bash
   mkdir -p data/knowledge_base
   ```

3. 下载并配置 Embedding 模型：

   本项目使用 HuggingFace 的 BAAI/bge-small-zh 模型作为默认的文本嵌入模型。这是一个轻量级的中文文本嵌入模型，适合大多数应用场景。

   a. 使用 HuggingFace CLI 下载模型：
   ```bash
   # 安装 huggingface_hub
   poetry add huggingface_hub

   # 下载模型
   poetry run python -c "from huggingface_hub import snapshot_download; snapshot_download('BAAI/bge-small-zh', local_dir='models/bge-small-zh')"
   ```

   b. 在 `.env` 文件中设置模型路径：
   ```
   EMBEDDING_MODEL_PATH=models/bge-small-zh
   ```

   注意：
   - 模型文件大小约为 500MB
   - 确保有足够的磁盘空间
   - 下载过程可能需要一些时间，取决于网络速度
   - 如果下载速度较慢，可以考虑使用代理或镜像

   可选：使用其他嵌入模型
   - 如果需要更好的性能，可以考虑使用更大的模型如 `BAAI/bge-large-zh`
   - 如果需要更快的速度，可以考虑使用更小的模型如 `BAAI/bge-tiny-zh`
   - 更改模型时，只需修改下载命令中的模型名称和 `.env` 文件中的路径即可

## 运行

使用 Poetry 运行服务器：

```bash
poetry run uvicorn app.server:app --reload
```

服务器将在 http://localhost:8000 启动。

## API 端点

- `GET /`: 重定向到聊天界面
- `POST /api/v1/upload`: 上传文件
- `POST /api/v1/query`: 查询 RAG 系统
- `POST /api/v1/clear`: 清空知识库
- `GET /api/v1/files`: 获取文件列表
- `DELETE /api/v1/files/{filename}`: 删除指定文件

## 使用说明

1. 访问 http://localhost:8000 打开聊天界面
2. 在右侧面板上传文件到知识库
3. 在左侧聊天界面提问
4. 使用文件管理功能管理知识库

## 开发

- 使用 Poetry 管理依赖
- 遵循 PEP 8 编码规范
- 使用 FastAPI 的自动文档功能（访问 /docs）

## 许可证

MIT
