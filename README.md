# RAG from Scratch 🚀

从零实现 RAG (Retrieval-Augmented Generation) 系统，包含完整流程：向量嵌入 → 检索 → 重排序 → 生成。

## ✨ 特性

- 📚 **向量化**: 使用阿里云 text-embedding-v4 模型
- 🔍 **语义检索**: 基于余弦相似度的向量检索
- 🎯 **重排序**: Cross-Encoder 精确重排序提升精度
- 🤖 **生成**: 通义千问大模型生成回答
- 💾 **持久化**: 向量数据库本地缓存

## 📦 安装

```bash
# 安装依赖
pip install -r requirements.txt
```

### 下载 Reranker 模型

**方式一：使用 ModelScope（推荐，国内速度快）**

```bash
# 安装 modelscope
pip install modelscope

# 下载模型到项目目录
modelscope download --model cross-encoder/ms-marco-MiniLM-L6-v2 --local_dir ./ms-marco-MiniLM-L6-v2
```

**方式二：手动下载**

从 [HuggingFace](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2) 下载模型文件到项目目录 `ms-marco-MiniLM-L6-v2/`

**方式三：自动下载（首次运行时）**

`sentence-transformers` 会在首次运行时自动下载模型到缓存目录，然后修改 `rag.py` 中的模型路径。

## 🔑 配置

创建 `.env` 文件：

```env
apikey=你的阿里云API密钥
```

## 🚀 使用

### 快速开始

```python
from rag import RAGPipeline

# 初始化
rag = RAGPipeline()

# 加载索引
rag.load_index()

# 启用重排序
rag.enable_reranker()

# 问答
rag.chat("猫能吃什么？")
```

### 命令行运行

```bash
python rag.py
```

## 📁 项目结构

```
.
├── rag.py              # 主程序
├── cat-facts.txt       # 知识库
├── vector_db.pkl       # 向量数据库缓存
├── ms-marco-MiniLM-L6-v2/  # Reranker 模型
└── requirements.txt    # 依赖
```

## 🔄 RAG 流程

```
用户问题
    ↓
[Embedding] → 向量化
    ↓
[Retriever] → 语义检索 (Top 10)
    ↓
[Reranker] → 精确重排序 (Top 3)
    ↓
[Generator] → LLM 生成回答
```

## 📊 模型说明

| 组件 | 模型 | 大小 |
|------|------|------|
| Embedding | text-embedding-v4 | API |
| Reranker | ms-marco-MiniLM-L6-v2 | 261MB |
| LLM | qwen-max | API |

## 📝 License

MIT
