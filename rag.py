"""
RAG (Retrieval-Augmented Generation) 从零实现
完整流程：知识库 → 向量化 → 检索 → 重排序 → 生成
"""

import os
import pickle
from typing import List, Tuple

import dotenv
import numpy as np
from openai import OpenAI
from sentence_transformers import CrossEncoder


# ================================
# 配置
# ================================
class Config:
    # API 配置
    API_KEY = os.getenv("apikey")
    BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    # 模型配置
    EMBEDDING_MODEL = "text-embedding-v4"  # 阿里云嵌入模型 (1024维)
    CHAT_MODEL = "qwen-max"                # 通义千问大模型
    RERANKER_MODEL = r"D:\sikm\Desktop\PythonProject\AI_Learning\RAG_Learn\RAG_from_Scratch\ms-marco-MiniLM-L6-v2"  # 本地 Cross-Encoder 模型

    # 数据配置
    KNOWLEDGE_FILE = "cat-facts.txt"
    VECTOR_DB_FILE = "vector_db.pkl"

    # 检索配置
    RETRIEVER_TOP_K = 10   # 初步检索数量
    RERANKER_TOP_K = 3     # 重排序后返回数量


# ================================
# 1. 文本嵌入 (Embedding)
# ================================
class Embedder:
    """将文本转换为向量"""

    def __init__(self, client: OpenAI, model: str = Config.EMBEDDING_MODEL):
        self.client = client
        self.model = model

    def embed(self, text: str) -> List[float]:
        """将单条文本转换为向量"""
        response = self.client.embeddings.create(
            model=self.model,
            input=text
        )
        return response.data[0].embedding

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """批量转换文本为向量"""
        return [self.embed(text) for text in texts]


# ================================
# 2. 向量数据库
# ================================
class VectorDB:
    """简单的内存向量数据库"""

    def __init__(self, embedder: Embedder):
        self.embedder = embedder
        self.data: List[Tuple[str, List[float]]] = []  # [(文本, 向量), ...]

    def add(self, text: str) -> None:
        """添加文本及其向量到数据库"""
        vector = self.embedder.embed(text)
        self.data.append((text, vector))

    def add_batch(self, texts: List[str]) -> None:
        """批量添加文本"""
        for text in texts:
            self.add(text)

    def save(self, path: str) -> None:
        """保存到本地文件"""
        with open(path, 'wb') as f:
            pickle.dump(self.data, f)
        print(f"✅ 向量数据库已保存: {path} ({len(self.data)} 条)")

    def load(self, path: str) -> None:
        """从本地文件加载"""
        with open(path, 'rb') as f:
            self.data = pickle.load(f)
        print(f"✅ 向量数据库已加载: {path} ({len(self.data)} 条)")

    @staticmethod
    def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        """余弦相似度: cos(θ) = (A·B) / (||A|| × ||B||)"""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5
        return 0.0 if norm1 == 0 or norm2 == 0 else dot_product / (norm1 * norm2)


# ================================
# 3. 检索器 (Retriever)
# ================================
class Retriever:
    """基于向量相似度的检索器"""

    def __init__(self, vector_db: VectorDB):
        self.db = vector_db

    def retrieve(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        检索与查询最相关的文本
        :return: [(文本, 相似度), ...] 按相似度降序排列
        """
        # 将查询转换为向量
        query_vector = self.db.embedder.embed(query)

        # 计算与所有文本的相似度
        results = []
        for text, vector in self.db.data:
            sim = VectorDB.cosine_similarity(query_vector, vector)
            results.append((text, sim))

        # 按相似度降序排序
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]


# ================================
# 4. 重排序器 (Reranker)
# ================================
class Reranker:
    """基于 Cross-Encoder 的精确重排序"""

    def __init__(self, model_path: str = Config.RERANKER_MODEL):
        print(f"⏳ 加载 Reranker 模型...")
        self.model = CrossEncoder(model_path)
        print(f"✅ Reranker 模型加载完成")

    def rerank(self, query: str, candidates: List[Tuple[str, float]], top_k: int = 3) -> List[Tuple[str, float]]:
        """
        对候选结果进行重排序
        :param candidates: [(文本, 初始相似度), ...]
        :return: [(文本, rerank分数), ...] 按分数降序排列
        """
        if not candidates:
            return []

        # 构建 [(query, doc1), (query, doc2), ...]
        pairs = [[query, text] for text, _ in candidates]

        # 预测相关性分数
        scores = self.model.predict(pairs)

        # 组合结果并排序
        results = [(candidates[i][0], float(scores[i])) for i in range(len(candidates))]
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]


# ================================
# 5. 生成器 (Generator)
# ================================
class Generator:
    """基于 LLM 的回答生成器"""

    def __init__(self, client: OpenAI, model: str = Config.CHAT_MODEL):
        self.client = client
        self.model = model

    def generate(self, query: str, context: List[str], stream: bool = True) -> str:
        """
        基于检索到的上下文生成回答
        :param context: 检索到的相关文本列表
        """
        # 构建提示词
        context_text = '\n'.join([f"- {txt}" for txt in context])
        prompt = f"""You are a helpful chatbot.
Use only the following pieces of context to answer the question. Don't make up any new information:
{context_text}"""

        # 调用 LLM
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": query}
            ],
            stream=stream
        )

        if stream:
            answer = ""
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    print(content, end='', flush=True)
                    answer += content
            return answer
        else:
            return response.choices[0].message.content


# ================================
# 6. 完整 RAG 管道
# ================================
class RAGPipeline:
    """RAG 端到端管道"""

    def __init__(self):
        # 初始化客户端
        dotenv.load_dotenv()
        self.client = OpenAI(api_key=Config.API_KEY, base_url=Config.BASE_URL)

        # 初始化组件
        self.embedder = Embedder(self.client)
        self.vector_db = VectorDB(self.embedder)
        self.retriever = Retriever(self.vector_db)
        self.reranker = None  # 按需加载
        self.generator = Generator(self.client)

    def build_index(self, knowledge_file: str = Config.KNOWLEDGE_FILE) -> None:
        """构建向量索引"""
        print(f"📖 加载知识库: {knowledge_file}")
        with open(knowledge_file, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]

        print(f"📊 生成向量 ({len(texts)} 条)...")
        self.vector_db.add_batch(texts)
        self.vector_db.save(Config.VECTOR_DB_FILE)

    def load_index(self, vector_file: str = Config.VECTOR_DB_FILE) -> None:
        """加载已有索引"""
        self.vector_db.load(vector_file)

    def enable_reranker(self) -> None:
        """启用重排序"""
        if self.reranker is None:
            self.reranker = Reranker()

    def chat(self, query: str, use_reranker: bool = True) -> str:
        """
        端到端问答
        :param use_reranker: 是否使用重排序
        """
        print(f"\n{'='*60}")
        print(f"📝 问题: {query}")
        print(f"{'='*60}")

        # 步骤1: 检索
        candidates = self.retriever.retrieve(query, top_k=Config.RETRIEVER_TOP_K)
        print(f"\n🔍 检索到 {len(candidates)} 个候选结果")

        # 步骤2: 重排序 (可选)
        if use_reranker and self.reranker:
            results = self.reranker.rerank(query, candidates, top_k=Config.RERANKER_TOP_K)
            print(f"🎯 重排序后取 Top {len(results)}")
        else:
            results = candidates[:Config.RERANKER_TOP_K]

        # 步骤3: 生成
        context = [text for text, _ in results]
        print(f"\n🤖 回答:")
        self.generator.generate(query, context)
        print(f"\n{'='*60}\n")


# ================================
# 主程序
# ================================
def main():
    """演示完整 RAG 流程"""
    rag = RAGPipeline()

    # 加载或构建索引
    try:
        rag.load_index()
    except FileNotFoundError:
        print("⚠️ 索引文件不存在，正在构建...")
        rag.build_index()

    # 启用重排序
    rag.enable_reranker()

    # 交互式问答
    print("\n" + "="*60)
    print("🐱 猫咪知识问答 (输入 'quit' 退出)")
    print("="*60)

    while True:
        query = input("\n请输入问题: ").strip()
        if query.lower() in ('quit', 'exit', 'q'):
            print("👋 再见!")
            break
        if query:
            rag.chat(query)


if __name__ == "__main__":
    main()
