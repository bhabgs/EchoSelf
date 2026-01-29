"""
RAG 检索增强模块 - 使用向量数据库增强对话上下文
RAG Engine Module - Retrieval-Augmented Generation for Enhanced Context

功能：
1. 将聊天记录索引到向量数据库
2. 根据用户查询检索相关对话
3. 增强 LLM 的上下文理解
4. 支持增量更新和持久化

依赖：
- ChromaDB: 向量数据库
- sentence-transformers: 文本嵌入模型
"""

import os
import sys
import json
from pathlib import Path
from typing import Optional, Union, List, Dict, Tuple
from loguru import logger
import hashlib

# 配置 loguru
logger.remove()
logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> - <level>{message}</level>")


class RAGEngine:
    """
    RAG 检索增强引擎
    使用 ChromaDB + sentence-transformers 实现
    """

    def __init__(
        self,
        persist_directory: Optional[Union[str, Path]] = None,
        embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        collection_name: str = "echosself_conversations",
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ):
        """
        初始化 RAG 引擎

        Args:
            persist_directory: 向量数据库持久化目录
            embedding_model: 嵌入模型名称（支持中文的多语言模型）
            collection_name: ChromaDB 集合名称
            chunk_size: 文本分块大小
            chunk_overlap: 分块重叠大小
        """
        self.persist_directory = Path(persist_directory) if persist_directory else Path("data/chroma_db")
        self.embedding_model_name = embedding_model
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # 组件
        self.client = None
        self.collection = None
        self.embedding_model = None

        # 确保目录存在
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        logger.info(f"RAGEngine 初始化 - 持久化目录: {self.persist_directory}")

    def initialize(self) -> bool:
        """
        初始化 RAG 引擎（加载模型和数据库）

        Returns:
            bool: 是否成功
        """
        logger.info("正在初始化 RAG 引擎...")

        try:
            # 加载嵌入模型
            if not self._load_embedding_model():
                return False

            # 初始化 ChromaDB
            if not self._init_chromadb():
                return False

            logger.info("✓ RAG 引擎初始化成功")
            return True

        except Exception as e:
            logger.error(f"RAG 引擎初始化失败: {e}")
            return False

    def _load_embedding_model(self) -> bool:
        """
        加载文本嵌入模型
        """
        try:
            from sentence_transformers import SentenceTransformer

            logger.info(f"加载嵌入模型: {self.embedding_model_name}")
            self.embedding_model = SentenceTransformer(self.embedding_model_name)

            # 测试模型
            test_embedding = self.embedding_model.encode("测试文本")
            logger.info(f"嵌入维度: {len(test_embedding)}")

            return True

        except ImportError:
            logger.error("sentence-transformers 未安装，请运行: pip install sentence-transformers")
            return False
        except Exception as e:
            logger.error(f"加载嵌入模型失败: {e}")
            return False

    def _init_chromadb(self) -> bool:
        """
        初始化 ChromaDB
        """
        try:
            import chromadb
            from chromadb.config import Settings

            logger.info("初始化 ChromaDB...")

            # 创建客户端（持久化模式）
            self.client = chromadb.Client(Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory=str(self.persist_directory),
                anonymized_telemetry=False
            ))

            # 获取或创建集合
            # 使用自定义嵌入函数
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "EchoSelf conversation history"}
            )

            logger.info(f"集合 '{self.collection_name}' 已就绪，当前文档数: {self.collection.count()}")
            return True

        except ImportError:
            logger.error("chromadb 未安装，请运行: pip install chromadb")
            return False
        except Exception as e:
            logger.error(f"初始化 ChromaDB 失败: {e}")
            return False

    def add_documents(
        self,
        documents: List[str],
        metadatas: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None
    ) -> bool:
        """
        添加文档到向量数据库

        Args:
            documents: 文档列表
            metadatas: 元数据列表（可选）
            ids: 文档ID列表（可选，自动生成）

        Returns:
            bool: 是否成功
        """
        if not documents:
            logger.warning("没有文档需要添加")
            return False

        if self.collection is None or self.embedding_model is None:
            logger.error("RAG 引擎未初始化")
            return False

        try:
            logger.info(f"正在添加 {len(documents)} 个文档...")

            # 生成 ID（如果没有提供）
            if ids is None:
                ids = [self._generate_id(doc) for doc in documents]

            # 生成元数据（如果没有提供）
            if metadatas is None:
                metadatas = [{"source": "user_chat"} for _ in documents]

            # 生成嵌入
            embeddings = self.embedding_model.encode(documents).tolist()

            # 添加到集合
            self.collection.add(
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )

            # 持久化
            self.client.persist()

            logger.info(f"✓ 成功添加 {len(documents)} 个文档")
            return True

        except Exception as e:
            logger.error(f"添加文档失败: {e}")
            return False

    def add_chat_history(
        self,
        messages: List[Dict[str, str]],
        chunk_conversations: bool = True
    ) -> bool:
        """
        添加聊天记录到向量数据库

        Args:
            messages: 消息列表 [{"role": "user", "content": "..."}, ...]
            chunk_conversations: 是否将对话分块

        Returns:
            bool: 是否成功
        """
        if not messages:
            return False

        documents = []
        metadatas = []

        if chunk_conversations:
            # 按对话轮次分组
            current_chunk = []
            current_length = 0

            for msg in messages:
                content = msg.get("content", "")
                role = msg.get("role", "user")

                line = f"{role}: {content}"
                line_length = len(line)

                if current_length + line_length > self.chunk_size and current_chunk:
                    # 保存当前块
                    documents.append("\n".join(current_chunk))
                    metadatas.append({"type": "conversation_chunk"})
                    current_chunk = []
                    current_length = 0

                current_chunk.append(line)
                current_length += line_length

            # 保存最后一个块
            if current_chunk:
                documents.append("\n".join(current_chunk))
                metadatas.append({"type": "conversation_chunk"})
        else:
            # 每条消息作为一个文档
            for msg in messages:
                content = msg.get("content", "")
                role = msg.get("role", "user")
                documents.append(f"{role}: {content}")
                metadatas.append({"type": "single_message", "role": role})

        return self.add_documents(documents, metadatas)

    def query(
        self,
        query_text: str,
        top_k: int = 5,
        filter_metadata: Optional[Dict] = None
    ) -> List[Dict]:
        """
        查询相关文档

        Args:
            query_text: 查询文本
            top_k: 返回的最相关文档数
            filter_metadata: 元数据过滤条件

        Returns:
            List[Dict]: 相关文档列表
        """
        if self.collection is None or self.embedding_model is None:
            logger.error("RAG 引擎未初始化")
            return []

        try:
            # 生成查询嵌入
            query_embedding = self.embedding_model.encode(query_text).tolist()

            # 查询
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=filter_metadata
            )

            # 格式化结果
            formatted_results = []
            if results and results['documents']:
                for i, doc in enumerate(results['documents'][0]):
                    formatted_results.append({
                        "content": doc,
                        "metadata": results['metadatas'][0][i] if results['metadatas'] else {},
                        "distance": results['distances'][0][i] if results['distances'] else None,
                        "id": results['ids'][0][i] if results['ids'] else None
                    })

            logger.debug(f"查询 '{query_text[:30]}...' 返回 {len(formatted_results)} 个结果")
            return formatted_results

        except Exception as e:
            logger.error(f"查询失败: {e}")
            return []

    def get_relevant_context(
        self,
        query_text: str,
        top_k: int = 5,
        max_context_length: int = 2000
    ) -> str:
        """
        获取与查询相关的上下文（用于增强 LLM 提示）

        Args:
            query_text: 查询文本
            top_k: 检索文档数
            max_context_length: 最大上下文长度

        Returns:
            str: 相关上下文文本
        """
        results = self.query(query_text, top_k)

        if not results:
            return ""

        context_parts = []
        current_length = 0

        for result in results:
            content = result.get("content", "")
            if current_length + len(content) <= max_context_length:
                context_parts.append(content)
                current_length += len(content)
            else:
                # 截断以适应长度限制
                remaining = max_context_length - current_length
                if remaining > 50:
                    context_parts.append(content[:remaining] + "...")
                break

        return "\n---\n".join(context_parts)

    def _generate_id(self, content: str) -> str:
        """
        为内容生成唯一 ID

        Args:
            content: 文本内容

        Returns:
            str: MD5 哈希 ID
        """
        return hashlib.md5(content.encode('utf-8')).hexdigest()

    def clear(self):
        """
        清空所有文档
        """
        if self.client and self.collection:
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "EchoSelf conversation history"}
            )
            logger.info("向量数据库已清空")

    def get_stats(self) -> Dict:
        """
        获取数据库统计信息

        Returns:
            Dict: 统计信息
        """
        if self.collection is None:
            return {"status": "not_initialized"}

        return {
            "status": "ready",
            "collection_name": self.collection_name,
            "document_count": self.collection.count(),
            "persist_directory": str(self.persist_directory),
            "embedding_model": self.embedding_model_name
        }


class RAGEnhancedChat:
    """
    RAG 增强的对话类
    将 RAG 检索与 LLM 调用结合
    """

    def __init__(
        self,
        rag_engine: RAGEngine,
        llm_client,
        system_prompt: str = ""
    ):
        """
        初始化 RAG 增强对话

        Args:
            rag_engine: RAG 引擎实例
            llm_client: LLM 客户端实例
            system_prompt: 系统提示词
        """
        self.rag_engine = rag_engine
        self.llm_client = llm_client
        self.system_prompt = system_prompt

    def chat(
        self,
        user_input: str,
        chat_history: Optional[List[Dict]] = None,
        use_rag: bool = True,
        top_k: int = 3
    ) -> str:
        """
        进行 RAG 增强的对话

        Args:
            user_input: 用户输入
            chat_history: 对话历史
            use_rag: 是否使用 RAG
            top_k: 检索文档数

        Returns:
            str: 模型回复
        """
        # 构建上下文增强的系统提示词
        enhanced_prompt = self.system_prompt

        if use_rag and self.rag_engine:
            # 获取相关上下文
            relevant_context = self.rag_engine.get_relevant_context(
                user_input, top_k=top_k
            )

            if relevant_context:
                enhanced_prompt += f"\n\n## 相关历史对话（供参考）：\n{relevant_context}"

        # 构建消息
        messages = chat_history or []
        messages = messages + [{"role": "user", "content": user_input}]

        # 调用 LLM
        reply = self.llm_client.generate_reply(
            messages,
            system_prompt=enhanced_prompt
        )

        return reply


# ============================================================
# 测试代码
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("RAGEngine 测试")
    print("=" * 60)

    # 创建 RAG 引擎
    rag = RAGEngine(
        persist_directory="/tmp/test_chroma",
        embedding_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    # 初始化
    print("\n1. 初始化 RAG 引擎...")
    if rag.initialize():
        print("✓ 初始化成功")
    else:
        print("✗ 初始化失败")
        exit(1)

    # 添加测试文档
    print("\n2. 添加测试文档...")
    test_docs = [
        "我最喜欢的颜色是蓝色",
        "周末我喜欢去公园散步",
        "我的工作是软件工程师",
        "最近在学习人工智能",
        "我养了一只叫小白的猫"
    ]
    rag.add_documents(test_docs)
    print(f"✓ 添加了 {len(test_docs)} 个文档")

    # 查询测试
    print("\n3. 测试查询...")
    queries = [
        "你喜欢什么颜色？",
        "你养宠物吗？",
        "你是做什么工作的？"
    ]

    for query in queries:
        print(f"\n查询: {query}")
        results = rag.query(query, top_k=2)
        for i, r in enumerate(results):
            print(f"  结果{i+1}: {r['content']} (距离: {r['distance']:.4f})")

    # 获取统计信息
    print("\n4. 数据库统计...")
    stats = rag.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # 清理
    print("\n5. 清理测试数据...")
    rag.clear()
    print("✓ 数据已清空")

    import shutil
    shutil.rmtree("/tmp/test_chroma", ignore_errors=True)

    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)
