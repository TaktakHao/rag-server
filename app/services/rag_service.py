from pathlib import Path
from typing import List, Dict, Any
import logging
from operator import itemgetter
from collections import deque

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import (
    TextLoader,
    PDFMinerLoader,
    Docx2txtLoader,
    UnstructuredWordDocumentLoader
)
from langchain_deepseek import ChatDeepSeek
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

from ..config import settings

logger = logging.getLogger(__name__)

class RAGService:
    def __init__(self, max_history: int = 10):
        # 初始化大语言模型
        self.llm = ChatDeepSeek(
            model="deepseek-reasoner",
            api_key=settings.DEEPSEEK_API_KEY
        )
        # 初始化文本嵌入模型
        self.embeddings = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL_PATH
        )
        # 初始化向量数据库
        self.db = Chroma(
            embedding_function=self.embeddings,
            persist_directory=str(settings.VECTOR_DB_DIR)
        )
        # 初始化文本分割器
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            separators=["\n\n", "\n", " ", ""]
        )
        # 初始化检索器
        self.retriever = self.db.as_retriever(
            search_kwargs={"k": settings.MAX_RETRIEVAL_DOCS}
        )
        
        # 初始化对话历史记录
        self.conversation_history = deque(maxlen=max_history)
        
        # 设置提示模板
        self.prompt = PromptTemplate.from_template(
            """你是一个风趣幽默的RAG助手，可以间接的回答问题，
            请优先根据以下提供的上下文信息来回答问题，
            如果上下文信息不足以回答问题，就抛弃上下文信息，
            根据问题自行思考回答，回答要幽默风趣。
            
            历史对话：
            {chat_history}
            
            当前问题：{question}
            上下文信息：{context}
            --------------------------------
            回答：
            """
        )
        
        # 构建处理链
        self.chain = (
            {
                "question": RunnablePassthrough(),
                "chat_history": lambda _: self._format_chat_history()
            }
            | RunnablePassthrough.assign(
                context=itemgetter("question") | self.retriever
            ) 
            | self.prompt 
            | self.llm 
            | StrOutputParser()
        )
    
    def _format_chat_history(self) -> str:
        """格式化对话历史记录"""
        if not self.conversation_history:
            return "无历史对话"
        
        formatted_history = []
        for q, a in self.conversation_history:
            formatted_history.append(f"问：{q}\n答：{a}\n")
        return "\n".join(formatted_history)
    
    def _get_loader(self, file_path: Path):
        """根据文件类型选择合适的加载器"""
        suffix = file_path.suffix.lower()
        if suffix == '.txt':
            return TextLoader(str(file_path))
        elif suffix == '.pdf':
            return PDFMinerLoader(str(file_path))
        elif suffix == '.docx':
            return Docx2txtLoader(str(file_path))
        elif suffix == '.doc':
            return UnstructuredWordDocumentLoader(str(file_path))
        else:
            raise ValueError(f"Unsupported file type: {suffix}")
    
    async def add_documents(self, file_paths: List[Path]) -> bool:
        """向向量数据库添加文档"""
        try:
            for file_path in file_paths:
                if not file_path.exists():
                    logger.error(f"文件未找到: {file_path}")
                    continue
                
                try:
                    loader = self._get_loader(file_path)
                    documents = loader.load()
                    chunks = self.text_splitter.split_documents(documents)
                    self.db.add_documents(documents=chunks)
                    logger.info(f"成功将文件 {file_path} 添加到向量数据库")
                except Exception as e:
                    logger.error(f"处理文件 {file_path} 时出错: {str(e)}")
                    continue
            
            return True
        except Exception as e:
            logger.error(f"添加文档时出错: {str(e)}")
            return False
    
    async def query(self, question: str):
        """查询RAG系统"""
        try:
            # 获取上下文
            context = self.retriever.get_relevant_documents(question)
            # 格式化对话历史
            chat_history = self._format_chat_history()
            
            # 构建提示
            prompt = self.prompt.format(
                question=question,
                context=context,
                chat_history=chat_history
            )
            
            # 使用流式输出
            async for chunk in self.llm.astream(prompt):
                if chunk.content:
                    yield chunk.content
                    
            # 将当前问答添加到历史记录
            self.conversation_history.append((question, "".join(chunk.content)))
        except Exception as e:
            logger.error(f"查询RAG系统时出错: {str(e)}")
            yield f"处理您的问题时出错: {str(e)}"
    
    def clear_history(self) -> None:
        """清空对话历史记录"""
        self.conversation_history.clear()
    
    async def clear_database(self) -> bool:
        """清空向量数据库和知识库文件"""
        try:
            # 清空向量数据库
            self.db.delete_collection()
            self.db = Chroma(
                embedding_function=self.embeddings,
                persist_directory=str(settings.VECTOR_DB_DIR)
            )
            
            # 删除知识库文件
            knowledge_base_dir = Path(settings.KNOWLEDGE_BASE_DIR)
            if knowledge_base_dir.exists():
                for file_path in knowledge_base_dir.glob("*"):
                    try:
                        file_path.unlink()
                        logger.info(f"已删除知识库文件: {file_path}")
                    except Exception as e:
                        logger.error(f"删除文件 {file_path} 时出错: {str(e)}")
            
            # 删除向量数据库目录下的所有文件和文件夹
            vector_db_dir = Path(settings.VECTOR_DB_DIR)
            if vector_db_dir.exists():
                for file_path in vector_db_dir.glob("*"):
                    try:
                        if file_path.is_file():
                            file_path.unlink()
                            logger.info(f"已删除向量数据库文件: {file_path}")
                    except Exception as e:
                        logger.error(f"删除向量数据库文件 {file_path} 时出错: {str(e)}")
            
            # 清空对话历史记录
            self.clear_history()
            
            return True
        except Exception as e:
            logger.error(f"清空数据库时出错: {str(e)}")
            return False 