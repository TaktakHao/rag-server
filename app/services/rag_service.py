from pathlib import Path
from typing import List
import logging
from operator import itemgetter

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
    def __init__(self):
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
        
        # 设置提示模板
        self.prompt = PromptTemplate.from_template(
            """你是一个严谨的RAG助手。
            请优先根据以下提供的上下文信息来回答问题，
            如果上下文信息不足以回答问题，
            就再根据问题自行思考回答。
            如果问题使用了上下文的信息，请在回答后输出使用了哪些上下文。
            
            问题：{question}
            上下文信息：{context}
            --------------------------------
            回答：
            """
        )
        
        # 构建处理链
        self.chain = (
            {"question": RunnablePassthrough()} 
            | RunnablePassthrough.assign(
                context=itemgetter("question") | self.retriever
            ) 
            | self.prompt 
            | self.llm 
            | StrOutputParser()
        )
    
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
    
    async def query(self, question: str) -> str:
        """查询RAG系统"""
        try:
            result = self.chain.invoke(question)
            return result
        except Exception as e:
            logger.error(f"查询RAG系统时出错: {str(e)}")
            return f"处理您的问题时出错: {str(e)}"
    
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
            knowledge_base_dir = Path(settings.DATA_DIR) / "knowledge_base"
            if knowledge_base_dir.exists():
                for file_path in knowledge_base_dir.glob("*"):
                    try:
                        file_path.unlink()
                        logger.info(f"已删除知识库文件: {file_path}")
                    except Exception as e:
                        logger.error(f"删除文件 {file_path} 时出错: {str(e)}")
            
            return True
        except Exception as e:
            logger.error(f"清空数据库时出错: {str(e)}")
            return False 