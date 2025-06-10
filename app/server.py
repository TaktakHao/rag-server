from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import RedirectResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import logging
from pathlib import Path
import shutil
from typing import List
from pydantic import BaseModel
import os

from .config import settings
from .services.rag_service import RAGService

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.PROJECT_NAME,
    version="0.1.0",
    description="RAG demo API"
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 添加静态文件服务
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# 初始化RAG服务
rag_service = RAGService()

class QueryRequest(BaseModel):
    question: str

@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/static/index.html")

@app.post(f"{settings.API_V1_STR}/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    """上传文件到知识库"""
    try:
        saved_files = []
        for file in files:
            # 将文件保存到知识库目录
            file_path = settings.KNOWLEDGE_BASE_DIR / file.filename
            with file_path.open("wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            saved_files.append(file_path)
        
        # 将文件添加到向量数据库
        success = await rag_service.add_documents(saved_files)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to process files")
        
        return {"message": f"Successfully uploaded and processed {len(saved_files)} files"}
    except Exception as e:
        logger.error(f"Error uploading files: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post(f"{settings.API_V1_STR}/query")
async def query(request: QueryRequest):
    """查询RAG系统"""
    try:
        return StreamingResponse(
            rag_service.query(request.question),
            media_type="text/event-stream"
        )
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post(f"{settings.API_V1_STR}/clear")
async def clear_database():
    """清空向量数据库"""
    try:
        success = await rag_service.clear_database()
        if not success:
            raise HTTPException(status_code=500, detail="Failed to clear database")
        return {"message": "Successfully cleared database"}
    except Exception as e:
        logger.error(f"Error clearing database: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get(f"{settings.API_V1_STR}/files")
async def list_files():
    """获取知识库中的文件列表"""
    try:
        # 确保知识库目录存在
        settings.KNOWLEDGE_BASE_DIR.mkdir(parents=True, exist_ok=True)
        
        # 获取所有文件
        files = []
        for file in settings.KNOWLEDGE_BASE_DIR.glob("*"):
            if file.is_file():
                # 获取文件大小和修改时间
                stat = file.stat()
                files.append({
                    "name": file.name,
                    "size": stat.st_size,
                    "modified": stat.st_mtime,
                    "type": file.suffix.lower()
                })
        
        # 按修改时间排序，最新的在前面
        files.sort(key=lambda x: x["modified"], reverse=True)
        
        return {"files": files}
    except Exception as e:
        logger.error(f"Error listing files: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete(f"{settings.API_V1_STR}/files/{{filename}}")
async def delete_file(filename: str):
    """删除知识库中的文件"""
    try:
        file_path = settings.KNOWLEDGE_BASE_DIR / filename
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
        
        # 删除文件
        os.remove(file_path)
        
        # 重新处理知识库
        success = await rag_service.clear_database()
        if not success:
            raise HTTPException(status_code=500, detail="Failed to clear database")
            
        # 重新添加所有文件
        files = list(settings.KNOWLEDGE_BASE_DIR.glob("*"))
        if files:
            success = await rag_service.add_documents(files)
            if not success:
                raise HTTPException(status_code=500, detail="Failed to process files")
        
        return {"message": f"Successfully deleted {filename}"}
    except Exception as e:
        logger.error(f"Error deleting file: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
