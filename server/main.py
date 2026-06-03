import os

# Allow duplicate OpenMP runtime to avoid libiomp5md.dll initialization error
# (If you prefer a stricter fix, remove duplicate OpenMP runtimes from dependencies.)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import uvicorn
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from routers import api_router
from server_config.database import get_db_manager
from server_config.settings import UVICORN_CONFIG
from services.agent_service import agent_manager

# 初始化 FastAPI 应用
app = FastAPI(title="知识图谱问答系统", description="基于知识图谱的智能问答系统后端API")

# 添加路由
app.include_router(api_router)


@app.get("/", include_in_schema=False)
def root():
    """根路径，重定向到交互文档"""
    return RedirectResponse(url="/docs")

# 获取数据库连接
db_manager = get_db_manager()
driver = db_manager.driver


@app.on_event("shutdown")
def shutdown_event():
    """应用关闭时清理资源"""
    # 关闭所有Agent资源
    agent_manager.close_all()
    
    # 关闭Neo4j连接
    if driver:
        driver.close()
        print("已关闭Neo4j连接")


# 启动服务器
if __name__ == "__main__":
    uvicorn.run("main:app", **UVICORN_CONFIG)
