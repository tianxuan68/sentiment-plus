"""
启动 Jeecg FastAPI 后端服务

用法（在 jeecg-fastapi 目录）:
    python run.py
"""
# 1.导包
import os
from pathlib import Path

import uvicorn
from dotenv import load_dotenv


# 2.配置类
class Config:
    def __init__(self):
        self.root_path = str(Path(__file__).resolve().parent).replace('\\', '/') + '/'
        self.env_file = self.root_path + '.env'
        self.env_local_file = self.root_path + '.env.local'
        self.host = '0.0.0.0'
        self.port = 8006
        self.app = 'app.main:app'


config = Config()


def start_server():
    # 3.加载环境变量
    load_dotenv(config.env_file)
    load_dotenv(config.env_local_file, override=True)
    reload = os.getenv('DEV_RELOAD', 'false').lower() in ('1', 'true', 'yes')
    # Windows + 远程 MySQL：reload=True 会频繁重启子进程，远端易重置连接(WinError 10054)
    print(f'启动服务：{config.host}:{config.port} reload={reload}')
    uvicorn.run(config.app, host=config.host, port=config.port, reload=reload)


if __name__ == '__main__':
    start_server()
