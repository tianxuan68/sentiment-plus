"""
检测 DATABASE_URL 是否可用

用法（在 jeecg-fastapi 目录）:
    python scripts/check_db.py
"""
# 1.导包
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.core.config import settings
from app.db.session import ping_db, warmup_db_pool


# 2.配置类
class Config:
    def __init__(self):
        self.root_path = str(ROOT).replace('\\', '/') + '/'


config = Config()


def check_db():
    # 3.检测数据库连接
    print(f'数据库地址：{settings.database_url.split("@")[-1]}')
    try:
        t = time.perf_counter()
        elapsed = ping_db()
        print(f'OK ping_db 耗时：{elapsed:.2f}s（总 {time.perf_counter() - t:.2f}s）')
        t = time.perf_counter()
        warmup_db_pool()
        print(f'OK warmup 耗时：{time.perf_counter() - t:.2f}s')
        return 0
    except Exception as exc:
        print(f'FAIL：{exc}')
        print(
            '\n建议：\n'
            '  1. 确认本地 MySQL 已启动，库 jeecg-boot 可访问\n'
            '  2. 复制 .env.local.example 为 .env.local，填写 DATABASE_URL\n'
            '  3. 首次初始化：python scripts/init_slim_db.py\n'
            '  4. 运行 python run.py 时本地库可设 DEV_RELOAD=true'
        )
        return 1


if __name__ == '__main__':
    raise SystemExit(check_db())
