"""
本地验证 JeecgBoot 密码算法是否与数据库哈希一致

用法（在 jeecg-fastapi 目录）:
    python scripts/verify_password.py
"""
# 1.导包
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.core.security import encrypt_password, verify_password


# 2.配置类
class Config:
    def __init__(self):
        self.root_path = str(ROOT).replace('\\', '/') + '/'
        self.username = 'admin'
        self.salt = 'RCGTeGiH'
        self.password = '123456'
        self.stored_hash = 'cb362cfeefbf3d8d'


config = Config()


def verify_password_demo():
    # 3.校验密码哈希
    ok = verify_password(config.password, config.username, config.salt, config.stored_hash)
    generated = encrypt_password(config.password, config.username, config.salt)
    print(f'admin 密码校验结果：{ok}')
    print(f'生成哈希：{generated}')


if __name__ == '__main__':
    verify_password_demo()
