"""
查看数据库中用户的密码哈希信息

用法（在 jeecg-fastapi 目录）:
    python scripts/inspect_user.py
"""
# 1.导包
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sqlalchemy import create_engine, text

from app.core.config import settings
from app.core.security import encrypt_password, verify_password, _legacy_encrypt_password


# 2.配置类
class Config:
    def __init__(self):
        self.root_path = str(ROOT).replace('\\', '/') + '/'
        self.username = 'gousan'


config = Config()


def inspect_user():
    # 3.查询用户并诊断哈希
    engine = create_engine(settings.database_url, pool_pre_ping=True)
    with engine.connect() as conn:
        row = conn.execute(
            text(
                'SELECT username, password, salt, phone, status, del_flag, create_time '
                'FROM sys_user WHERE username=:u'
            ),
            {'u': config.username},
        ).mappings().first()
    if not row:
        print(f'用户不存在：{config.username!r}')
        return
    print(f'用户信息：{dict(row)}')
    u, salt, stored = row['username'], row['salt'] or '', row['password'] or ''
    std = encrypt_password('PLACEHOLDER', u, salt)
    print(f'标准哈希格式样例（错误密码）：{std[:16]}...')
    print(f'存储哈希：{stored}')
    print(f'legacy 匹配 123456：{_legacy_encrypt_password("123456", u, salt) == stored}')
    print(f'verify 123456：{verify_password("123456", u, salt, stored)}')


if __name__ == '__main__':
    inspect_user()
