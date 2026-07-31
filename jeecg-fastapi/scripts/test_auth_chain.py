"""
端到端验证：注册/改密/登录 密码链路

用法（在 jeecg-fastapi 目录）:
    python scripts/test_auth_chain.py
"""
# 1.导包
import base64
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from Crypto.Cipher import AES

from app.core.security import (
    AES_IV,
    AES_KEY,
    encrypt_password,
    parse_client_password,
    verify_password,
)


# 2.配置类
class Config:
    def __init__(self):
        self.root_path = str(ROOT).replace('\\', '/') + '/'
        self.demo_username = 'demo_user'
        self.demo_salt = 'abcd1234'
        self.demo_password = 'Test@123456'
        self.new_salt = 'xyz98765'
        self.new_password = 'NewPass@789'


config = Config()


def aes_encrypt_frontend(plain):
    # 3.模拟前端 AES 加密
    data = plain.encode('utf-8')
    pad_len = 16 - (len(data) % 16)
    data = data + bytes([pad_len] * pad_len)
    cipher = AES.new(AES_KEY, AES.MODE_CBC, AES_IV)
    return base64.b64encode(cipher.encrypt(data)).decode()


def simulate_register(username, password, salt):
    # 4.模拟注册密码处理
    enc = aes_encrypt_frontend(password)
    plain = parse_client_password(enc)
    return encrypt_password(plain, username, salt)


def simulate_login(username, password, salt, stored):
    # 5.模拟登录密码校验
    enc = aes_encrypt_frontend(password)
    plain = parse_client_password(enc)
    return verify_password(plain, username, salt, stored)


def simulate_password_change(username, new_password, old_salt, new_salt):
    # 6.模拟改密
    enc = aes_encrypt_frontend(new_password)
    plain = parse_client_password(enc)
    stored = encrypt_password(plain, username, new_salt)
    return new_salt, stored


def test_auth_chain():
    # 7.运行注册→登录→改密链路测试
    stored = simulate_register(config.demo_username, config.demo_password, config.demo_salt)
    assert simulate_login(
        config.demo_username, config.demo_password, config.demo_salt, stored
    ), 'register -> login failed'

    _, new_stored = simulate_password_change(
        config.demo_username, config.new_password, config.demo_salt, config.new_salt
    )
    assert not simulate_login(
        config.demo_username, config.demo_password, config.new_salt, new_stored
    ), 'old password should fail'
    assert simulate_login(
        config.demo_username, config.new_password, config.new_salt, new_stored
    ), 'change -> login failed'

    print('Auth chain OK：register / change-password / login')

    try:
        from sqlalchemy import create_engine, text
        from app.core.config import settings

        engine = create_engine(settings.database_url, pool_pre_ping=True)
        with engine.connect() as conn:
            rows = conn.execute(
                text(
                    'SELECT username, password, salt, status FROM sys_user '
                    'WHERE del_flag=0 ORDER BY create_time DESC LIMIT 5'
                )
            ).mappings().all()
            print('\n最近 DB 用户：')
            for r in rows:
                ok = verify_password('123456', r['username'], r['salt'] or '', r['password'] or '')
                print(f'  {r["username"]}：salt={r["salt"]} default123456={ok}')
    except Exception as exc:
        print(f'\nDB 跳过：{exc}')


if __name__ == '__main__':
    test_auth_chain()
