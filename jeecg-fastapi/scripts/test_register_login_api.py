"""
Live API 测试：注册然后登录

用法（在 jeecg-fastapi 目录）:
    python scripts/test_register_login_api.py
"""
# 1.导包
import base64
import sys
import time
from pathlib import Path

import requests
from Crypto.Cipher import AES
from sqlalchemy import create_engine, text

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.core.config import settings
from app.core.security import AES_IV, AES_KEY, encrypt_password, verify_password


# 2.配置类
class Config:
    def __init__(self):
        self.root_path = str(ROOT).replace('\\', '/') + '/'
        self.base_url = 'http://127.0.0.1:8000/jeecg-boot'
        self.phone = '13900001234'
        self.password = 'Test@123456'
        self.request_timeout = 15


config = Config()


def aes_encrypt(plain):
    # 3.AES 加密（模拟前端传参）
    data = plain.encode('utf-8')
    pad_len = 16 - (len(data) % 16)
    data = data + bytes([pad_len] * pad_len)
    cipher = AES.new(AES_KEY, AES.MODE_CBC, AES_IV)
    return base64.b64encode(cipher.encrypt(data)).decode()


def test_register_login_api():
    # 4.注册→查库→登录全流程
    username = f'autotest_{int(time.time()) % 100000}'

    sms = requests.post(
        f'{config.base_url}/sys/sms',
        json={'mobile': config.phone, 'smsmode': '1'},
        timeout=config.request_timeout,
    ).json()
    print(f'SMS 响应：{sms}')
    code = (sms.get('result') or {}).get('devCode') or ''
    if not code and '开发模式验证码' in (sms.get('message') or ''):
        code = (sms.get('message') or '').split(':')[-1].strip().rstrip('）')
    if not code:
        print('未获取到开发模式验证码')
        return

    reg = requests.post(
        f'{config.base_url}/sys/user/register',
        json={
            'username': username,
            'phone': config.phone,
            'smscode': code,
            'password': aes_encrypt(config.password),
        },
        timeout=config.request_timeout,
    ).json()
    print(f'注册响应：{reg}')
    if not reg.get('success'):
        return

    engine = create_engine(settings.database_url, pool_pre_ping=True)
    with engine.connect() as conn:
        row = conn.execute(
            text('SELECT username, password, salt FROM sys_user WHERE username=:u'),
            {'u': username},
        ).mappings().first()
    print(f'DB 行：{dict(row) if row else None}')
    if row:
        ok = verify_password(config.password, row['username'], row['salt'], row['password'])
        print(f'Direct verify_password：{ok}')
        expected = encrypt_password(config.password, row['username'], row['salt'])
        print(f'哈希匹配预期：{expected == row["password"]}')

    login = requests.post(
        f'{config.base_url}/sys/login',
        json={
            'username': username,
            'password': aes_encrypt(config.password),
            'captcha': '',
            'checkKey': 'test',
        },
        timeout=config.request_timeout,
    ).json()
    print(f'登录响应：{login}')


if __name__ == '__main__':
    test_register_login_api()
