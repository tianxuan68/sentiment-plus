"""
诊断存储哈希是否匹配正确或错误的密码链路

用法（在 jeecg-fastapi 目录）:
    python scripts/diagnose_hash.py
"""
# 1.导包
import base64
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from Crypto.Cipher import AES

from app.core.security import AES_IV, AES_KEY, encrypt_password, parse_client_password


# 2.配置类
class Config:
    def __init__(self):
        self.root_path = str(ROOT).replace('\\', '/') + '/'
        self.username = 'gousan'
        self.salt = '53bc0d16'
        self.stored_hash = '8b2e2eacd1219831'
        self.candidates = [
            '123456',
            'gousan',
            'Gousan123',
            'Gousan@123',
            'Test@123',
            'Aa123456',
            'Abc12345',
            'Password1',
            'password123',
            '19857027644',
            'Gousan1234',
            'gousan123',
            'GoUsan123',
        ]


config = Config()


def aes_encrypt(plain):
    # 3.AES 加密（模拟前端）
    data = plain.encode('utf-8')
    pad_len = 16 - (len(data) % 16)
    data = data + bytes([pad_len] * pad_len)
    cipher = AES.new(AES_KEY, AES.MODE_CBC, AES_IV)
    return base64.b64encode(cipher.encrypt(data)).decode()


def diagnose_hash():
    # 4.逐候选密码诊断哈希匹配
    print(f'用户={config.username} salt={config.salt} stored={config.stored_hash}')
    for pwd in config.candidates:
        enc = aes_encrypt(pwd)
        dec = parse_client_password(enc)
        correct_hash = encrypt_password(pwd, config.username, config.salt)
        dec_hash = encrypt_password(dec, config.username, config.salt)
        buggy_hash = encrypt_password(enc, config.username, config.salt)
        if config.stored_hash == correct_hash:
            print(f'  CORRECT pipeline password={pwd!r}')
        if config.stored_hash == buggy_hash and enc != pwd:
            print(f'  BUGGY (hash AES ciphertext) password={pwd!r} enc={enc[:24]}...')
        if dec != pwd:
            print(f'  parse mismatch for {pwd!r}: dec={dec!r}')


if __name__ == '__main__':
    diagnose_hash()
