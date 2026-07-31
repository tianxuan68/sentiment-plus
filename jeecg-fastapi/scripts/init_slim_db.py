"""
初始化精简版数据库并验证登录链路

用法（在 jeecg-fastapi 目录）:
    python scripts/init_slim_db.py
"""
# 1.导包
import sys
from pathlib import Path

import pymysql
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.core.security import create_token, verify_password, verify_token
from app.services.dict_service import query_all_dict_items
from app.services.permission_service import build_menu_tree, query_permissions_by_user
from app.services.user_service import get_user_roles


# 2.配置类
class Config:
    def __init__(self):
        self.root_path = str(ROOT).replace('\\', '/') + '/'
        self.sql_file = self.root_path + 'sql/jeecgboot-slim.sql'
        self._load_from_settings()

    def _load_from_settings(self):
        from app.core.config import settings
        body = settings.database_url.split('://', 1)[1]
        auth, hostpart = body.rsplit('@', 1)
        user, password = auth.split(':', 1)
        host_port, database = hostpart.split('/', 1)
        if '?' in database:
            database = database.split('?', 1)[0]
        host, port = host_port.split(':')
        self.host = host
        self.port = int(port)
        self.user = user
        self.password = password.replace('%40', '@')
        self.database = database


config = Config()


def run_sql_file(conn, sql_path):
    # 3.执行 SQL 文件
    content = Path(sql_path).read_text(encoding='utf-8')
    statements = []
    buf = []
    for line in content.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith('--'):
            continue
        buf.append(line)
        if stripped.endswith(';'):
            statements.append('\n'.join(buf))
            buf = []
    with conn.cursor() as cur:
        for stmt in statements:
            cur.execute(stmt)
    conn.commit()
    print(f'已执行 SQL 语句数：{len(statements)}')


def verify_backend(db_url):
    # 4.校验登录 / 菜单 / 字典链路
    engine = create_engine(db_url, pool_pre_ping=True)
    with Session(engine) as db:
        user = db.execute(
            text(
                "SELECT id, username, password, salt, status, del_flag "
                "FROM sys_user WHERE username='admin'"
            )
        ).mappings().first()
        assert user, 'admin 用户不存在'
        assert user['status'] == 1 and user['del_flag'] == 0, 'admin 用户状态异常'
        assert verify_password('123456', user['username'], user['salt'], user['password']), 'admin 密码校验失败'

        roles = get_user_roles(db, user['id'])
        assert roles, 'admin 未分配角色'

        perms = query_permissions_by_user(db, user['id'])
        assert len(perms) >= 5, f'admin 菜单权限不足：{len(perms)}'
        menu = build_menu_tree(perms)
        assert menu, '菜单树为空'

        dict_items = query_all_dict_items(db)
        assert dict_items, '字典数据为空'

        token = create_token(user['username'], user['password'])
        assert verify_token(token, user['username'], user['password']), 'Token 校验失败'

    tables = [
        'sys_user', 'sys_role', 'sys_user_role', 'sys_permission',
        'sys_role_permission', 'sys_depart', 'sys_user_depart',
        'sys_dict', 'sys_dict_item',
    ]
    with engine.connect() as conn:
        for table in tables:
            count = conn.execute(text(f'SELECT COUNT(*) FROM `{table}`')).scalar()
            assert count and count > 0, f'{table} 无数据'
            print(f'  OK {table}：{count} 行')


def process_data():
    # 5.主流程
    print(f'连接：{config.user}@{config.host}:{config.port}/{config.database}')
    conn = pymysql.connect(
        host=config.host,
        port=config.port,
        user=config.user,
        password=config.password,
        charset='utf8mb4',
        autocommit=False,
        connect_timeout=15,
    )
    try:
        print(f'执行 SQL：{config.sql_file}')
        run_sql_file(conn, config.sql_file)
        print('SQL 执行完成')
    finally:
        conn.close()

    db_url = (
        f"mysql+pymysql://{config.user}:{config.password.replace('@', '%40')}"
        f"@{config.host}:{config.port}/{config.database}?charset=utf8mb4"
    )
    print('开始校验后端逻辑...')
    verify_backend(db_url)
    print('全部检查通过，可用 admin / 123456 登录')


if __name__ == '__main__':
    process_data()
