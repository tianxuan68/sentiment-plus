"""
检查情感分析菜单（sa010-sa015）是否存在于 sys_permission

用法（在 jeecg-fastapi 目录）:
    python scripts/check_sentiment_menus.py
"""
# 1.导包
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sqlalchemy import text

from app.db.session import SessionLocal


# 2.配置类
class Config:
    def __init__(self):
        self.root_path = str(ROOT).replace('\\', '/') + '/'
        self.expected_menu_ids = ['sa010', 'sa011', 'sa012', 'sa013', 'sa014', 'sa015']


config = Config()


def check_sentiment_menus():
    # 3.查询并校验菜单
    with SessionLocal() as db:
        rows = db.execute(
            text(
                'SELECT id, name, url, component FROM sys_permission '
                "WHERE id LIKE 'sa%' ORDER BY id"
            )
        ).fetchall()
        found = {r[0] for r in rows}
        print(f'找到情感菜单行数：{len(rows)}')
        for r in rows:
            print(f'  {r[0]} | {r[1]} | {r[2]} | {r[3]}')

        missing = [x for x in config.expected_menu_ids if x not in found]
        if missing:
            print(f'\n缺失菜单：{", ".join(missing)}')
            print('请执行：mysql ... < sql/patch_add_sentiment_menus.sql')
            return 1

        user_count = db.execute(text('SELECT COUNT(*) FROM sys_user')).scalar()
        print(f'\nOK 全部 sa010-sa015 已存在；sys_user 数量 = {user_count}')
        return 0


if __name__ == '__main__':
    raise SystemExit(check_sentiment_menus())
