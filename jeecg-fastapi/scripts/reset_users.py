"""
重置/创建默认用户

用法（在 jeecg-fastapi 目录）:
    python scripts/reset_users.py
"""
# 1.导包
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.core.security import encrypt_password, verify_password
from app.db.session import SessionLocal
from app.models.entities import SysUser, SysUserDepart, SysUserRole
from app.utils.common import new_id


# 2.配置类
class Config:
    def __init__(self):
        self.root_path = str(ROOT).replace('\\', '/') + '/'
        self.admin_role = 'f6817f48af4fb3af11b9e8bf182f618b'
        self.test_role = 'ee8626f80f7c2619917b6236f3a7f02b'
        self.depart_id = '4f1765520d6346f9bd9c79e2479e5b12'
        self.org_code = 'A01A03'
        self.default_pwd = '123456'
        self.users = [
            {'username': 'admin', 'realname': '管理员', 'role_id': self.admin_role, 'salt': 'RCGTeGiH'},
            {'username': 'gouda', 'realname': 'gouda', 'role_id': self.admin_role},
            {'username': 'gouer', 'realname': 'gouer', 'role_id': self.test_role},
        ]


config = Config()


def _ensure_depart(db, user_id):
    # 3.确保用户部门关联
    if not db.query(SysUserDepart).filter(SysUserDepart.user_id == user_id).first():
        db.add(SysUserDepart(id=new_id(), user_id=user_id, dep_id=config.depart_id))


def _set_role(db, user_id, role_id):
    # 4.设置用户角色
    db.query(SysUserRole).filter(SysUserRole.user_id == user_id).delete()
    db.add(SysUserRole(id=new_id(), user_id=user_id, role_id=role_id))


def upsert_user(db, spec):
    # 5.创建或更新单个用户
    username = spec['username']
    user = db.query(SysUser).filter(SysUser.username == username).first()
    salt = (user.salt if user and user.salt else None) or spec.get('salt') or new_id()[:8]
    password = encrypt_password(config.default_pwd, username, salt)
    now = datetime.now()

    if user:
        user.realname = spec.get('realname') or username
        user.password = password
        user.salt = salt
        user.status = 1
        user.del_flag = 0
        user.org_code = config.org_code
        user.update_time = now
    else:
        user = SysUser(
            id=new_id(),
            username=username,
            realname=spec.get('realname') or username,
            password=password,
            salt=salt,
            status=1,
            del_flag=0,
            org_code=config.org_code,
            create_by='system',
            create_time=now,
        )
        db.add(user)
        db.flush()

    _set_role(db, user.id, spec['role_id'])
    _ensure_depart(db, user.id)
    assert verify_password(config.default_pwd, username, salt, password), f'{username} 密码校验失败'
    return user


def reset_users():
    # 6.主流程：批量重置用户
    db = SessionLocal()
    try:
        for spec in config.users:
            user = upsert_user(db, spec)
            print(
                f'OK {user.username}/{config.default_pwd} '
                f'role={spec["role_id"][-8:]} hash={user.password}'
            )
        db.commit()
        print('\n全部用户已重置，请重启后端：python run.py')
        return 0
    except Exception as exc:
        db.rollback()
        print(f'FAIL：{exc}')
        return 1
    finally:
        db.close()


if __name__ == '__main__':
    raise SystemExit(reset_users())
