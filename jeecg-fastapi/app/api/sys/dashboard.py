"""
首页统计（精简版占位数据）
"""

# 1.导包
from datetime import datetime, timedelta

from fastapi import APIRouter

from app.schemas.response import Result

# 2.路由
router = APIRouter(tags=['首页统计'])


@router.get('/loginfo')
def loginfo():
    # 今日访问占位
    return Result.ok({
        'todayIp': 0,
        'todayVisitCount': 0,
        'totalVisitCount': 0,
    })


@router.get('/visitInfo')
def visit_info():
    # 近 7 日访问占位
    today = datetime.now()
    result = []
    for offset in range(6, -1, -1):
        day = today - timedelta(days=offset)
        result.append({
            'type': day.strftime('%m-%d'),
            'ip': 0,
            'visit': 0,
        })
    return Result.ok(result)
