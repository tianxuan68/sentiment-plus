"""
评论分析业务 API：聚合情感 / 关键词 / 属性，供前端演示与联调
路径与响应字段不变；风格对齐教学注释
"""

# 1.导包
from fastapi import APIRouter, Depends

from app.core.deps import get_current_user
from app.models.entities import SysUser
from app.schemas.response import Result
from app.schemas.review import AnalyzeRequest, BatchAnalyzeRequest
from app.services import sentiment_ai_client as ai

# 2.路由
router = APIRouter(prefix='/review', tags=['评论分析'])


@router.post('/analyze')
async def analyze(body: AnalyzeRequest, user: SysUser = Depends(get_current_user)):
    #  1.单条评论分析
    text = body.text.strip()
    if not text:
        return Result.error('评论文本不能为空', code=400)
    result = await ai.analyze_review(text, body.top_n)
    print(f'单条分析完成，来源：{result.source}，预览：{result.text_preview}')
    return Result.ok(result.model_dump())


@router.post('/analyze/batch')
async def analyze_batch(body: BatchAnalyzeRequest, user: SysUser = Depends(get_current_user)):
    # 批量评论分析（最多 10 条）
    texts = [t.strip() for t in body.texts if t and t.strip()]
    if not texts:
        return Result.error('请至少输入一条有效评论', code=400)
    if len(texts) > 10:
        return Result.error('批量分析最多 10 条评论', code=400)
    result = await ai.analyze_batch(texts, body.top_n)
    print(f'批量分析完成，条数：{result.total}，平均耗时：{result.avg_latency_ms}ms')
    return Result.ok(result.model_dump())


@router.get('/stats/polarity')
async def stats_polarity(user: SysUser = Depends(get_current_user)):
    return Result.ok((await ai.get_polarity_stats()).model_dump())


@router.get('/stats/trend')
async def stats_trend(user: SysUser = Depends(get_current_user)):
    points = await ai.get_trend_stats()
    return Result.ok([p.model_dump() for p in points])


@router.get('/stats/summary')
async def stats_summary(user: SysUser = Depends(get_current_user)):
    return Result.ok((await ai.get_dashboard_summary()).model_dump())


@router.get('/pros-cons')
async def pros_cons(user: SysUser = Depends(get_current_user)):
    return Result.ok((await ai.get_pros_cons()).model_dump())


@router.get('/models/compare')
async def models_compare(user: SysUser = Depends(get_current_user)):
    return Result.ok((await ai.get_model_compare()).model_dump())


@router.post('/keywords/compare')
async def keywords_compare(body: AnalyzeRequest, user: SysUser = Depends(get_current_user)):
    text = body.text.strip()
    if not text:
        return Result.error('评论文本不能为空', code=400)
    return Result.ok((await ai.compare_keywords(text, body.top_n)).model_dump())


@router.get('/stats/rating')
async def stats_rating(user: SysUser = Depends(get_current_user)):
    return Result.ok((await ai.get_rating_distribution()).model_dump())


@router.get('/ai/status')
async def ai_status(user: SysUser = Depends(get_current_user)):
    status = await ai.get_ai_status()
    print(f'AI 状态：mode={status.mode}，mock={status.mock_enabled}')
    return Result.ok(status.model_dump())
