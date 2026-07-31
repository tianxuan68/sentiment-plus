"""JeecgBoot 系统消息 WebSocket（精简版：维持连接 + 心跳）。"""

# 1.导包
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

# 2.路由
router = APIRouter(tags=["WebSocket"])


@router.websocket("/websocket/{user_id}")
async def system_websocket(websocket: WebSocket, user_id: str):
    subprotocols = websocket.scope.get("subprotocols") or []
    subprotocol = subprotocols[0] if subprotocols else None
    await websocket.accept(subprotocol=subprotocol)
    print(f'WebSocket 已连接：{user_id}')
    try:
        while True:
            data = await websocket.receive_text()
            # 前端 @vueuse/core heartbeat 发送 "ping"
            if data == "ping":
                await websocket.send_text("ping")
    except WebSocketDisconnect:
        print(f'WebSocket 已断开：{user_id}')
    except Exception as exc:
        print(f'WebSocket 关闭（{user_id}）：{exc}')
