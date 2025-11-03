import logging
from fastapi import WebSocket


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        logger.info(
            f"Websocket client {websocket.client}:{websocket.url} disconnected"
        )
        self.active_connections.remove(websocket)

    async def send_json(self, websocket: WebSocket, data: any):
        await websocket.send_json(data)
