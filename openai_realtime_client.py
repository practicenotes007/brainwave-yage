import websockets
import json
import base64
import logging
import time
from typing import Optional, Callable, Dict, List
import asyncio

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

'''
OpenAI的实时语音转文本服务
'''
class OpenAIRealtimeAudioTextClient:
    def __init__(self, api_key: str, model: str = "gpt-4o-realtime-preview"):
        self.api_key = api_key
        self.model = model
        self.ws = None
        self.session_id = None
        self.base_url = "wss://api.openai.com/v1/realtime"
        self.last_audio_time = None 
        self.auto_commit_interval = 5
        self.receive_task = None
        self.handlers: Dict[str, Callable[[dict], asyncio.Future]] = {}
        self.queue = asyncio.Queue()
        
    async def connect(self, modalities: List[str] = ["text"]):
        """Connect to OpenAI's realtime API and configure the session"""
        self.ws = await websockets.connect(
            f"{self.base_url}?model={self.model}",
            extra_headers={
                "Authorization": f"Bearer {self.api_key}",
                "OpenAI-Beta": "realtime=v1"
            }
        )
        
        # Wait for session creation
        response = await self.ws.recv()
        response_data = json.loads(response)
        if response_data["type"] == "session.created":
            self.session_id = response_data["session"]["id"]
            logger.info(f"Session created with ID: {self.session_id}")
            
            # Configure session
            await self.ws.send(json.dumps({
                "type": "session.update",
                "session": {
                    "modalities": modalities,
                    "input_audio_format": "pcm16",
                    "input_audio_transcription": None,
                    "turn_detection": None,
                }
            }))
        
        # Register the default handler
        self.register_handler("default", self.default_handler)
        
        # Start the receiver coroutine
        self.receive_task = asyncio.create_task(self.receive_messages())
    
    async def receive_messages(self):
        try:
            async for message in self.ws:
                data = json.loads(message)
                message_type = data.get("type", "default")
                handler = self.handlers.get(message_type, self.handlers.get("default"))
                if handler:
                    await handler(data)
                else:
                    logger.warning(f"No handler for message type: {message_type}")
        except websockets.exceptions.ConnectionClosed as e:
            logger.error(f"OpenAI WebSocket connection closed: {e}")
        except Exception as e:
            logger.error(f"Error in receive_messages: {e}", exc_info=True)
    
    def register_handler(self, message_type: str, handler: Callable[[dict], asyncio.Future]):
        self.handlers[message_type] = handler
    
    async def default_handler(self, data: dict):
        message_type = data.get("type", "unknown")
        logger.warning(f"Unhandled message type received from OpenAI: {message_type}")
    
    async def send_audio(self, audio_data: bytes):
        if self.ws and self.ws.open:
            await self.ws.send(json.dumps({
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(audio_data).decode('utf-8')
            }))
            logger.info("Sent input_audio_buffer.append message to OpenAI")
        else:
            logger.error("WebSocket is not open. Cannot send audio.")
    
    async def commit_audio(self):
        """Commit the audio buffer and notify OpenAI"""
        if self.ws and self.ws.open:
            commit_message = json.dumps({"type": "input_audio_buffer.commit"})
            await self.ws.send(commit_message)
            logger.info("Sent input_audio_buffer.commit message to OpenAI")
            # No recv call here. The receive_messages coroutine handles incoming messages.
        else:
            logger.error("WebSocket is not open. Cannot commit audio.")
    
    async def clear_audio_buffer(self):
        """Clear the audio buffer"""
        if self.ws and self.ws.open:
            clear_message = json.dumps({"type": "input_audio_buffer.clear"})
            await self.ws.send(clear_message)
            logger.info("Sent input_audio_buffer.clear message to OpenAI")
        else:
            logger.error("WebSocket is not open. Cannot clear audio buffer.")
    
    async def start_response(self, instructions: str):
        """Start a new response with given instructions"""
        if self.ws and self.ws.open:
            await self.ws.send(json.dumps({
                "type": "response.create",
                "response": {
                    "modalities": ["text"],
                    "instructions": instructions
                }
            }))
            logger.info(f"Started response with instructions: {instructions}")
        else:
            logger.error("WebSocket is not open. Cannot start response.")
    
    async def close(self):
        """Close the WebSocket connection"""
        if self.ws:
            await self.ws.close()
            logger.info("Closed OpenAI WebSocket connection")
        if self.receive_task:
            self.receive_task.cancel()
            try:
                await self.receive_task
            except asyncio.CancelledError:
                pass


'''
阿里云的实时语音转文本服务，示例代码
'''
class AliyunRealtimeAudioTextClient:
    def __init__(self, api_key: str, app_key: str):
        self.api_key = api_key
        self.app_key = app_key
        self.ws = None
        self.session_id = None
        self.base_url = "wss://nls-gateway-cn-shanghai.aliyuncs.com/ws/v1"
        self.last_audio_time = None 
        self.auto_commit_interval = 5
        self.receive_task = None
        self.handlers: Dict[str, Callable[[dict], asyncio.Future]] = {}
        self.queue = asyncio.Queue()

    async def connect(self):
        url = f"{self.base_url}?appkey={self.app_key}"
        self.ws = await websockets.connect(url)
        await self.send_start_request()

    async def send_start_request(self):
        request = {
            "header": {
                "app_key": self.app_key,
                "status": "start"
            },
            "parameter": {
                "speech_transcriber": {
                    "enable_intermediate_result": True,
                    "format": "pcm",
                    "sample_rate": 16000,
                    "domain": "general"
                }
            }
        }
        await self.ws.send(json.dumps(request))
        logger.info("Sent start request to Aliyun")

    async def send_audio(self, audio_data: bytes):
        if self.ws and self.ws.open:
            await self.ws.send(audio_data)
            logger.info("Sent audio data to Aliyun")
        else:
            logger.error("WebSocket is not open. Cannot send audio.")

    async def commit_audio(self):
        """Commit the audio buffer and notify Aliyun"""
        if self.ws and self.ws.open:
            commit_message = json.dumps({
                "header": {
                    "name": "speech.transcriber",
                    "status": "end"  # 修改：将"complete"改为"end"，符合阿里云要求的结束状态
                }
            })
            await self.ws.send(commit_message)
            logger.info("Sent commit message to Aliyun")
        else:
            logger.error("WebSocket is not open. Cannot commit audio.")

    async def clear_audio_buffer(self):
        """Clear the audio buffer"""
        if self.ws and self.ws.open:
            clear_message = json.dumps({
                "header": {
                    "name": "speech.transcriber",
                    "status": "cancel"
                }
            })
            await self.ws.send(clear_message)
            logger.info("Sent clear message to Aliyun")
        else:
            logger.error("WebSocket is not open. Cannot clear audio buffer.")

    async def start_response(self, instructions: str):
        """Start a new response with given instructions"""
        if self.ws and self.ws.open:
            start_message = json.dumps({
                "header": {
                    "name": "speech.transcriber",
                    "status": "start"
                },
                "parameter": {
                    "speech_transcriber": {
                        "enable_intermediate_result": True,
                        "format": "pcm",
                        "sample_rate": 16000,
                        "domain": "general"
                    }
                }
            })
            await self.ws.send(start_message)
            logger.info(f"Started response with instructions: {instructions}")
        else:
            logger.error("WebSocket is not open. Cannot start response.")

    def register_handler(self, message_type: str, handler: Callable[[dict], asyncio.Future]):
        self.handlers[message_type] = handler

    async def default_handler(self, data: dict):
        message_type = data.get("type", "unknown")
        logger.warning(f"Unhandled message type received from Aliyun: {message_type}")

    async def receive_messages(self):
        try:
            async for message in self.ws:
                data = json.loads(message)
                message_type = data.get("header", {}).get("name", "default")
                handler = self.handlers.get(message_type, self.handlers.get("default"))
                if handler:
                    await handler(data)
                else:
                    logger.warning(f"No handler for message type: {message_type}")
        except websockets.exceptions.ConnectionClosed as e:
            logger.error(f"Aliyun WebSocket connection closed: {e}")
        except Exception as e:
            logger.error(f"Error in receive_messages: {e}", exc_info=True)

    async def connect_and_receive(self):
        await self.connect()
        self.receive_task = asyncio.create_task(self.receive_messages())
 
    async def close(self):
        """Close the WebSocket connection"""
        if self.ws:
            await self.ws.close()
            logger.info("Closed Aliyun WebSocket connection")
        if self.receive_task:
            self.receive_task.cancel()
            try:
                await self.receive_task
            except asyncio.CancelledError:
                pass