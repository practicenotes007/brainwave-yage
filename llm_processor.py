import os
from abc import ABC, abstractmethod
import google.generativeai as genai
from openai import OpenAI, AsyncOpenAI
from typing import AsyncGenerator, Generator, Optional
import logging

import httpx  # 新增导入语句以修复httpx未定义错误

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

'''
大模型服务的抽象类
'''
class LLMProcessor(ABC):
    @abstractmethod
    async def process_text(self, text: str, prompt: str, model: Optional[str] = None) -> AsyncGenerator[str, None]:
        pass
    
    @abstractmethod
    def process_text_sync(self, text: str, prompt: str, model: Optional[str] = None) -> str:
        pass

'''
大模型服务为 Google Gemini，具体实现
'''
class GeminiProcessor(LLMProcessor):
    def __init__(self, default_model: str = 'gemini-1.5-pro'):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise EnvironmentError("GOOGLE_API_KEY is not set")
        genai.configure(api_key=api_key)
        self.default_model = default_model

    async def process_text(self, text: str, prompt: str, model: Optional[str] = None) -> AsyncGenerator[str, None]:
        all_prompt = f"{prompt}\n\n{text}"
        model_name = model or self.default_model
        logger.info(f"Using model: {model_name} for processing")
        logger.info(f"Prompt: {all_prompt}")
        genai_model = genai.GenerativeModel(model_name)
        response = await genai_model.generate_content_async(
            all_prompt,
            stream=True
        )
        async for chunk in response:
            if chunk.text:
                yield chunk.text

    def process_text_sync(self, text: str, prompt: str, model: Optional[str] = None) -> str:
        all_prompt = f"{prompt}\n\n{text}"
        model_name = model or self.default_model
        logger.info(f"Using model: {model_name} for sync processing")
        logger.info(f"Prompt: {all_prompt}")
        genai_model = genai.GenerativeModel(model_name)
        response = genai_model.generate_content(all_prompt)
        return response.text

'''
大模型服务为 OpenAI GPT，具体实现
'''
class GPTProcessor(LLMProcessor):
    def __init__(self):
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OpenAI API key not found in environment variables")
        self.async_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.sync_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.default_model = "gpt-4"

    async def process_text(self, text: str, prompt: str, model: Optional[str] = None) -> AsyncGenerator[str, None]:
        all_prompt = f"{prompt}\n\n{text}"
        model_name = model or self.default_model
        logger.info(f"Using model: {model_name} for processing")
        logger.info(f"Prompt: {all_prompt}")
        response = await self.async_client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": all_prompt}
            ],
            stream=True
        )
        async for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def process_text_sync(self, text: str, prompt: str, model: Optional[str] = None) -> str:
        all_prompt = f"{prompt}\n\n{text}"
        model_name = model or self.default_model
        logger.info(f"Using model: {model_name} for sync processing")
        logger.info(f"Prompt: {all_prompt}")
        response = self.sync_client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": all_prompt}
            ]
        )
        return response.choices[0].message.content

'''
大模型服务为 DeepSeek，具体实现
'''
class DeepSeekProcessor(LLMProcessor):
    def __init__(self, default_model: str = 'deepseek-chat'):  # 修改默认模型为deepseek-chat
        self.default_model = default_model
        self.api_key = os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise EnvironmentError("DEEPSEEK_API_KEY is not set")

    async def process_text(self, text: str, prompt: str, model: Optional[str] = None) -> AsyncGenerator[str, None]:
        all_prompt = f"{prompt}\n\n{text}"
        model_name = model or self.default_model
        logger.info(f"Using model: {model_name} for processing")
        logger.info(f"Prompt: {all_prompt}")

        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {
            "model": model_name,
            "prompt": all_prompt,
            "max_tokens": 512,
            "stream": True
        }

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://api.deepseek.com/beta/v1/completions",  # 修改API地址为beta路径
                    json=payload,
                    headers=headers
                )
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.strip():
                        yield line.strip()
        except httpx.ConnectError as e:
            logger.error(f"Failed to connect to DeepSeek API endpoint: {e.request.url}. Error: {str(e)}. Please check DNS resolution (run 'nslookup api.deepseek.ai') and network connectivity.")
            yield f"Connection error: {str(e)}"
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error {e.response.status_code} from {e.request.url}: {e.response.text}")
            yield f"API error: {e.response.status_code} - {e.response.text}"

    def process_text_sync(self, text: str, prompt: str, model: Optional[str] = None) -> str:
        all_prompt = f"{prompt}\n\n{text}"
        model_name = model or self.default_model
        logger.info(f"Using model: {model_name} for sync processing")
        logger.info(f"Prompt: {all_prompt}")

        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {
            "model": model_name,
            "prompt": all_prompt,
            "max_tokens": 512
        }

        try:
            response = httpx.post(
                "https://api.deepseek.com/beta/v1/completions",  # 修改API地址为beta路径
                json=payload,
                headers=headers
            )
            response.raise_for_status()
            return response.json()["choices"][0]["text"]
        except httpx.ConnectError as e:
            logger.error(f"Failed to connect to DeepSeek API endpoint: {e.request.url}. Error: {str(e)}. Please check DNS resolution (run 'nslookup api.deepseek.ai') and network connectivity.")
            return f"Connection error: {str(e)}"
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error (sync): {e.response.status_code} from {e.request.url} - {e.response.text}")
            return f"API error: {e.response.status_code} - {e.response.text}"

def get_llm_processor(model: str) -> LLMProcessor:
    model = model.lower()
    logger.debug(f"Creating processor for model: {model}")  # 添加日志记录模型选择过程
    if model.startswith(('gemini', 'gemini-')):
        logger.debug("Selected GeminiProcessor")
        return GeminiProcessor(default_model=model)
    elif model.startswith(('gpt-', 'o1-')):
        logger.debug("Selected GPTProcessor")
        return GPTProcessor()
    elif model.startswith('deepseek'):
        logger.debug("Selected DeepSeekProcessor")
        return DeepSeekProcessor(default_model=model)
    else:
        logger.error(f"Unsupported model type: {model}")
        raise ValueError(f"Unsupported model type: {model}")
