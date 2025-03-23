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
    def __init__(self, default_model: str = 'deepseek-chat'):
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
                response = await client.post("https://api.deepseek.ai/v1/completions", json=payload, headers=headers)
                response.raise_for_status()  # 新增：检查HTTP状态码
                async for line in response.aiter_lines():
                    if line.strip():
                        yield line.strip()
        except httpx.ConnectError as e:
            logger.error(f"Connection error to DeepSeek API: {str(e)}")
            yield f"Connection error: {str(e)}"  # 返回错误信息给客户端
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error {e.response.status_code}: {e.response.text}")
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
            response = httpx.post("https://api.deepseek.ai/v1/completions", json=payload, headers=headers)
            response.raise_for_status()
            return response.json()["choices"][0]["text"]
        except httpx.ConnectError as e:
            logger.error(f"Connection error to DeepSeek API (sync): {str(e)}")
            return f"Connection error: {str(e)}"
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error (sync): {e.response.status_code} - {e.response.text}")
            return f"API error: {e.response.status_code} - {e.response.text}"

def get_llm_processor(model: str) -> LLMProcessor:
    model = model.lower()
    if model.startswith(('gemini', 'gemini-')):
        return GeminiProcessor(default_model=model)
    elif model.startswith(('gpt-', 'o1-')):
        return GPTProcessor()
    elif model.startswith('deepseek'):
        return DeepSeekProcessor(default_model=model)
    else:
        raise ValueError(f"Unsupported model type: {model}")
