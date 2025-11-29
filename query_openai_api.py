"""
将 GraphRAG 的 local_search 查询包装成 OpenAI 兼容的 Chat Completions 接口。

用法示例：

    # 在某个 GraphRAG 项目根目录启动服务（包含 settings.yaml / output/ 等）
    python query_openai_api.py --root /root/rag_case_quality --host 0.0.0.0 --port 8000

然后可以用 OpenAI 兼容的方式调用：

    POST http://localhost:8000/v1/chat/completions
    Content-Type: application/json

    {
      "model": "graphrag-local",
      "messages": [
        {"role": "user", "content": "请根据本地知识库回答这个问题"}
      ],
      "stream": true  # 可选：支持流式返回
    }

注意：
- 支持流式和非流式调用（通过 stream 参数控制）。
- 每次调用都会使用 GraphRAG 的 local_search 或 global_search API。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import uvicorn
import yaml
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

import graphrag.api as graphrag_api
from graphrag.config.create_graphrag_config import create_graphrag_config
from graphrag.utils.api import create_storage_from_config
from graphrag.utils.storage import load_table_from_storage, storage_has_table


logger = logging.getLogger("query_openai_api")

# 由 main() 注入的启动参数
_SERVICE_PARAMS: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI 生命周期管理：启动时初始化 SearchService。"""
    root_dir = Path(_SERVICE_PARAMS["root_dir"]).resolve()
    community_level = int(_SERVICE_PARAMS.get("community_level", 2))
    response_type = str(_SERVICE_PARAMS.get("response_type", "multiple paragraphs"))
    method = str(_SERVICE_PARAMS.get("method", "local")).lower()

    # 注册自定义模块（兼容 ext_parquet / milvus），与 run_update.py 一致
    try:
        sys.path.insert(0, ".")
        from custom_parquet_loader import register_parquet_loader  # type: ignore
        from custom_workflows import register_custom_workflows  # type: ignore
        from milvusdb import register_milvus_vector_store  # type: ignore

        register_parquet_loader()
        register_custom_workflows()
        register_milvus_vector_store()
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "注册自定义模块失败（custom_parquet_loader/custom_workflows/milvusdb）: %s",
            exc,
        )

    try:
        service = SearchService(
            root_dir=root_dir,
            community_level=community_level,
            response_type=response_type,
        )
        await service.load_index_data()
        app.state.service = service
        app.state.search_method = method
        logger.info(
            "SearchService initialized: root=%s, community_level=%d, method=%s",
            root_dir,
            community_level,
            method,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("初始化 SearchService 失败")
        raise

    yield  # 应用运行期间

    # 可选：清理资源
    logger.info("Shutting down SearchService")


app = FastAPI(
    title="GraphRAG LocalSearch OpenAI-Compatible API",
    lifespan=lifespan,
)


class SearchService:
    """封装 GraphRAG local_search 的服务对象。"""

    def __init__(
        self,
        root_dir: Path,
        community_level: int = 2,
        response_type: str = "multiple paragraphs",
    ) -> None:
        self.root_dir = root_dir.resolve()
        self.community_level = community_level
        self.response_type = response_type

        # 加载配置（兼容 ext_parquet，自定义 loader，逻辑与 run_update.py 一致）
        logger.info("Loading GraphRAG config from %s", self.root_dir)

        settings_file = self.root_dir / "settings.yaml"
        if not settings_file.exists():
            raise FileNotFoundError(f"未找到配置文件: {settings_file}")

        # 加载 .env
        env_file = self.root_dir / ".env"
        if env_file.exists():
            load_dotenv(env_file)
            logger.info("  ✓ 环境变量加载完成: %s", env_file)

        # 手动读取并解析配置文件（需要特殊处理 ext_parquet loader）
        import os
        import re

        with open(settings_file, "r", encoding="utf-8") as f:
            yaml_content = f.read()

        # 替换环境变量 ${VAR} -> 实际值
        def replace_env_vars(match: "re.Match[str]") -> str:
            var_name = match.group(1)
            return os.environ.get(var_name, match.group(0))

        yaml_content = re.sub(r"\$\{(\w+)\}", replace_env_vars, yaml_content)
        config_data = yaml.safe_load(yaml_content)

        # 临时保存自定义的 file_type（GraphRAG 验证只接受 csv/text/json）
        real_file_type = config_data.get("input", {}).get("file_type", "text")
        if "input" in config_data:
            config_data["input"]["file_type"] = "text"

        # 创建配置对象
        self.config = create_graphrag_config(config_data, str(self.root_dir))

        # 恢复真正的 file_type（用于自定义 ext_parquet loader）
        self.config.input.file_type = real_file_type

        # 以下 DataFrame 在异步的 load_index_data 中加载
        self.communities: Optional[pd.DataFrame] = None
        self.community_reports: Optional[pd.DataFrame] = None
        self.text_units: Optional[pd.DataFrame] = None
        self.relationships: Optional[pd.DataFrame] = None
        self.entities: Optional[pd.DataFrame] = None
        self.covariates: Optional[pd.DataFrame] = None

    async def load_index_data(self) -> None:
        """异步加载 local_search 所需的 parquet 数据（单 index）。"""
        logger.info("Loading index data from output storage...")
        storage_obj = create_storage_from_config(self.config.output)

        self.communities = await load_table_from_storage(
            name="communities", storage=storage_obj
        )
        self.community_reports = await load_table_from_storage(
            name="community_reports", storage=storage_obj
        )
        self.text_units = await load_table_from_storage(
            name="text_units", storage=storage_obj
        )
        self.relationships = await load_table_from_storage(
            name="relationships", storage=storage_obj
        )
        self.entities = await load_table_from_storage(
            name="entities", storage=storage_obj
        )

        if await storage_has_table("covariates", storage_obj):
            self.covariates = await load_table_from_storage(
                name="covariates", storage=storage_obj
            )
        else:
            self.covariates = None

        logger.info(
            "Loaded index data: communities=%d, reports=%d, text_units=%d, "
            "relationships=%d, entities=%d, covariates=%s",
            len(self.communities),
            len(self.community_reports),
            len(self.text_units),
            len(self.relationships),
            len(self.entities),
            "yes" if self.covariates is not None else "no",
        )

    async def run_local_search(self, query: str) -> str:
        """执行一次 local_search，返回回答文本。"""
        logger.info("Running local_search query: %s", query)
        response, _context = await graphrag_api.local_search(
            config=self.config,
            entities=self.entities,
            communities=self.communities,
            community_reports=self.community_reports,
            text_units=self.text_units,
            relationships=self.relationships,
            covariates=self.covariates,
            community_level=self.community_level,
            response_type=self.response_type,
            query=query,
            verbose=False,
        )
        # response 可能是 str 或 dict/list，这里统一转成字符串
        if isinstance(response, str):
            return response
        return json.dumps(response, ensure_ascii=False)

    async def run_global_search(self, query: str) -> str:
        """执行一次 global_search，返回回答文本。"""
        logger.info("Running global_search query: %s", query)
        response, _context = await graphrag_api.global_search(
            config=self.config,
            entities=self.entities,  # type: ignore[arg-type]
            communities=self.communities,  # type: ignore[arg-type]
            community_reports=self.community_reports,  # type: ignore[arg-type]
            community_level=self.community_level,
            dynamic_community_selection=False,
            response_type=self.response_type,
            query=query,
            verbose=False,
        )
        if isinstance(response, str):
            return response
        return json.dumps(response, ensure_ascii=False)

    async def run_search(self, method: str, query: str) -> str:
        """根据 method 调用 local 或 global 搜索。"""
        method = (method or "local").lower()
        if method == "local":
            return await self.run_local_search(query)
        if method == "global":
            return await self.run_global_search(query)
        raise ValueError(f"Unsupported method: {method}")

    async def run_local_search_streaming(self, query: str):
        """执行流式 local_search，返回异步生成器。"""
        logger.info("Running streaming local_search query: %s", query)
        async for chunk in graphrag_api.local_search_streaming(
            config=self.config,
            entities=self.entities,
            communities=self.communities,
            community_reports=self.community_reports,
            text_units=self.text_units,
            relationships=self.relationships,
            covariates=self.covariates,
            community_level=self.community_level,
            response_type=self.response_type,
            query=query,
            verbose=False,
        ):
            yield chunk

    async def run_global_search_streaming(self, query: str):
        """执行流式 global_search，返回异步生成器。"""
        logger.info("Running streaming global_search query: %s", query)
        async for chunk in graphrag_api.global_search_streaming(
            config=self.config,
            entities=self.entities,  # type: ignore[arg-type]
            communities=self.communities,  # type: ignore[arg-type]
            community_reports=self.community_reports,  # type: ignore[arg-type]
            community_level=self.community_level,
            dynamic_community_selection=False,
            response_type=self.response_type,
            query=query,
            verbose=False,
        ):
            yield chunk

    async def run_search_streaming(self, method: str, query: str):
        """根据 method 调用 local 或 global 流式搜索。"""
        method = (method or "local").lower()
        if method == "local":
            async for chunk in self.run_local_search_streaming(query):
                yield chunk
        elif method == "global":
            async for chunk in self.run_global_search_streaming(query):
                yield chunk
        else:
            raise ValueError(f"Unsupported method: {method}")


class ChatMessage(BaseModel):
    role: str
    content: Any


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = Field(default=None)
    messages: Optional[list[ChatMessage]] = Field(default=None)
    prompt: Optional[str] = Field(default=None)
    stream: Optional[bool] = Field(default=False)


async def generate_openai_stream(
    service: SearchService,
    method: str,
    query: str,
    model_name: str,
) -> AsyncGenerator[str, None]:
    """生成 OpenAI 兼容的 SSE (Server-Sent Events) 流。

    Args:
        service: SearchService 实例
        method: 搜索方法 ("local" 或 "global")
        query: 查询文本
        model_name: 模型名称

    Yields:
        SSE 格式的字符串数据块
    """
    created = int(time.time())
    chunk_id = f"chatcmpl-{created}"

    try:
        # 逐块返回内容
        async for chunk_text in service.run_search_streaming(method, query):
            chunk_data = {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_name,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": chunk_text},
                        "finish_reason": None,
                    }
                ],
            }
            yield f"data: {json.dumps(chunk_data, ensure_ascii=False)}\n\n"

        # 发送结束标记
        final_chunk = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }
            ],
        }
        yield f"data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    except Exception as exc:  # noqa: BLE001
        # 发送错误信息
        logger.exception("Streaming search failed")
        error_chunk = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": f"\n\n[Error: {exc}]"},
                    "finish_reason": "stop",
                }
            ],
        }
        yield f"data: {json.dumps(error_chunk, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"



@app.post("/v1/chat/completions", response_model=None)
async def chat_completions(req: ChatCompletionRequest) -> dict[str, Any] | StreamingResponse:
    """OpenAI Chat Completions 兼容接口，内部调用 GraphRAG local_search 或 global_search。

    支持流式和非流式返回，通过 stream 参数控制。
    """
    # 解析 query：优先用 messages 中最后一个 user 消息
    query_text: Optional[str] = None
    messages = req.messages or []
    if isinstance(messages, list):
        for msg in reversed(messages):
            if msg.role == "user":
                content = msg.content
                if isinstance(content, str):
                    query_text = content
                elif isinstance(content, list):
                    parts: list[str] = []
                    for part in content:
                        if isinstance(part, dict) and "text" in part:
                            parts.append(str(part["text"]))
                        elif isinstance(part, str):
                            parts.append(part)
                    query_text = "\n".join(parts)
                break

    if not query_text and isinstance(req.prompt, str):
        query_text = req.prompt

    if not query_text:
        raise HTTPException(
            status_code=400,
            detail="Missing user message in 'messages' or 'prompt'.",
        )

    service: SearchService = app.state.service  # type: ignore[assignment]
    method: str = getattr(app.state, "search_method", "local")
    model_name = req.model or "graphrag-local"

    # 流式返回
    if req.stream:
        return StreamingResponse(
            generate_openai_stream(service, method, query_text, model_name),
            media_type="text/event-stream",
        )

    # 非流式返回
    try:
        answer = await service.run_search(method, query_text)
    except Exception as exc:  # noqa: BLE001
        logger.exception("search failed")
        raise HTTPException(
            status_code=500,
            detail=f"search failed: {exc}",
        ) from exc

    created = int(time.time())
    return {
        "id": f"chatcmpl-{created}",
        "object": "chat.completion",
        "created": created,
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": answer,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GraphRAG local_search → OpenAI Chat Completions 兼容服务",
    )
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="GraphRAG 项目根目录（包含 settings.yaml / output/ 等）",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="监听地址（默认 0.0.0.0）",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="监听端口（默认 8000）",
    )
    parser.add_argument(
        "--community-level",
        type=int,
        default=2,
        help="local_search 使用的 community_level（默认 2）",
    )
    parser.add_argument(
        "--response-type",
        type=str,
        default="multiple paragraphs",
        help="GraphRAG local_search 的 response_type（默认 multiple paragraphs）",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["local", "global"],
        default="local",
        help="使用 local 或 global search（默认 local）",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    args = parse_args(argv)

    root_dir = Path(args.root).resolve()
    if not root_dir.exists():
        logger.error("指定的 root 目录不存在: %s", root_dir)
        sys.exit(1)

    global _SERVICE_PARAMS
    _SERVICE_PARAMS = {
        "root_dir": str(root_dir),
        "community_level": args.community_level,
        "response_type": args.response_type,
        "method": args.method,
    }

    logger.info(
        "GraphRAG %s_search OpenAI 兼容服务已启动: http://%s:%d/v1/chat/completions (root=%s)",
        args.method,
        args.host,
        args.port,
        root_dir,
    )

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
