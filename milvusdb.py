"""
自定义 Milvus 向量库实现与注册。

设计目标：
- 不修改 graphrag 源码 / site-packages；
- 完全复用 graphrag 的 BaseVectorStore / VectorStoreFactory；
- 通过 VectorStoreFactory.register(\"milvus\", MilvusVectorStore) 挂入；
- generate_text_embeddings / update_text_embeddings 等工作流无需修改即可使用；
- 连接参数通过环境变量配置（避免改动 graphrag 的 VectorStoreConfig）。

环境变量（可在项目根目录 .env 中配置，run_update 会自动加载）：
- MILVUS_HOST：默认 \"localhost\"
- MILVUS_PORT：默认 \"19530\"
- MILVUS_USER：可选
- MILVUS_PASSWORD：可选
- MILVUS_TOKEN：可选
- MILVUS_DATABASE：可选，多 DB 场景使用
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from graphrag.config.models.vector_store_schema_config import VectorStoreSchemaConfig
from graphrag.data_model.types import TextEmbedder
from graphrag.vector_stores.base import (
    BaseVectorStore,
    VectorStoreDocument,
    VectorStoreSearchResult,
)
from graphrag.vector_stores.factory import VectorStoreFactory

logger = logging.getLogger(__name__)


class MilvusVectorStore(BaseVectorStore):
    """基于 Milvus (pymilvus) 的向量存储实现。"""

    def __init__(
        self, vector_store_schema_config: VectorStoreSchemaConfig, **kwargs: Any
    ) -> None:
        # BaseVectorStore 会保存 schema 信息和 kwargs
        super().__init__(
            vector_store_schema_config=vector_store_schema_config, **kwargs
        )
        # collection_name 默认使用 index_name（由 embed_text 里的 index_name 逻辑生成）
        self.collection_name: str | None = kwargs.get(
            "collection_name", getattr(vector_store_schema_config, "index_name", None)
        )

    # ------------------------------------------------------------------ #
    # 内部工具
    # ------------------------------------------------------------------ #

    def _require_pymilvus(self):
        """惰性导入 pymilvus，并缓存相关类。"""
        try:
            from pymilvus import (  # type: ignore[import-not-found]
                Collection,
                CollectionSchema,
                DataType,
                FieldSchema,
                connections,
                db,
                utility,
            )
        except Exception as exc:  # pragma: no cover - 环境问题
            msg = (
                "使用 Milvus 向量库需要安装 pymilvus。\n"
                "请在当前项目环境中执行：pip install pymilvus\n"
                "或在 pyproject.toml 中加入依赖后重新安装。"
            )
            raise ImportError(msg) from exc

        self._MilvusCollection = Collection
        self._MilvusCollectionSchema = CollectionSchema
        self._MilvusDataType = DataType
        self._MilvusFieldSchema = FieldSchema
        self._milvus_connections = connections
        self._milvus_db = db
        self._milvus_utility = utility

    def _resolve_collection_name(self) -> str:
        """确定要使用的 collection 名称，并转换为 Milvus 支持的格式。

        Milvus 要求：
        - 只能包含数字、字母、下划线；
        - 不能以数字开头。
        GraphRAG 生成的 index_name 里可能包含 `-`、`.` 等，需要转成 `_`。
        """

        def sanitize(name: str) -> str:
            # 替换非法字符为下划线
            sanitized = re.sub(r"[^0-9A-Za-z_]", "_", name)
            if not sanitized:
                sanitized = "collection"
            # 不能以数字开头，必要时加前缀
            if sanitized[0].isdigit():
                sanitized = f"c_{sanitized}"
            return sanitized

        if self.collection_name:
            safe = sanitize(self.collection_name)
            if safe != self.collection_name:
                logger.info(
                    "Sanitized Milvus collection name from '%s' to '%s'",
                    self.collection_name,
                    safe,
                )
            self.collection_name = safe
            return self.collection_name

        if self.index_name:
            raw = self.index_name
            safe = sanitize(raw)
            if safe != raw:
                logger.info(
                    "Sanitized Milvus collection name from index_name '%s' to '%s'",
                    raw,
                    safe,
                )
            self.collection_name = safe
            return self.collection_name

        msg = (
            "Milvus collection 名称无法确定。\n"
            "请在 embeddings_schema 中设置 index_name，或在 MilvusVectorStore 初始化时显式传入 collection_name。"
        )
        raise ValueError(msg)

    # ------------------------------------------------------------------ #
    # BaseVectorStore 接口实现
    # ------------------------------------------------------------------ #

    def connect(self, **kwargs: Any) -> None:
        """连接 Milvus 服务。

        优先顺序：
        - kwargs 中传入的参数（目前 graphrag 不会传 host/port，这里只是预留）；
        - 环境变量 MILVUS_*；
        - 默认 host=localhost, port=19530。
        """
        self._require_pymilvus()

        host = (
            kwargs.get("host")
            or os.getenv("MILVUS_HOST")
            or "localhost"
        )
        port = int(
            kwargs.get("port")
            or os.getenv("MILVUS_PORT", "19530")
        )
        uri = kwargs.get("uri") or kwargs.get("url") or os.getenv("MILVUS_URI")

        user = kwargs.get("user") or os.getenv("MILVUS_USER")
        password = kwargs.get("password") or os.getenv("MILVUS_PASSWORD")
        token = kwargs.get("token") or os.getenv("MILVUS_TOKEN")
        database = kwargs.get("database") or os.getenv("MILVUS_DATABASE")

        self._alias = kwargs.get("alias", "default")

        conn_args: dict[str, Any] = {"alias": self._alias}
        if uri:
            conn_args["uri"] = uri
        else:
            conn_args["host"] = host
            conn_args["port"] = str(port)

        if token:
            conn_args["token"] = token
        elif user and password:
            conn_args["user"] = user
            conn_args["password"] = password

        logger.info(
            "Connecting Milvus (alias=%s) at %s:%s (database=%s)",
            self._alias,
            host,
            port,
            database or "default",
        )
        self._milvus_connections.connect(**conn_args)
        self.db_connection = self._milvus_connections

        if database:
            try:
                self._milvus_db.using_database(database)
            except Exception:  # pragma: no cover - 依赖服务版本
                logger.warning("Milvus 数据库 '%s' 不可用，继续使用默认数据库。", database)

        name = kwargs.get("collection_name") or self.collection_name
        if name:
            self.collection_name = name

        # 如果已经存在 collection，则直接打开，给 overwrite=False 的增量更新用
        resolved = self._resolve_collection_name()
        if self._milvus_utility.has_collection(resolved):
            logger.info("Using existing Milvus collection '%s'", resolved)
            self.document_collection = self._MilvusCollection(resolved)
            try:
                self.document_collection.load()
            except Exception:  # pragma: no cover
                logger.warning("加载 collection '%s' 到内存失败，将在查询时按需加载。", resolved)

    def load_documents(
        self, documents: list[VectorStoreDocument], overwrite: bool = True
    ) -> None:
        """向 Milvus 中写入向量数据。

        语义与 LanceDBVectorStore 保持一致：
        - 第一个 batch 且 overwrite=True 时，删除并重建 collection；
        - 后续 batch 使用 overwrite=False 追加写入。
        """
        if not documents:
            return

        self._require_pymilvus()
        name = self._resolve_collection_name()

        ids: list[str] = []
        texts: list[str] = []
        vectors: list[list[float]] = []
        attributes: list[str] = []

        for doc in documents:
            if doc.vector is None:
                continue

            # 与 LanceDB 实现对齐：根据实际 embedding 长度设置向量维度。
            self.vector_size = len(doc.vector)

            ids.append(str(doc.id))

            # text 字段在部分情况下可能是 list/tuple，Milvus 要求 string，
            # 这里统一转换为字符串，避免 ParamError。
            text = doc.text
            if isinstance(text, (list, tuple)):
                text = " ".join(str(t) for t in text)
            elif text is not None and not isinstance(text, str):
                text = str(text)
            texts.append(text or "")

            vectors.append(list(doc.vector))
            attributes.append(json.dumps(doc.attributes or {}))

        if not vectors:
            return

        # 覆盖模式：存在则删除
        if overwrite and self._milvus_utility.has_collection(name):
            logger.info("Dropping existing Milvus collection '%s' (overwrite=True)", name)
            self._milvus_utility.drop_collection(name)
            self.document_collection = None

        # 如无 collection，则创建
        if self.document_collection is None:
            if not self._milvus_utility.has_collection(name):
                logger.info("Creating Milvus collection '%s'", name)
                fields = [
                    self._MilvusFieldSchema(
                        name=self.id_field,
                        dtype=self._MilvusDataType.VARCHAR,
                        max_length=512,
                        is_primary=True,
                    ),
                    self._MilvusFieldSchema(
                        name=self.text_field,
                        dtype=self._MilvusDataType.VARCHAR,
                        max_length=65535,
                    ),
                    self._MilvusFieldSchema(
                        name=self.vector_field,
                        dtype=self._MilvusDataType.FLOAT_VECTOR,
                        dim=self.vector_size,
                    ),
                    self._MilvusFieldSchema(
                        name=self.attributes_field,
                        dtype=self._MilvusDataType.VARCHAR,
                        max_length=65535,
                    ),
                ]
                schema = self._MilvusCollectionSchema(
                    fields=fields,
                    description="GraphRAG text embeddings",
                )
                self.document_collection = self._MilvusCollection(
                    name=name,
                    schema=schema,
                )

                # 默认索引，可通过环境变量或 kwargs 自行扩展，这里保持简单
                index_params = {
                    "index_type": "IVF_FLAT",
                    "metric_type": "IP",
                    "params": {"nlist": 1024},
                }
                try:
                    self.document_collection.create_index(
                        field_name=self.vector_field,
                        index_params=index_params,
                    )
                except Exception:  # pragma: no cover
                    logger.warning(
                        "为 collection '%s' 创建索引失败，后续搜索可能退化为全表扫描。", name
                    )
            else:
                self.document_collection = self._MilvusCollection(name)

        try:
            self.document_collection.load()
        except Exception:  # pragma: no cover
            logger.warning("在插入前加载 collection '%s' 失败，将继续插入。", name)

        # 注意：entities 的顺序必须与 schema 中字段顺序一致：
        # [id_field (str), text_field (str), vector_field (list[float]), attributes_field (str)]
        entities = [ids, texts, vectors, attributes]
        self.document_collection.insert(entities)

    def filter_by_id(self, include_ids: list[str] | list[int]) -> Any:
        """构建 Milvus 的 expr 过滤表达式。"""
        if not include_ids:
            self.query_filter = None
            return self.query_filter

        if isinstance(include_ids[0], str):
            id_list = ", ".join(f'"{id_}"' for id_ in include_ids)
        else:
            id_list = ", ".join(str(id_) for id_ in include_ids)

        self.query_filter = f"{self.id_field} in [{id_list}]"
        return self.query_filter

    def similarity_search_by_vector(
        self, query_embedding: list[float], k: int = 10, **kwargs: Any
    ) -> list[VectorStoreSearchResult]:
        """基于向量的相似度检索。"""
        if self.document_collection is None:
            name = self._resolve_collection_name()
            if self._milvus_utility.has_collection(name):
                self.document_collection = self._MilvusCollection(name)
            else:
                return []

        base_params = {
            "metric_type": "IP",
            "params": {"nprobe": 10},
        }
        search_params = kwargs.get("search_params") or base_params
        expr = kwargs.get("expr") or self.query_filter

        try:
            results = self.document_collection.search(
                data=[query_embedding],
                anns_field=self.vector_field,
                param=search_params,
                limit=k,
                expr=expr,
                output_fields=[self.id_field, self.text_field, self.attributes_field],
            )
        except Exception as exc:  # pragma: no cover
            logger.error("Milvus 搜索失败: %s", exc)
            return []

        if not results:
            return []

        hits = results[0]
        output: list[VectorStoreSearchResult] = []
        for hit in hits:
            entity = getattr(hit, "entity", hit)
            raw_attrs = entity.get(self.attributes_field, "{}")
            try:
                attrs = (
                    json.loads(raw_attrs)
                    if isinstance(raw_attrs, str)
                    else (raw_attrs or {})
                )
            except Exception:
                attrs = {}

            score = getattr(hit, "score", None)
            if score is None:
                score = getattr(hit, "distance", 0.0)

            output.append(
                VectorStoreSearchResult(
                    document=VectorStoreDocument(
                        id=entity.get(self.id_field, ""),
                        text=entity.get(self.text_field, ""),
                        vector=None,  # GraphRAG 查询不依赖返回向量本身
                        attributes=attrs,
                    ),
                    score=float(score),
                )
            )
        return output

    def similarity_search_by_text(
        self, text: str, text_embedder: TextEmbedder, k: int = 10, **kwargs: Any
    ) -> list[VectorStoreSearchResult]:
        """基于文本的相似度检索。"""
        query_embedding = text_embedder(text)
        if query_embedding:
            return self.similarity_search_by_vector(
                query_embedding=query_embedding,
                k=k,
                **kwargs,
            )
        return []

    def search_by_id(self, id: str) -> VectorStoreDocument:
        """按 ID 查询单条文档。"""
        if self.document_collection is None:
            name = self._resolve_collection_name()
            if self._milvus_utility.has_collection(name):
                self.document_collection = self._MilvusCollection(name)
            else:
                return VectorStoreDocument(id=id, text=None, vector=None)

        if isinstance(id, str):
            expr = f'{self.id_field} == "{id}"'
        else:
            expr = f"{self.id_field} == {id}"

        try:
            results = self.document_collection.query(
                expr=expr,
                output_fields=[self.id_field, self.text_field, self.attributes_field],
                limit=1,
            )
        except Exception as exc:  # pragma: no cover
            logger.error("Milvus 按 ID 查询失败: %s", exc)
            return VectorStoreDocument(id=id, text=None, vector=None)

        if not results:
            return VectorStoreDocument(id=id, text=None, vector=None)

        item = results[0]
        raw_attrs = item.get(self.attributes_field, "{}")
        try:
            attrs = (
                json.loads(raw_attrs)
                if isinstance(raw_attrs, str)
                else (raw_attrs or {})
            )
        except Exception:
            attrs = {}

        return VectorStoreDocument(
            id=item.get(self.id_field, id),
            text=item.get(self.text_field, None),
            vector=None,
            attributes=attrs,
        )


def register_milvus_vector_store() -> None:
    """注册自定义 Milvus 向量库到 VectorStoreFactory。

    使用方式：
        from milvusdb import register_milvus_vector_store
        register_milvus_vector_store()

    然后在 settings.yaml 中：
        vector_store:
          default_vector_store:
            type: milvus
            container_name: default

    连接信息通过 .env / 环境变量配置：
        MILVUS_HOST, MILVUS_PORT, ...
    """
    # 只要注册一次即可，多次调用也安全（后者覆盖前者）
    VectorStoreFactory.register("milvus", MilvusVectorStore)
    logger.info("✓ Custom vector store 'milvus' registered with VectorStoreFactory")
