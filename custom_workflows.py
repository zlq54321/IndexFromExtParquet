"""
自定义工作流：加载外部准备的 text_units.parquet

用于跳过 create_base_text_units 步骤

存储位置说明：
- input_storage: 配置的 input.storage.base_dir，放你的外部 parquet 文件
- output_storage:
  * 标准模式: output.base_dir
  * Update模式: update_output/{timestamp}/delta/
"""

import logging
import pandas as pd
from io import BytesIO

from graphrag.config.models.graph_rag_config import GraphRagConfig
from graphrag.index.typing.context import PipelineRunContext
from graphrag.index.typing.workflow import WorkflowFunctionOutput
from graphrag.utils.storage import load_table_from_storage, write_table_to_storage

logger = logging.getLogger(__name__)


async def load_external_text_units(
    config: GraphRagConfig,
    context: PipelineRunContext,
) -> WorkflowFunctionOutput:
    """
    从外部存储加载已经准备好的 text_units.parquet

    此工作流用于替代 create_base_text_units，当你已经有外部切分好的文本单元时使用。

    存储读写位置：
    - 读取 documents.parquet: context.output_storage
      (由 load_update_documents 工作流写入的新文档，在 Update 模式下是 delta/ 目录)
    - 读取 text_units.parquet: context.input_storage
      (你的外部数据，配置的 input.storage.base_dir)
    - 写入 text_units.parquet: context.output_storage
      (在 Update 模式下是 delta/ 目录)

    前置条件：
    - input_storage 中必须有 text_units.parquet 文件
    - 文件必须包含列：id, text, document_ids, n_tokens

    工作流程：
    1. 从 output_storage 读取新文档列表（由上一个工作流写入）
    2. 从 input_storage 加载外部的 text_units.parquet（你的外部数据）
    3. 过滤：只保留属于新文档的 text_units
    4. 写入到 output_storage（Update 模式下是 delta/ 目录）
    """
    logger.info("=" * 60)
    logger.info("Workflow started: load_external_text_units")
    logger.info("=" * 60)

    # 1. 从 output_storage 加载新文档列表
    # 在 Update 模式下，这里读取的是 delta/documents.parquet（只有新文档）
    documents = await load_table_from_storage("documents", context.output_storage)
    new_doc_ids = set(documents["id"].tolist())

    logger.info(f"从 output_storage 加载了 {len(new_doc_ids)} 个文档")
    logger.info(f"文档 IDs: {list(new_doc_ids)[:5]}...")  # 只显示前5个

    # 2. 从 input_storage 加载外部的 text_units.parquet
    # 这是你配置的 input.storage.base_dir 目录
    try:
        text_units_bytes = await context.input_storage.get("text_units.parquet", as_bytes=True)
        text_units = pd.read_parquet(BytesIO(text_units_bytes))
        logger.info(f"从 input_storage 加载了 {len(text_units)} 个文本单元")
    except Exception as e:
        logger.error(f"无法加载 text_units.parquet: {e}")
        raise ValueError(
            "text_units.parquet 未找到！\n"
            "请确保文件位于配置的 input.storage.base_dir 目录下。\n"
            f"当前 input_storage 路径信息: {context.input_storage}"
        )

    # 3. 验证必需的列
    required_columns = ["id", "text", "document_ids", "n_tokens"]
    missing_columns = [col for col in required_columns if col not in text_units.columns]
    if missing_columns:
        raise ValueError(f"text_units.parquet 缺少必需的列: {missing_columns}")

    # 4. 过滤：只保留属于新文档的 text_units
    def belongs_to_new_docs(doc_ids):
        """检查文档ID是否属于新文档集合"""
        if doc_ids is None:
            return False

        # 递归展开嵌套的列表/数组，提取所有字符串
        def flatten(item):
            """递归展开嵌套结构"""
            import numpy as np

            if isinstance(item, str):
                return [item]
            elif isinstance(item, (list, tuple)):
                result = []
                for sub in item:
                    result.extend(flatten(sub))
                return result
            elif isinstance(item, np.ndarray):
                result = []
                for sub in item:
                    result.extend(flatten(sub))
                return result
            else:
                # 其他类型（int, float等），转为字符串
                return [str(item)]

        # 展开并检查
        flat_doc_ids = flatten(doc_ids)
        return any(doc_id in new_doc_ids for doc_id in flat_doc_ids)


    filtered_text_units = text_units[
        text_units["document_ids"].apply(belongs_to_new_docs)
    ].reset_index(drop=True)

    logger.info(f"过滤后剩余 {len(filtered_text_units)} 个文本单元（属于新文档）")

    if len(filtered_text_units) == 0:
        logger.warning("警告：没有找到属于新文档的文本单元！")
        logger.warning(f"新文档 IDs: {new_doc_ids}")
        logger.warning(f"text_units 中的 document_ids 示例: {text_units['document_ids'].head().tolist()}")

    # 5. 写入到 output_storage
    # 在 Update 模式下，这里写入的是 delta/text_units.parquet
    await write_table_to_storage(
        filtered_text_units,
        "text_units",
        context.output_storage
    )

    logger.info(f"已写入 {len(filtered_text_units)} 个文本单元到 output_storage")
    logger.info("Workflow completed: load_external_text_units")
    logger.info("=" * 60)

    return WorkflowFunctionOutput(result=filtered_text_units)


def register_custom_workflows():
    """
    注册自定义工作流到 PipelineFactory

    使用方法：
    ```python
    from custom_workflows import register_custom_workflows
    register_custom_workflows()

    # 然后在 settings.yaml 中使用：
    # workflows:
    #   - ensure_initial_output      # 首次运行必需，后续可注释
    #   - load_update_documents
    #   - load_external_text_units  # 替代 create_base_text_units
    #   - create_final_documents
    #   - ...
    #   - create_graph_store  # 可选：导出到 Neo4j
    ```
    """
    from graphrag.index.workflows.factory import PipelineFactory

    # 注册初始化工作流
    PipelineFactory.register("ensure_initial_output", ensure_initial_output)
    logger.info("✓ Custom workflow 'ensure_initial_output' registered")

    # 注册外部 text_units 加载工作流
    PipelineFactory.register("load_external_text_units", load_external_text_units)
    logger.info("✓ Custom workflow 'load_external_text_units' registered")

    # 注册所有修复版的 update 工作流（修复空 DataFrame 问题）
    PipelineFactory.register("custom_update_final_documents", custom_update_final_documents)
    logger.info("✓ Custom workflow 'custom_update_final_documents' registered (修复空表)")

    PipelineFactory.register("custom_update_text_units", custom_update_text_units)
    logger.info("✓ Custom workflow 'custom_update_text_units' registered (修复空表)")

    PipelineFactory.register("custom_update_entities_relationships", custom_update_entities_relationships)
    logger.info("✓ Custom workflow 'custom_update_entities_relationships' registered (修复空表)")

    PipelineFactory.register("custom_update_communities", custom_update_communities)
    logger.info("✓ Custom workflow 'custom_update_communities' registered (修复空表)")

    PipelineFactory.register("custom_update_community_reports", custom_update_community_reports)
    logger.info("✓ Custom workflow 'custom_update_community_reports' registered (修复空表)")

    # 注册 Neo4j 导出工作流（如果有 neo4j 包）
    PipelineFactory.register("create_graph_store", create_graph_store)
    if HAS_NEO4J:
        logger.info("✓ Custom workflow 'create_graph_store' registered (neo4j available)")
    else:
        logger.info("✓ Custom workflow 'create_graph_store' registered (neo4j not installed, will fail if used)")


async def ensure_initial_output(
    config: GraphRagConfig,
    context: PipelineRunContext,
) -> WorkflowFunctionOutput:
    """
    确保 output 目录有初始空表，以便第一次 update 运行时不会报错。

    此工作流可作为 update 流程的第一个工作流，会检查 output 目录：
    - 如果已有索引文件，则跳过
    - 如果为空，则创建所有必需的空表

    使用方法（在 settings.yaml 中）:
    ```yaml
    workflows:
      - ensure_initial_output  # 第一次运行时需要，后续可注释掉
      - load_update_documents
      - load_external_text_units
      - ...
    ```
    """
    from pathlib import Path

    logger.info("=" * 60)
    logger.info("Workflow started: ensure_initial_output")
    logger.info("=" * 60)

    # 从配置获取 update_output/{timestamp}/previous/ 目录 
    previous_dir = context.previous_storage._root_dir
    previous_path = Path(previous_dir)
    previous_path.mkdir(parents=True, exist_ok=True)

    output_dir = config.output.base_dir
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 检查是否已有索引
    if (previous_path / "documents.parquet").exists():
        logger.info(f"✓ {previous_path} 已有索引，跳过初始化")
        logger.info("Workflow completed: ensure_initial_output (skipped)")
        logger.info("=" * 60)
        return WorkflowFunctionOutput(result=None)

    logger.info(f"首次运行 update 流程，初始化空表到 {previous_path}")

    # 创建所有需要的空表
    empty_tables = {
        "documents": {
            "id": pd.Series([], dtype=str),
            "human_readable_id": pd.Series([], dtype=int),
            "title": pd.Series([], dtype=str),
            "text_unit_ids": pd.Series([], dtype=object),
        },
        "text_units": {
            "id": pd.Series([], dtype=str),
            "human_readable_id": pd.Series([], dtype=int),
            "text": pd.Series([], dtype=str),
            "n_tokens": pd.Series([], dtype=int),
            "document_ids": pd.Series([], dtype=object),
            "entity_ids": pd.Series([], dtype=object),
            "relationship_ids": pd.Series([], dtype=object),
            "covariate_ids": pd.Series([], dtype=object),
        },
        "entities": {
            "id": pd.Series([], dtype=str),
            "human_readable_id": pd.Series([], dtype=int),
            "title": pd.Series([], dtype=str),
            "type": pd.Series([], dtype=str),
            "description": pd.Series([], dtype=str),
            "text_unit_ids": pd.Series([], dtype=object),
        },
        "relationships": {
            "id": pd.Series([], dtype=str),
            "human_readable_id": pd.Series([], dtype=int),
            "source": pd.Series([], dtype=str),
            "target": pd.Series([], dtype=str),
            "description": pd.Series([], dtype=str),
            "text_unit_ids": pd.Series([], dtype=object),
            "weight": pd.Series([], dtype=float),
        },
        "communities": {
            "id": pd.Series([], dtype=str),
            "human_readable_id": pd.Series([], dtype=int),
            "community": pd.Series([], dtype=int),
            "level": pd.Series([], dtype=int),
            "title": pd.Series([], dtype=str),
        },
        "community_reports": {
            "id": pd.Series([], dtype=str),
            "human_readable_id": pd.Series([], dtype=int),
            "community": pd.Series([], dtype=int),
            "level": pd.Series([], dtype=int),
            "title": pd.Series([], dtype=str),
            "summary": pd.Series([], dtype=str),
            "full_content": pd.Series([], dtype=str),
        },
    }

    for table_name, columns in empty_tables.items():
        df = pd.DataFrame(columns)
        file_path = previous_path / f"{table_name}.parquet"
        df.to_parquet(file_path, index=False)
        logger.info(f"  ✓ 创建空表: {file_path}")
        file_path = output_path / f"{table_name}.parquet"
        df.to_parquet(file_path, index=False)
        logger.info(f"  ✓ 创建空表: {file_path}")

    logger.info(f"✓ 初始化完成，共创建 {len(empty_tables)} 个空表")
    logger.info("Workflow completed: ensure_initial_output")
    logger.info("=" * 60)

    return WorkflowFunctionOutput(result=None)


# ============================================================================
# Custom Update Workflows (修复空 DataFrame 的 human_readable_id 问题)
# ============================================================================

async def custom_concat_dataframes(
    name: str,
    previous_storage,
    delta_storage,
    output_storage,
) -> pd.DataFrame:
    """
    合并 previous 和 delta 数据框，修复空 DataFrame 时 human_readable_id.max() 返回 nan 的问题。

    这是对 graphrag.index.update.incremental_index.concat_dataframes 的修复版本。

    原始问题：
    - 空 DataFrame 的 max() 返回 nan (float)
    - np.arange(nan, nan + len) 报错: "cannot compute length"

    修复方案：
    - 检查 old_df 是否为空或 max() 返回 nan
    - 如果为空，从 0 开始分配 human_readable_id
    """
    import numpy as np
    from graphrag.utils.storage import load_table_from_storage, write_table_to_storage

    old_df = await load_table_from_storage(name, previous_storage)
    delta_df = await load_table_from_storage(name, delta_storage)

    # 修复：处理空 DataFrame 的情况
    if len(old_df) == 0 or pd.isna(old_df["human_readable_id"].max()):
        # previous 为空，从 0 开始
        initial_id = 0
        logger.info(f"[{name}] previous 为空，human_readable_id 从 0 开始")
    else:
        # previous 有数据，从 max + 1 开始
        initial_id = int(old_df["human_readable_id"].max()) + 1
        logger.info(f"[{name}] previous 有 {len(old_df)} 行，human_readable_id 从 {initial_id} 开始")

    # 为 delta 分配 human_readable_id
    delta_df["human_readable_id"] = np.arange(initial_id, initial_id + len(delta_df))

    # 合并
    final_df = pd.concat([old_df, delta_df], ignore_index=True, copy=False)
    logger.info(f"[{name}] 合并完成: previous={len(old_df)}, delta={len(delta_df)}, final={len(final_df)}")

    await write_table_to_storage(final_df, name, output_storage)

    return final_df


async def custom_update_final_documents(
    config: GraphRagConfig,
    context: PipelineRunContext,
) -> WorkflowFunctionOutput:
    """
    更新最终文档表（修复版本）。

    此工作流替代 graphrag 内置的 update_final_documents，
    修复了当 previous/documents.parquet 为空时的 human_readable_id 分配问题。

    用法：在 settings.yaml 中使用 custom_update_final_documents 替代 update_final_documents
    """
    from graphrag.index.run.utils import get_update_storages

    logger.info("=" * 60)
    logger.info("Workflow started: custom_update_final_documents")
    logger.info("=" * 60)

    output_storage, previous_storage, delta_storage = get_update_storages(
        config, context.state["update_timestamp"]
    )

    final_documents = await custom_concat_dataframes(
        "documents", previous_storage, delta_storage, output_storage
    )

    context.state["incremental_update_final_documents"] = final_documents

    logger.info("Workflow completed: custom_update_final_documents")
    logger.info("=" * 60)
    return WorkflowFunctionOutput(result=None)


async def custom_update_text_units(
    config: GraphRagConfig,
    context: PipelineRunContext,
) -> WorkflowFunctionOutput:
    """
    更新文本单元表（修复版本）。

    修复了当 previous/text_units.parquet 为空时的 human_readable_id 分配问题。
    """
    from graphrag.index.run.utils import get_update_storages
    from graphrag.utils.storage import load_table_from_storage, write_table_to_storage
    import numpy as np

    logger.info("=" * 60)
    logger.info("Workflow started: custom_update_text_units")
    logger.info("=" * 60)

    output_storage, previous_storage, delta_storage = get_update_storages(
        config, context.state["update_timestamp"]
    )
    entity_id_mapping = context.state["incremental_update_entity_id_mapping"]

    # 加载数据
    old_text_units = await load_table_from_storage("text_units", previous_storage)
    delta_text_units = await load_table_from_storage("text_units", delta_storage)

    # 更新 entity_ids（使用映射）
    if entity_id_mapping:
        delta_text_units["entity_ids"] = delta_text_units["entity_ids"].apply(
            lambda x: [entity_id_mapping.get(i, i) for i in x] if x is not None else x
        )

    # 修复：处理空 DataFrame 的情况
    if len(old_text_units) == 0 or pd.isna(old_text_units["human_readable_id"].max()):
        initial_id = 0
        logger.info(f"[text_units] previous 为空，human_readable_id 从 0 开始")
    else:
        initial_id = int(old_text_units["human_readable_id"].max()) + 1
        logger.info(f"[text_units] previous 有 {len(old_text_units)} 行，human_readable_id 从 {initial_id} 开始")

    delta_text_units["human_readable_id"] = np.arange(
        initial_id, initial_id + len(delta_text_units)
    )

    # 合并
    merged_text_units = pd.concat([old_text_units, delta_text_units], ignore_index=True, copy=False)
    logger.info(f"[text_units] 合并完成: previous={len(old_text_units)}, delta={len(delta_text_units)}, final={len(merged_text_units)}")

    await write_table_to_storage(merged_text_units, "text_units", output_storage)

    context.state["incremental_update_merged_text_units"] = merged_text_units

    logger.info("Workflow completed: custom_update_text_units")
    logger.info("=" * 60)
    return WorkflowFunctionOutput(result=None)


async def custom_update_entities_relationships(
    config: GraphRagConfig,
    context: PipelineRunContext,
) -> WorkflowFunctionOutput:
    """
    更新实体和关系表（修复版本）。

    修复了当 previous 表为空时的 human_readable_id 分配问题。
    """
    from graphrag.index.run.utils import get_update_storages
    from graphrag.utils.storage import load_table_from_storage, write_table_to_storage
    from graphrag.index.workflows.extract_graph import get_summarized_entities_relationships
    import numpy as np
    import itertools
    from graphrag.data_model.schemas import ENTITIES_FINAL_COLUMNS, RELATIONSHIPS_FINAL_COLUMNS

    logger.info("=" * 60)
    logger.info("Workflow started: custom_update_entities_relationships")
    logger.info("=" * 60)

    output_storage, previous_storage, delta_storage = get_update_storages(
        config, context.state["update_timestamp"]
    )

    # === 处理 Entities ===
    old_entities = await load_table_from_storage("entities", previous_storage)
    delta_entities = await load_table_from_storage("entities", delta_storage)

    # 创建 ID 映射（title 相同的实体）
    merged = delta_entities[["id", "title"]].merge(
        old_entities[["id", "title"]],
        on="title",
        suffixes=("_B", "_A"),
        copy=False,
    )
    entity_id_mapping = dict(zip(merged["id_B"], merged["id_A"], strict=True))

    # 修复：处理空 DataFrame
    if len(old_entities) == 0 or pd.isna(old_entities["human_readable_id"].max()):
        initial_id = 0
        logger.info(f"[entities] previous 为空，human_readable_id 从 0 开始")
    else:
        initial_id = int(old_entities["human_readable_id"].max()) + 1
        logger.info(f"[entities] previous 有 {len(old_entities)} 行，human_readable_id 从 {initial_id} 开始")

    delta_entities["human_readable_id"] = np.arange(
        initial_id, initial_id + len(delta_entities)
    )

    # 合并和聚合
    combined = pd.concat([old_entities, delta_entities], ignore_index=True, copy=False)

    aggregated = (
        combined.groupby("title")
        .agg({
            "id": "first",
            "type": "first",
            "human_readable_id": "first",
            "description": lambda x: list(x.astype(str)),
            "text_unit_ids": lambda x: list(itertools.chain(*x.tolist())),
            "degree": "first",
            "x": "first",
            "y": "first",
        })
        .reset_index()
    )

    aggregated["frequency"] = aggregated["text_unit_ids"].apply(len)
    merged_entities_df = pd.DataFrame(aggregated)
    merged_entities_df = merged_entities_df.loc[:, ENTITIES_FINAL_COLUMNS]

    logger.info(f"[entities] 合并完成: previous={len(old_entities)}, delta={len(delta_entities)}, final={len(merged_entities_df)}")

    # === 处理 Relationships ===
    old_relationships = await load_table_from_storage("relationships", previous_storage)
    delta_relationships = await load_table_from_storage("relationships", delta_storage)

    # 确保类型正确
    delta_relationships["human_readable_id"] = delta_relationships["human_readable_id"].astype(int)
    old_relationships["human_readable_id"] = old_relationships["human_readable_id"].astype(int)

    # 修复：处理空 DataFrame
    if len(old_relationships) == 0 or pd.isna(old_relationships["human_readable_id"].max()):
        initial_id = 0
        logger.info(f"[relationships] previous 为空，human_readable_id 从 0 开始")
    else:
        initial_id = int(old_relationships["human_readable_id"].max()) + 1
        logger.info(f"[relationships] previous 有 {len(old_relationships)} 行，human_readable_id 从 {initial_id} 开始")

    delta_relationships["human_readable_id"] = np.arange(
        initial_id, initial_id + len(delta_relationships)
    )

    # 合并
    merged_relationships = pd.concat(
        [old_relationships, delta_relationships], ignore_index=True, copy=False
    )

    # 聚合（按 source + target）
    aggregated = (
        merged_relationships.groupby(["source", "target"])
        .agg({
            "id": "first",
            "human_readable_id": "first",
            "description": lambda x: list(x.astype(str)),
            "text_unit_ids": lambda x: list(itertools.chain(*x.tolist())),
            "weight": "mean",
            "combined_degree": "sum",
        })
        .reset_index()
    )

    merged_relationships_df = pd.DataFrame(aggregated)

    # 重新计算 degree
    merged_relationships_df["source_degree"] = merged_relationships_df.groupby("source")["target"].transform("count")
    merged_relationships_df["target_degree"] = merged_relationships_df.groupby("target")["source"].transform("count")
    merged_relationships_df["combined_degree"] = (
        merged_relationships_df["source_degree"] + merged_relationships_df["target_degree"]
    )

    merged_relationships_df = merged_relationships_df.loc[:, RELATIONSHIPS_FINAL_COLUMNS]

    logger.info(f"[relationships] 合并完成: previous={len(old_relationships)}, delta={len(delta_relationships)}, final={len(merged_relationships_df)}")

    # === LLM 总结（描述去重） ===
    summarization_llm_settings = config.get_language_model_config(
        config.summarize_descriptions.model_id
    )
    summarization_strategy = config.summarize_descriptions.resolved_strategy(
        config.root_dir, summarization_llm_settings
    )

    (
        merged_entities_df,
        merged_relationships_df,
    ) = await get_summarized_entities_relationships(
        extracted_entities=merged_entities_df,
        extracted_relationships=merged_relationships_df,
        callbacks=context.callbacks,
        cache=context.cache,
        summarization_strategy=summarization_strategy,
        summarization_num_threads=summarization_llm_settings.concurrent_requests,
    )

    # 保存
    await write_table_to_storage(merged_entities_df, "entities", output_storage)
    await write_table_to_storage(merged_relationships_df, "relationships", output_storage)

    context.state["incremental_update_merged_entities"] = merged_entities_df
    context.state["incremental_update_merged_relationships"] = merged_relationships_df
    context.state["incremental_update_entity_id_mapping"] = entity_id_mapping

    logger.info("Workflow completed: custom_update_entities_relationships")
    logger.info("=" * 60)
    return WorkflowFunctionOutput(result=None)


async def custom_update_communities(
    config: GraphRagConfig,
    context: PipelineRunContext,
) -> WorkflowFunctionOutput:
    """
    更新社区表（修复版本）。

    修复了当 previous/communities.parquet 为空时的 community ID 分配问题。
    """
    from graphrag.index.run.utils import get_update_storages
    from graphrag.utils.storage import load_table_from_storage, write_table_to_storage
    from graphrag.data_model.schemas import COMMUNITIES_FINAL_COLUMNS

    logger.info("=" * 60)
    logger.info("Workflow started: custom_update_communities")
    logger.info("=" * 60)

    output_storage, previous_storage, delta_storage = get_update_storages(
        config, context.state["update_timestamp"]
    )

    old_communities = await load_table_from_storage("communities", previous_storage)
    delta_communities = await load_table_from_storage("communities", delta_storage)

    # 确保有 size 和 period 列
    if "size" not in old_communities.columns:
        old_communities["size"] = None
    if "period" not in old_communities.columns:
        old_communities["period"] = None
    if "size" not in delta_communities.columns:
        delta_communities["size"] = None
    if "period" not in delta_communities.columns:
        delta_communities["period"] = None

    # 修复：处理空 DataFrame（fillna 对空 Series 无效）
    if len(old_communities) == 0:
        old_max_community_id = -1
        logger.info(f"[communities] previous 为空，community ID 从 0 开始")
    else:
        old_max_community_id = old_communities["community"].fillna(0).astype(int).max()
        logger.info(f"[communities] previous 最大 community ID: {old_max_community_id}")

    # 创建 community ID 映射
    community_id_mapping = {
        v: v + old_max_community_id + 1
        for k, v in delta_communities["community"].dropna().astype(int).items()
    }
    community_id_mapping.update({-1: -1})

    # 更新 delta 的 community IDs
    delta_communities["community"] = (
        delta_communities["community"]
        .astype(int)
        .apply(lambda x: community_id_mapping.get(x, x))
    )

    delta_communities["parent"] = (
        delta_communities["parent"]
        .astype(int)
        .apply(lambda x: community_id_mapping.get(x, x))
    )

    old_communities["community"] = old_communities["community"].astype(int)

    # 合并
    merged_communities = pd.concat(
        [old_communities, delta_communities], ignore_index=True, copy=False
    )

    # 重命名 title 和 human_readable_id
    merged_communities["title"] = "Community " + merged_communities["community"].astype(str)
    merged_communities["human_readable_id"] = merged_communities["community"]

    merged_communities = merged_communities.loc[:, COMMUNITIES_FINAL_COLUMNS]

    logger.info(f"[communities] 合并完成: previous={len(old_communities)}, delta={len(delta_communities)}, final={len(merged_communities)}")

    await write_table_to_storage(merged_communities, "communities", output_storage)

    context.state["incremental_update_community_id_mapping"] = community_id_mapping

    logger.info("Workflow completed: custom_update_communities")
    logger.info("=" * 60)
    return WorkflowFunctionOutput(result=None)


async def custom_update_community_reports(
    config: GraphRagConfig,
    context: PipelineRunContext,
) -> WorkflowFunctionOutput:
    """
    更新社区报告表（修复版本）。

    依赖 custom_update_communities 生成的 community_id_mapping。
    """
    from graphrag.index.run.utils import get_update_storages
    from graphrag.utils.storage import load_table_from_storage, write_table_to_storage
    from graphrag.data_model.schemas import COMMUNITY_REPORTS_FINAL_COLUMNS

    logger.info("=" * 60)
    logger.info("Workflow started: custom_update_community_reports")
    logger.info("=" * 60)

    output_storage, previous_storage, delta_storage = get_update_storages(
        config, context.state["update_timestamp"]
    )

    community_id_mapping = context.state["incremental_update_community_id_mapping"]

    old_community_reports = await load_table_from_storage("community_reports", previous_storage)
    delta_community_reports = await load_table_from_storage("community_reports", delta_storage)

    # 确保有 size 和 period 列
    if "size" not in old_community_reports.columns:
        old_community_reports["size"] = None
    if "period" not in old_community_reports.columns:
        old_community_reports["period"] = None
    if "size" not in delta_community_reports.columns:
        delta_community_reports["size"] = None
    if "period" not in delta_community_reports.columns:
        delta_community_reports["period"] = None

    # 更新 delta 的 community IDs
    delta_community_reports["community"] = (
        delta_community_reports["community"]
        .astype(int)
        .apply(lambda x: community_id_mapping.get(x, x))
    )

    delta_community_reports["parent"] = (
        delta_community_reports["parent"]
        .astype(int)
        .apply(lambda x: community_id_mapping.get(x, x))
    )

    old_community_reports["community"] = old_community_reports["community"].astype(int)

    # 合并
    merged_community_reports = pd.concat(
        [old_community_reports, delta_community_reports], ignore_index=True, copy=False
    )

    # 维护类型兼容性
    merged_community_reports["community"] = merged_community_reports["community"].astype(int)
    merged_community_reports["human_readable_id"] = merged_community_reports["community"]

    merged_community_reports = merged_community_reports.loc[:, COMMUNITY_REPORTS_FINAL_COLUMNS]

    logger.info(f"[community_reports] 合并完成: previous={len(old_community_reports)}, delta={len(delta_community_reports)}, final={len(merged_community_reports)}")

    await write_table_to_storage(merged_community_reports, "community_reports", output_storage)

    # 重要：设置 context.state 供后续 update_text_embeddings 使用
    context.state["incremental_update_merged_community_reports"] = merged_community_reports

    logger.info("Workflow completed: custom_update_community_reports")
    logger.info("=" * 60)
    return WorkflowFunctionOutput(result=None)


# ============================================================================
# Neo4j Graph Store Workflow
# ============================================================================

try:
    from neo4j import GraphDatabase  # type: ignore[import-not-found]
    import concurrent.futures
    import json
    import time
    from typing import Any

    HAS_NEO4J = True
except ImportError:
    HAS_NEO4J = False
    GraphDatabase = None  # type: ignore[assignment,misc]


async def create_graph_store(
    config: GraphRagConfig,
    context: PipelineRunContext,
) -> WorkflowFunctionOutput:
    """
    导出 GraphRAG 索引到 Neo4j 图数据库。

    此工作流会将以下数据导入 Neo4j：
    - Document 节点
    - Chunk 节点及其与 Document 的关系
    - Entity 节点及其与 Chunk 的关系（支持动态类型标签）
    - Relationship 节点及实体间的 RELATED 关系
    - Community 节点及其与实体/关系的关联
    - Community Report 和 Finding 节点

    配置示例（settings.yaml）：
    ```yaml
    graph_store:
      uri: "bolt://localhost:7687"
      username: "neo4j"
      password: "your-password"
      database: "neo4j"
      clear_database: false  # 是否清空现有数据
    ```

    前置条件：
    - 安装 neo4j 包：pip install neo4j
    - Neo4j 服务运行中
    - output_storage 中有完整索引数据
    """
    logger.info("=" * 60)
    logger.info("Workflow started: create_graph_store")
    logger.info("=" * 60)

    # 方案：从 settings.yaml 直接读取 graph_store 配置
    graph_store_config = None
    try:
        import yaml
        import os
        from pathlib import Path

        # 从 config.root_dir 找到 settings.yaml
        settings_file = Path(config.root_dir) / "settings.yaml"

        if settings_file.exists():
            with open(settings_file, "r", encoding="utf-8") as f:
                yaml_content = f.read()

            # 替换环境变量
            import re
            def replace_env_vars(match):
                var_name = match.group(1)
                return os.environ.get(var_name, match.group(0))

            yaml_content = re.sub(r'\$\{(\w+)\}', replace_env_vars, yaml_content)
            config_data = yaml.safe_load(yaml_content)
            graph_store_config = config_data.get("graph_store")

    except Exception as e:
        logger.warning(f"无法读取 graph_store 配置: {e}")

    # 检查是否配置了 graph_store
    if graph_store_config is None:
        logger.warning("⚠️  未配置 graph_store，跳过 Neo4j 导出")
        logger.info("提示: 在 settings.yaml 中添加 graph_store 配置以启用此功能")
        logger.info("Workflow completed: create_graph_store (skipped)")
        logger.info("=" * 60)
        return WorkflowFunctionOutput(result=None)

    if not HAS_NEO4J:
        msg = "neo4j package is required for graph store workflow. Install with: pip install neo4j"
        raise ImportError(msg)

    # 从配置字典读取参数
    uri = graph_store_config.get("uri", "bolt://localhost:7687")
    username = graph_store_config.get("username", "neo4j")
    password = graph_store_config.get("password", "neo4j")
    database = graph_store_config.get("database", "neo4j")
    clear_database = graph_store_config.get("clear_database", False)

    logger.info(f"Neo4j 连接信息: {uri}, database={database}")

    driver = GraphDatabase.driver(  # type: ignore[union-attr]
        uri,
        auth=(username, password),
    )

    try:
        if clear_database:
            _clear_neo4j_database(driver, database)

        with driver.session(database=database) as session:
            session.run("MATCH (n) RETURN n LIMIT 1")
        logger.info("✓ Neo4j 连接成功")

        # 加载所有索引数据
        logger.info("\n加载索引数据...")

        # 使用配置的 output storage 而非 context.output_storage
        from graphrag.storage.file_pipeline_storage import FilePipelineStorage
        output_dir = Path(config.root_dir) / config.output.base_dir
        output_storage = FilePipelineStorage(base_dir=str(output_dir))

        logger.info(f"从 {output_dir} 读取索引数据...")

        documents = await load_table_from_storage("documents", output_storage)
        text_units = await load_table_from_storage("text_units", output_storage)
        entities = await load_table_from_storage("entities", output_storage)
        relationships = await load_table_from_storage("relationships", output_storage)
        communities = await load_table_from_storage("communities", output_storage)
        community_reports = await load_table_from_storage("community_reports", output_storage)

        logger.info(f"  - 文档: {len(documents)}")
        logger.info(f"  - 文本单元: {len(text_units)}")
        logger.info(f"  - 实体: {len(entities)}")
        logger.info(f"  - 关系: {len(relationships)}")
        logger.info(f"  - 社区: {len(communities)}")
        logger.info(f"  - 社区报告: {len(community_reports)}")

        # 依次导入
        logger.info("\n开始导入到 Neo4j...")
        _create_document_nodes(driver, documents, database)
        _import_chunks(driver, text_units, database=database)
        _import_entities(driver, entities, database=database)
        _import_relationships(driver, relationships, database=database)
        _import_communities(driver, communities, database=database)
        _import_community_reports(driver, community_reports, database=database)

        logger.info("\n✓ Neo4j 图数据构建完成")
    finally:
        driver.close()
        logger.info("已关闭 Neo4j 连接")

    logger.info("=" * 60)
    logger.info("Workflow completed: create_graph_store")
    logger.info("=" * 60)

    return WorkflowFunctionOutput(result=None)
def _parallel_batched_import(
    driver,
    statement: str,
    df: pd.DataFrame,
    batch_size: int = 100,
    max_workers: int = 8,
    database: str = "neo4j",
) -> dict[str, Any]:
    """使用并行处理进行批量导入数据到Neo4j.

    Parameters
    ----------
    driver
        Neo4j驱动实例
    statement
        Cypher查询语句,使用value作为每行数据的引用
    df
        要导入的DataFrame
    batch_size
        每批处理的行数
    max_workers
        并行线程数
    database
        Neo4j数据库名称

    Returns
    -------
    dict[str, Any]
        导入统计信息的字典
    """
    total = len(df)
    batches = (total + batch_size - 1) // batch_size
    start_time = time.time()
    results = []

    logger.info(
        "开始并行导入 %s 行数据,分为 %s 个批次,每批 %s 条",
        total,
        batches,
        batch_size,
    )

    def process_batch(batch_idx):
        start = batch_idx * batch_size
        end = min(start + batch_size, total)
        batch = df.iloc[start:end]

        batch_start_time = time.time()

        try:
            with driver.session(database=database) as session:
                result = session.run(
                    "UNWIND $rows AS value " + statement,
                    rows=batch.to_dict("records"),
                )
                summary = result.consume()
                batch_duration = time.time() - batch_start_time

                return {
                    "batch": batch_idx,
                    "rows": end - start,
                    "success": True,
                    "duration": batch_duration,
                    "counters": summary.counters,
                }
        except Exception as exc:
            batch_duration = time.time() - batch_start_time
            logger.exception(
                "批次 %s (行 %s-%s) 处理失败",
                batch_idx,
                start,
                end - 1,
            )
            return {
                "batch": batch_idx,
                "rows": end - start,
                "success": False,
                "duration": batch_duration,
                "error": str(exc),
            }

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_batch, i) for i in range(batches)]

        for i, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            result = future.result()
            results.append(result)

            if result["success"]:
                # logger.info(
                #     "批次 %s 完成: %s 行, 耗时 %.2f 秒",
                #     result["batch"],
                #     result["rows"],
                #     result["duration"],
                # )
                pass
            else:
                logger.error(
                    "批次 %s 失败: %s 行, 耗时 %.2f 秒, 错误: %s",
                    result["batch"],
                    result["rows"],
                    result["duration"],
                    result.get("error", "未知错误"),
                )

            logger.info(
                "进度: %s/%s 批次完成 (%.1f%%)",
                i,
                batches,
                (i / batches * 100),
            )

    successful_rows = sum(r["rows"] for r in results if r["success"])
    failed_rows = sum(r["rows"] for r in results if not r["success"])

    duration = time.time() - start_time
    rows_per_second = successful_rows / duration if duration > 0 else 0

    logger.info(
        "导入完成: 总计 %s 行, 成功 %s 行, 失败 %s 行",
        total,
        successful_rows,
        failed_rows,
    )
    logger.info("总耗时: %.2f 秒, 平均速度: %.2f 行/秒", duration, rows_per_second)

    return {
        "total_rows": total,
        "successful_rows": successful_rows,
        "failed_rows": failed_rows,
        "duration_seconds": duration,
        "rows_per_second": rows_per_second,
        "batch_results": results,
    }


def _create_document_nodes(
    driver, df_documents: pd.DataFrame, database: str = "neo4j"
) -> dict[str, Any]:
    """创建 Document 节点."""
    with driver.session(database=database) as session:
        try:
            session.run(
                "CREATE CONSTRAINT IF NOT EXISTS FOR (d:__Document__) "
                "REQUIRE d.id IS UNIQUE"
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("创建 Document 约束时出错 (可能已存在): %s", exc)

    cypher_statement = """
    MERGE (d:__Document__ {id: value.id})
    ON CREATE SET
        d.human_readable_id = value.human_readable_id,
        d.title = value.title,
        d.text = value.text,
        d.creation_date = value.creation_date,
        d.import_timestamp = timestamp()
    """

    return _parallel_batched_import(
        driver, cypher_statement, df_documents, database=database
    )


def _setup_chunk_constraints(driver, database: str = "neo4j") -> None:
    """创建 Chunk 标签的约束."""
    with driver.session(database=database) as session:
        try:
            session.run(
                "CREATE CONSTRAINT IF NOT EXISTS FOR (c:__Chunk__) "
                "REQUIRE c.id IS UNIQUE"
            )
            logger.info("已创建 Chunk.id 唯一性约束")
        except Exception as exc:  # noqa: BLE001
            logger.warning("创建 __Chunk__ 约束时出错 (可能已存在): %s", exc)


def _import_chunks(
    driver,
    df_chunks: pd.DataFrame,
    batch_size: int = 100,
    max_workers: int = 8,
    database: str = "neo4j",
) -> dict[str, Any]:
    """导入文档块 (Chunk) 到 Neo4j."""
    _setup_chunk_constraints(driver, database)

    logger.info("开始导入 Chunk 节点...")
    chunk_statement = """
    MERGE (c:__Chunk__ {id: value.id})
    SET c.text = value.text,
        c.n_tokens = value.n_tokens,
        c.human_readable_id = value.human_readable_id,
        c.name = value.human_readable_id
    """

    chunk_result = _parallel_batched_import(
        driver,
        chunk_statement,
        df_chunks,
        batch_size,
        max_workers,
        database,
    )

    logger.info("准备 Chunk-Document 关系数据...")
    relations_data = []

    for _, row in df_chunks.iterrows():
        chunk_id = row["id"]
        doc_ids_container = row["document_ids"]

        flat_doc_ids = []

        if hasattr(doc_ids_container, "dtype") and hasattr(
            doc_ids_container,
            "tolist",
        ):
            doc_ids_container = doc_ids_container.tolist()

        if isinstance(doc_ids_container, list):
            for item in doc_ids_container:
                if hasattr(item, "dtype") and hasattr(item, "tolist"):
                    flat_doc_ids.extend(item.tolist())
                elif isinstance(item, list):
                    flat_doc_ids.extend(item)
                else:
                    flat_doc_ids.append(item)
        elif doc_ids_container is not None:
            flat_doc_ids.append(doc_ids_container)

        for doc_id in flat_doc_ids:
            if doc_id is not None and str(doc_id).strip() != "":
                doc_id_str = str(doc_id).strip()
                relations_data.append({
                    "chunk_id": chunk_id,
                    "document_id": doc_id_str,
                })

    if relations_data:
        logger.info(
            "开始创建 %s 个 Chunk-Document 关系...",
            len(relations_data),
        )
        df_relations = pd.DataFrame(relations_data)

        relation_statement = """
        MATCH (c:__Chunk__ {id: value.chunk_id})
        MATCH (d:__Document__ {id: value.document_id})
        MERGE (c)-[:PART_OF]->(d)
        """

        relation_result = _parallel_batched_import(
            driver,
            relation_statement,
            df_relations,
            batch_size,
            max_workers,
            database,
        )
        logger.info(
            "已创建 %s 个 Chunk-Document 关系",
            relation_result["successful_rows"],
        )

    return chunk_result


def _setup_entity_constraints(driver, database: str = "neo4j") -> None:
    """创建 Entity 标签的约束."""
    with driver.session(database=database) as session:
        try:
            session.run(
                "CREATE CONSTRAINT IF NOT EXISTS FOR (e:__Entity__) "
                "REQUIRE e.id IS UNIQUE"
            )
            session.run(
                "CREATE CONSTRAINT IF NOT EXISTS FOR (e:__Entity__) "
                "REQUIRE e.name IS UNIQUE"
            )
            logger.info("已创建 __Entity__.id/name 唯一性约束")
        except Exception as exc:  # noqa: BLE001
            logger.warning("创建 __Entity__ 约束时出错 (可能已存在): %s", exc)


def _import_entities(
    driver,
    df_entities: pd.DataFrame,
    batch_size: int = 100,
    max_workers: int = 8,
    database: str = "neo4j",
) -> dict[str, Any]:
    """导入实体 (Entity) 到 Neo4j."""
    _setup_entity_constraints(driver, database)

    logger.info("预处理 text_unit_ids ...")
    df_entities = df_entities.copy()

    for idx, row in df_entities.iterrows():  # type: ignore[assignment]
        text_unit_ids = row.get("text_unit_ids")

        if not isinstance(text_unit_ids, list):
            if isinstance(text_unit_ids, str):
                try:
                    text_unit_ids = json.loads(text_unit_ids)
                except Exception:  # noqa: BLE001
                    text_unit_ids = [text_unit_ids]
            elif hasattr(text_unit_ids, "dtype") and hasattr(
                text_unit_ids,
                "tolist",
            ):
                text_unit_ids = text_unit_ids.tolist()  # type: ignore[union-attr]
            else:
                text_unit_ids = [text_unit_ids] if text_unit_ids is not None else []

        flat_text_unit_ids = []
        for item in text_unit_ids:
            if isinstance(item, list) or (
                hasattr(item, "dtype") and hasattr(item, "tolist")
            ):
                if hasattr(item, "tolist"):
                    flat_text_unit_ids.extend(item.tolist())  # type: ignore[union-attr]
                else:
                    flat_text_unit_ids.extend(item)
            else:
                flat_text_unit_ids.append(item)

        flat_text_unit_ids = [
            str(id_)
            for id_ in flat_text_unit_ids
            if id_ is not None and str(id_).strip() != ""
        ]
        # 使用 .at 而不是 .loc 来赋值列表，避免长度不匹配错误
        df_entities.at[idx, "text_unit_ids"] = flat_text_unit_ids

    logger.info("开始导入 __Entity__ 节点...")

    entity_statement = """
    MERGE (e:__Entity__ {id:value.id})
    SET e += value {.human_readable_id, .description, .frequency, .degree, .x, .y}
    SET e.name = replace(coalesce(value.title, value.human_readable_id, ''), '"', '')
    SET e.type = value.type

    // 动态添加type字段值作为标签（使用APOC）
    WITH e, value.type AS entity_type
    CALL apoc.do.when(
        entity_type IS NOT NULL AND entity_type <> '',
        'CALL apoc.create.addLabels(e, [entity_type]) YIELD node RETURN node',
        'RETURN e as node',
        {e: e, entity_type: entity_type}
    ) YIELD value as result
    RETURN result.node as e
    """

    entity_result = _parallel_batched_import(
        driver,
        entity_statement,
        df_entities,
        batch_size,
        max_workers,
        database,
    )
    logger.info(
        "已创建 %s 个 __Entity__ 节点(含动态标签)",
        entity_result["successful_rows"],
    )

    logger.info("准备 Entity-Chunk 关系数据...")
    entity_chunk_relations = []

    for _, row in df_entities.iterrows():
        entity_id = row["id"]
        text_unit_ids = row.get("text_unit_ids", [])

        entity_chunk_relations.extend(
            {
                "entity_id": entity_id,
                "chunk_id": chunk_id,
            }
            for chunk_id in text_unit_ids
            if chunk_id
        )

    if entity_chunk_relations:
        logger.info(
            "开始创建 %s 个 HAS_ENTITY 关系...",
            len(entity_chunk_relations),
        )
        df_entity_chunk_relations = pd.DataFrame(entity_chunk_relations)

        relation_statement = """
        MATCH (c:__Chunk__ {id: value.chunk_id})
        MATCH (e:__Entity__ {id: value.entity_id})
        MERGE (c)-[:HAS_ENTITY]->(e)
        """

        relation_result = _parallel_batched_import(
            driver,
            relation_statement,
            df_entity_chunk_relations,
            batch_size,
            max_workers=1,
            database=database,
        )
        logger.info(
            "已创建 %s 个 HAS_ENTITY 关系",
            relation_result["successful_rows"],
        )

    with driver.session(database=database) as session:
        result = session.run("MATCH (e:__Entity__) RETURN count(e) as count")
        entity_count = result.single()["count"]

        result = session.run(
            "MATCH (c:__Chunk__)-[r:HAS_ENTITY]->(e:__Entity__) "
            "RETURN count(r) as count"
        )
        relation_count = result.single()["count"]

        logger.info(
            "验证结果: %s 个 __Entity__ 节点, %s 个 HAS_ENTITY 关系",
            entity_count,
            relation_count,
        )

    return entity_result


def _setup_relationship_constraints(driver, database: str = "neo4j") -> None:
    """创建 Relationship 标签的约束."""
    with driver.session(database=database) as session:
        try:
            session.run(
                "CREATE CONSTRAINT IF NOT EXISTS FOR (r:__Relationship__) "
                "REQUIRE r.id IS UNIQUE"
            )
            logger.info("已创建 __Relationship__.id 唯一性约束")
        except Exception as exc:  # noqa: BLE001
            logger.warning("创建 __Relationship__ 约束时出错 (可能已存在): %s", exc)


def _import_relationships(
    driver,
    df_relationships: pd.DataFrame,
    batch_size: int = 100,
    max_workers: int = 8,
    database: str = "neo4j",
) -> dict[str, Any]:
    """导入关系数据到 Neo4j."""
    _setup_relationship_constraints(driver, database)

    logger.info("预处理关系中的 text_unit_ids ...")
    df_relationships = df_relationships.copy()

    for idx, row in df_relationships.iterrows():  # type: ignore[assignment]
        text_unit_ids = row.get("text_unit_ids")

        if not isinstance(text_unit_ids, list):
            if isinstance(text_unit_ids, str):
                try:
                    text_unit_ids = json.loads(text_unit_ids)
                except Exception:  # noqa: BLE001
                    text_unit_ids = [text_unit_ids]
            elif hasattr(text_unit_ids, "dtype") and hasattr(
                text_unit_ids,
                "tolist",
            ):
                text_unit_ids = text_unit_ids.tolist()  # type: ignore[union-attr]
            else:
                text_unit_ids = [text_unit_ids] if text_unit_ids is not None else []

        flat_text_unit_ids = []
        for item in text_unit_ids:
            if isinstance(item, list) or (
                hasattr(item, "dtype") and hasattr(item, "tolist")
            ):
                if hasattr(item, "tolist"):
                    flat_text_unit_ids.extend(item.tolist())  # type: ignore[union-attr]
                else:
                    flat_text_unit_ids.extend(item)
            else:
                flat_text_unit_ids.append(item)

        flat_text_unit_ids = [
            str(id_)
            for id_ in flat_text_unit_ids
            if id_ is not None and str(id_).strip() != ""
        ]
        # 使用 .at 而不是 .loc 来赋值列表，避免长度不匹配错误
        df_relationships.at[idx, "text_unit_ids"] = flat_text_unit_ids

    logger.info("开始导入关系数据...")

    logger.info("步骤1: 创建 Relationship 节点...")
    relationship_node_statement = """
    MERGE (r:__Relationship__ {id: value.id})
    SET r.human_readable_id = value.human_readable_id,
        r.description = value.description,
        r.weight = value.weight,
        r.combined_degree = value.combined_degree,
        r.name = value.human_readable_id
    """

    relationship_result = _parallel_batched_import(
        driver,
        relationship_node_statement,
        df_relationships,
        batch_size,
        max_workers,
        database,
    )
    logger.info(
        "已创建 %s 个 __Relationship__ 节点",
        relationship_result["successful_rows"],
    )

    logger.info("步骤2: 创建 Entity 之间的 RELATED 关系...")
    entity_relations = []
    for _, row in df_relationships.iterrows():
        entity_relations.append({
            "relationship_id": row["id"],
            "source_name": row["source"],
            "target_name": row["target"],
            "description": row.get("description", ""),
            "weight": row.get("weight", 0),
        })

    if entity_relations:
        df_entity_relations = pd.DataFrame(entity_relations)

        entity_rel_statement = """
        MATCH (r:__Relationship__ {id: value.relationship_id})
        MATCH (source:__Entity__ {name: value.source_name})
        MATCH (target:__Entity__ {name: value.target_name})
        MERGE (source)-[rel:RELATED]->(target)
        SET rel.relationship_id = value.relationship_id,
            rel.description = value.description,
            rel.weight = value.weight
        """

        entity_rel_result = _parallel_batched_import(
            driver,
            entity_rel_statement,
            df_entity_relations,
            batch_size,
            max_workers=1,
            database=database,
        )
        logger.info(
            "已创建 %s 个 Entity 之间的 RELATED 关系",
            entity_rel_result["successful_rows"],
        )

    logger.info("步骤3: 创建 Chunk-Relationship 关系...")
    chunk_relations = []
    for _, row in df_relationships.iterrows():
        rel_id = row["id"]
        chunk_relations.extend(
            {
                "relationship_id": rel_id,
                "chunk_id": chunk_id,
            }
            for chunk_id in row["text_unit_ids"]
        )

    if chunk_relations:
        df_chunk_relations = pd.DataFrame(chunk_relations)

        chunk_rel_statement = """
        MATCH (r:__Relationship__ {id: value.relationship_id})
        MATCH (c:__Chunk__ {id: value.chunk_id})
        MERGE (c)-[:HAS_RELATIONSHIP]->(r)
        """

        chunk_rel_result = _parallel_batched_import(
            driver,
            chunk_rel_statement,
            df_chunk_relations,
            batch_size,
            max_workers=1,
            database=database,
        )
        logger.info(
            "已创建 %s 个 Chunk-Relationship 关系",
            chunk_rel_result["successful_rows"],
        )

    return relationship_result


def _setup_community_constraints(driver, database: str = "neo4j") -> None:
    """创建 Community 标签的约束."""
    with driver.session(database=database) as session:
        try:
            session.run(
                "CREATE CONSTRAINT IF NOT EXISTS FOR (c:__Community__) "
                "REQUIRE c.id IS UNIQUE"
            )
            logger.info("已创建 __Community__.id 唯一性约束")
        except Exception as exc:  # noqa: BLE001
            logger.warning("创建 __Community__ 约束时出错 (可能已存在): %s", exc)


def _import_communities(
    driver,
    df_communities: pd.DataFrame,
    batch_size: int = 100,
    max_workers: int = 8,
    database: str = "neo4j",
) -> dict[str, Any]:
    """导入社区 (Community) 数据到 Neo4j."""
    _setup_community_constraints(driver, database)

    logger.info("预处理社区中的列表字段...")
    df_communities = df_communities.copy()

    list_fields = ["children", "entity_ids", "relationship_ids", "text_unit_ids"]

    for field in list_fields:
        if field in df_communities.columns:
            for idx, row in df_communities.iterrows():  # type: ignore[assignment]
                field_value = row.get(field)

                if not isinstance(field_value, list):
                    if isinstance(field_value, str):
                        try:
                            field_value = json.loads(field_value)
                        except Exception:  # noqa: BLE001
                            field_value = [field_value]
                    elif hasattr(field_value, "dtype") and hasattr(
                        field_value,
                        "tolist",
                    ):
                        field_value = field_value.tolist()  # type: ignore[union-attr]
                    else:
                        field_value = [field_value] if field_value is not None else []

                flat_field_value = []
                for item in field_value:
                    if isinstance(item, list) or (
                        hasattr(item, "dtype") and hasattr(item, "tolist")
                    ):
                        if hasattr(item, "tolist"):
                            flat_field_value.extend(item.tolist())  # type: ignore[union-attr]
                        else:
                            flat_field_value.extend(item)
                    else:
                        flat_field_value.append(item)

                if field in ["entity_ids", "relationship_ids", "text_unit_ids"]:
                    flat_field_value = [
                        str(id_)
                        for id_ in flat_field_value
                        if id_ is not None and str(id_).strip() != ""
                    ]

                # 使用 .at 而不是 .loc 来赋值列表
                df_communities.at[idx, field] = flat_field_value

    logger.info("开始导入社区节点...")

    community_statement = """
    MERGE (c:__Community__ {id: value.id})
    SET c.human_readable_id = value.human_readable_id,
        c.community = value.community,
        c.level = value.level,
        c.parent = value.parent,
        c.children = value.children,
        c.title = value.title,
        c.period = value.period,
        c.size = value.size,
        c.name = coalesce(value.title, value.human_readable_id, 'Community_' + value.id)

    RETURN c.id as community_id
    """

    community_result = _parallel_batched_import(
        driver,
        community_statement,
        df_communities,
        batch_size,
        max_workers,
        database,
    )
    logger.info(
        "已创建 %s 个 __Community__ 节点",
        community_result["successful_rows"],
    )

    logger.info("开始创建社区与实体的关系...")
    entity_relations = []
    for _, row in df_communities.iterrows():
        community_id = row["id"]
        entity_ids = row.get("entity_ids", [])

        entity_relations.extend(
            {
                "community_id": community_id,
                "entity_id": entity_id,
            }
            for entity_id in entity_ids
        )

    if entity_relations:
        df_entity_relations = pd.DataFrame(entity_relations)

        entity_rel_statement = """
        MATCH (c:__Community__ {id: value.community_id})
        MATCH (e:__Entity__ {id: value.entity_id})
        MERGE (e)-[:IN_COMMUNITY]->(c)
        """

        entity_rel_result = _parallel_batched_import(
            driver,
            entity_rel_statement,
            df_entity_relations,
            batch_size,
            max_workers=1,
            database=database,
        )
        logger.info(
            "已创建 %s 个 Entity-Community 关系",
            entity_rel_result["successful_rows"],
        )

    logger.info("开始创建社区与关系的关系...")
    rel_relations = []
    for _, row in df_communities.iterrows():
        community_id = row["id"]
        relationship_ids = row.get("relationship_ids", [])

        rel_relations.extend(
            {
                "community_id": community_id,
                "relationship_id": rel_id,
            }
            for rel_id in relationship_ids
        )

    if rel_relations:
        df_rel_relations = pd.DataFrame(rel_relations)

        rel_rel_statement = """
        MATCH (c:__Community__ {id: value.community_id})
        MATCH (r:__Relationship__ {id: value.relationship_id})
        MERGE (r)-[:IN_COMMUNITY]->(c)
        """

        rel_rel_result = _parallel_batched_import(
            driver,
            rel_rel_statement,
            df_rel_relations,
            batch_size,
            max_workers=1,
            database=database,
        )
        logger.info(
            "已创建 %s 个 Relationship-Community 关系",
            rel_rel_result["successful_rows"],
        )

    return community_result


def _import_community_reports(
    driver,
    df_reports: pd.DataFrame,
    batch_size: int = 100,
    max_workers: int = 8,
    database: str = "neo4j",
) -> dict[str, Any]:
    """导入社区报告数据到 Neo4j."""
    logger.info("预处理社区报告数据...")
    df_reports = df_reports.copy()

    df_reports["community_str"] = None
    df_reports["processed_findings"] = None

    for idx, row in df_reports.iterrows():  # type: ignore[assignment]
        if "community" in row:
            community_str = str(row["community"])
            df_reports.at[idx, "community_str"] = community_str

        findings = row.get("findings")

        if hasattr(findings, "dtype") and hasattr(findings, "tolist"):
            try:
                findings = findings.tolist()  # type: ignore[union-attr]
            except Exception as exc:  # noqa: BLE001
                logger.warning("行 %s: 转换 NumPy 数组失败: %s", idx, exc)
                findings = []
        elif not isinstance(findings, list):
            if isinstance(findings, str):
                try:
                    findings = json.loads(findings)
                except Exception:  # noqa: BLE001
                    findings = []
            else:
                findings = []

        valid_findings = []
        for i, finding in enumerate(findings):
            if isinstance(finding, dict):
                if "summary" not in finding:
                    finding["summary"] = f"Finding_{i}"
                if "explanation" not in finding:
                    finding["explanation"] = ""
                valid_findings.append(finding)

        # 使用 .at 而不是 .loc 来赋值列表
        df_reports.at[idx, "processed_findings"] = valid_findings

    logger.info("准备 Finding 数据...")
    findings_data = []

    for _idx, row in df_reports.iterrows():
        community_str = row["community_str"]
        processed_findings = row["processed_findings"]

        if not isinstance(processed_findings, list):
            continue

        for i, finding in enumerate(processed_findings):
            if isinstance(finding, dict):
                finding_id = f"{community_str}_{i}"
                findings_data.append({
                    "finding_id": finding_id,
                    "community_id": community_str,
                    "summary": finding.get("summary", f"Finding_{i}"),
                    "explanation": finding.get("explanation", ""),
                })

    logger.info("准备了 %s 个 Finding 数据", len(findings_data))

    logger.info("步骤1: 导入社区节点...")

    community_statement = """
    MERGE (c:__Community__ {community: value.community_str})
    SET c.level = value.level,
        c.title = value.title,
        c.rank = value.rank,
        c.rating_explanation = value.rating_explanation,
        c.full_content = value.full_content,
        c.summary = value.summary,
        c.name = coalesce(value.title, 'Community_' + value.community_str)
    RETURN c.community as community_id
    """

    community_result = _parallel_batched_import(
        driver,
        community_statement,
        df_reports,
        batch_size,
        max_workers,
        database,
    )
    logger.info(
        "已创建/更新 %s 个社区节点",
        community_result["successful_rows"],
    )

    if findings_data:
        logger.info("步骤2: 导入 Finding 节点和关系...")
        df_findings = pd.DataFrame(findings_data)

        finding_statement = """
        MERGE (f:__Finding__ {id: value.finding_id})
        SET f.summary = value.summary,
            f.explanation = value.explanation,
            f.name = value.summary

        WITH f, value
        MATCH (c:__Community__ {community: value.community_id})
        MERGE (c)-[:HAS_FINDING]->(f)
        """

        finding_result = _parallel_batched_import(
            driver,
            finding_statement,
            df_findings,
            batch_size,
            max_workers,
            database,
        )
        logger.info(
            "已创建 %s 个 Finding 节点和 HAS_FINDING 关系",
            finding_result["successful_rows"],
        )

    return community_result


def _clear_neo4j_database(driver, database: str = "neo4j") -> None:
    """清空现有 Neo4j 数据库中的数据、约束和索引。."""
    logger.info("清空现有 Neo4j 图数据...")
    with driver.session(database=database) as session:
        session.run("MATCH (n) DETACH DELETE n")

        constraints = session.run("SHOW CONSTRAINTS")
        for constraint in constraints:
            constraint_name = constraint["name"]
            session.run(f"DROP CONSTRAINT {constraint_name}")

        indexes = session.run("SHOW INDEXES")
        for index in indexes:
            index_name = index["name"]
            session.run(f"DROP INDEX {index_name}")

    logger.info("数据、约束和索引清空完成")


# ============================================================================
# 自动注册所有自定义工作流
# ============================================================================
# 注意：必须在所有函数定义之后调用，否则会出现 NameError
# register_custom_workflows()
