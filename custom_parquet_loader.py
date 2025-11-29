"""自定义 Parquet 加载器，用于直接加载外部准备的 parquet 文件。

使用方法：
1. 在 settings.yaml 中配置：
   input:
     storage:
       type: file
       base_dir: "./input"
     file_type: ext_parquet
     # 注意：不需要 file_pattern，会直接读取 documents.parquet

2. 在运行索引前导入此模块：
   from custom_parquet_loader import register_parquet_loader
   register_parquet_loader()

存储说明：
- 从 input.storage.base_dir 目录下读取 documents.parquet
- 这个 loader 直接读取整个 parquet 表，不需要 file_pattern
"""

import logging
from io import BytesIO

import pandas as pd

from graphrag.config.models.input_config import InputConfig
from graphrag.index.input.factory import loaders
from graphrag.storage.pipeline_storage import PipelineStorage

logger = logging.getLogger(__name__)


async def load_ext_parquet(
    config: InputConfig,
    storage: PipelineStorage,
) -> pd.DataFrame:
    """从存储中直接加载 documents.parquet 文件。

    这个 loader 会直接读取 input.storage.base_dir 下的 documents.parquet 文件，
    不需要像 csv/text loader 那样扫描目录。

    参数:
        config: 输入配置
        storage: 存储对象（对应 input.storage.base_dir）

    返回:
        文档 DataFrame
    """
    logger.info("=" * 60)
    logger.info("Loading external parquet file")
    logger.info(f"Storage base_dir: {config.storage.base_dir if config.storage else 'default'}")
    logger.info("=" * 60)

    # 直接读取 documents.parquet
    parquet_file = "documents.parquet"

    try:
        buffer = BytesIO(await storage.get(parquet_file, as_bytes=True))
        data = pd.read_parquet(buffer)
        logger.info(f"Successfully loaded {len(data)} rows from {parquet_file}")
    except Exception as e:
        logger.error(f"Failed to load {parquet_file}: {e}")
        raise ValueError(
            f"无法加载 {parquet_file}！\n"
            f"请确保文件位于配置的 input.storage.base_dir 目录下。\n"
            f"错误信息: {e}"
        )

    # 验证必需的列
    required_columns = ["id", "title", "text"]
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise ValueError(
            f"documents.parquet 缺少必需的列: {missing_columns}\n"
            f"必需列: {required_columns}\n"
            f"实际列: {data.columns.tolist()}"
        )

    logger.info(f"Columns: {data.columns.tolist()}")
    logger.info(f"Sample titles: {data['title'].head().tolist()}")

    return data


def register_parquet_loader():
    """注册 ext_parquet loader 到 GraphRAG 的 loaders 字典。

    调用此函数后，可以在配置中使用 file_type: ext_parquet

    注意：使用字符串 "ext_parquet" 而不是 InputFileType 枚举，
    因为 InputFileType 枚举中没有 ext_parquet。
    GraphRAG 的 create_input 函数会先尝试枚举匹配，失败后会直接用字符串作为 key。
    """
    # 直接用字符串 key 注册
    loaders["ext_parquet"] = load_ext_parquet
    logger.info("✓ ext_parquet loader registered successfully")
    logger.info("  Usage: file_type: ext_parquet")
    logger.info("  Will read: documents.parquet from input.storage.base_dir")
