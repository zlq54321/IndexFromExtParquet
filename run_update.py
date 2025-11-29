"""运行增量索引

使用外部 parquet 数据进行增量索引的完整脚本。

用法:
    # 使用当前目录作为项目根目录
    python run_update.py

    # 指定项目根目录
    python run_update.py --root /path/to/project

    # 指定多个项目目录
    python run_update.py --root /path/to/project1
    python run_update.py --root /path/to/project2

环境变量:
    GRAPHRAG_API_KEY: OpenAI API Key

项目目录结构:
    project_root/
    ├── input/
    │   ├── documents.parquet
    │   └── text_units.parquet
    ├── output/
    ├── update_output/
    └── settings.yaml
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 确保当前目录在路径中
sys.path.insert(0, ".")

# 1. 注册自定义模块（必须在导入 graphrag 之前！）
from custom_parquet_loader import register_parquet_loader
from custom_workflows import register_custom_workflows
from prompt_validator import validate_prompts_before_run
from milvusdb import register_milvus_vector_store

register_parquet_loader()
register_custom_workflows()
register_milvus_vector_store()

# 2. 导入 GraphRAG
from graphrag.index.run.run_pipeline import run_pipeline
from graphrag.index.workflows.factory import PipelineFactory
from graphrag.callbacks.noop_workflow_callbacks import NoopWorkflowCallbacks
from graphrag.config.create_graphrag_config import create_graphrag_config
import yaml
from dotenv import load_dotenv
import os
import re


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="运行 GraphRAG 增量索引（方案B）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用当前目录
  python run_update.py

  # 指定项目目录
  python run_update.py --root ./my_project

  # 电商领域项目
  python run_update.py --root /data/graphrag/ecommerce

  # 医疗领域项目
  python run_update.py --root /data/graphrag/medical
        """
    )
    parser.add_argument(
        "--root",
        type=str,
        default=".",
        help="项目根目录路径（包含 settings.yaml）。默认: 当前目录"
    )
    return parser.parse_args()


async def main():
    """运行增量索引主流程"""
    # 解析命令行参数
    args = parse_args()
    root_dir = Path(args.root).resolve()

    logger.info("=" * 60)
    logger.info("GraphRAG 增量索引")
    logger.info("=" * 60)
    logger.info(f"\n项目根目录: {root_dir}")

    # 验证目录
    if not root_dir.exists():
        logger.error(f"错误：目录不存在: {root_dir}")
        return 1

    settings_file = root_dir / "settings.yaml"
    if not settings_file.exists():
        logger.error(f"错误：未找到配置文件: {settings_file}")
        logger.info(f"请确保项目目录包含 settings.yaml")
        return 1

    # 步骤1: 加载配置
    logger.info("\n步骤1: 加载配置")

    # 加载 .env 文件
    env_file = root_dir / ".env"
    if env_file.exists():
        load_dotenv(env_file)
        logger.info(f"  ✓ 环境变量加载完成: {env_file}")

    # 手动读取并解析配置文件（需要特殊处理 ext_parquet loader）
    with open(settings_file, "r", encoding="utf-8") as f:
        yaml_content = f.read()

    # 替换环境变量 ${VAR} -> 实际值
    def replace_env_vars(match):
        var_name = match.group(1)
        return os.environ.get(var_name, match.group(0))

    yaml_content = re.sub(r'\$\{(\w+)\}', replace_env_vars, yaml_content)
    config_data = yaml.safe_load(yaml_content)

    # 临时保存自定义的 file_type（GraphRAG 验证只接受 csv/text/json）
    real_file_type = config_data.get("input", {}).get("file_type", "text")
    if "input" in config_data:
        config_data["input"]["file_type"] = "text"  # 临时使用有效值通过验证

    # 创建配置对象
    config = create_graphrag_config(config_data, str(root_dir))

    # 恢复真正的 file_type（用于自定义 ext_parquet loader）
    config.input.file_type = real_file_type

    logger.info(f"  ✓ 配置加载完成")
    logger.info(f"    - Root: {config.root_dir}")
    logger.info(f"    - Input: {config.input.storage.base_dir if config.input.storage else 'default'}")
    logger.info(f"    - Output: {config.output.base_dir}")
    logger.info(f"    - Update: {config.update_index_output.base_dir}")

    # 步骤1.5: 验证提示词模板
    logger.info("\n步骤1.5: 验证提示词模板")
    is_valid = validate_prompts_before_run(settings_file)

    if not is_valid:
        logger.error("\n" + "="*60)
        logger.error("❌ 提示词模板验证失败！")
        logger.error("请检查生成的 *_fix.txt 文件，确认修复内容后：")
        logger.error("1. 备份原始文件")
        logger.error("2. 用修复文件替换原文件")
        logger.error("3. 重新运行索引")
        logger.error("="*60)
        return 1

    # 步骤2: 运行增量索引
    logger.info("\n步骤2: 运行增量索引流水线")
    logger.info("-" * 60)

    # 创建 pipeline（使用自定义工作流列表，method 参数无意义）
    pipeline = PipelineFactory.create_pipeline(config)
    callbacks = NoopWorkflowCallbacks()

    workflow_count = 0
    finish_cnt = 0
    try:
        async for result in run_pipeline(pipeline, config, callbacks, is_update_run=True):
            workflow_count += 1

            if result.errors:
                logger.error(f"❌ 工作流 {result.workflow} 失败")
                for error in result.errors:
                    logger.error(f"   错误: {error}")
                break
            else:
                finish_cnt += 1
                logger.info(f"✓ 工作流 {result.workflow} 完成")

        logger.info("-" * 60)
        logger.info(f"\n✅ 完成 {finish_cnt} 个 工作流")

    except Exception as e:
        logger.error(f"\n❌ 增量索引失败: {e}", exc_info=True)
        return 1

    # 步骤4: 输出结果摘要
    logger.info("\n步骤4: 结果摘要")
    try:
        import pandas as pd

        # 使用配置中的 output 路径（支持 --root 参数）
        output_dir = root_dir / config.output.base_dir

        docs = pd.read_parquet(output_dir / "documents.parquet")
        text_units = pd.read_parquet(output_dir / "text_units.parquet")

        logger.info(f"  最终索引统计:")
        logger.info(f"    - 文档数: {len(docs)}")
        logger.info(f"    - 文本单元数: {len(text_units)}")

        try:
            entities = pd.read_parquet(output_dir / "entities.parquet")
            relationships = pd.read_parquet(output_dir / "relationships.parquet")
            logger.info(f"    - 实体数: {len(entities)}")
            logger.info(f"    - 关系数: {len(relationships)}")
        except:
            pass

    except Exception as e:
        logger.warning(f"  无法读取结果统计: {e}")

    logger.info("\n" + "=" * 60)
    logger.info(f"索引完成！输出位置: {output_dir}")
    logger.info("=" * 60)

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
