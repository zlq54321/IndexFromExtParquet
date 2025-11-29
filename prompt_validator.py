"""
提示词模板验证器
在索引开始前检测并修复 prompt 文件中的括号配对问题
"""

import logging
import sys
from pathlib import Path
from typing import List, Dict, Tuple
import yaml

logger = logging.getLogger(__name__)


class BraceError:
    """括号错误记录"""
    def __init__(self, line_num: int, position: int, error_type: str, context: str, fix: str):
        self.line_num = line_num
        self.position = position
        self.error_type = error_type
        self.context = context
        self.fix = fix

    def __repr__(self):
        return (f"行 {self.line_num}, 位置 {self.position}: {self.error_type}\n"
                f"  上下文: ...{self.context}...\n"
                f"  修复: {self.fix}")


class PromptValidator:
    """提示词模板验证器"""

    # 需要检查的配置路径
    PROMPT_CONFIGS = [
        ("extract_graph", "prompt"),  # extract_graph.prompt
        ("summarize_descriptions", "prompt"),  # summarize_descriptions.prompt
        ("community_reports", "graph_prompt"),  # community_reports.graph_prompt
    ]

    def __init__(self, settings_path: Path):
        self.settings_path = settings_path
        self.root_dir = settings_path.parent
        self.errors_found = False

    def validate_all(self) -> bool:
        """
        验证所有配置的提示词文件

        Returns:
            bool: True 如果所有文件都没问题，False 如果发现需要修复的问题
        """
        print("\n" + "="*80)
        print("开始验证提示词模板文件...")
        print("="*80 + "\n")

        # 读取配置
        with open(self.settings_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        prompt_files = self._extract_prompt_files(config)

        if not prompt_files:
            print("⚠️  警告: 未找到任何提示词文件配置")
            return True

        all_valid = True

        for prompt_name, prompt_path in prompt_files.items():
            result = self._validate_single_file(prompt_name, prompt_path)
            if not result:
                all_valid = False

        if not all_valid:
            print("\n" + "="*80)
            print("❌ 发现提示词模板错误！已生成修复文件，请检查后重新运行。")
            print("="*80 + "\n")
        else:
            print("\n" + "="*80)
            print("✅ 所有提示词模板验证通过")
            print("="*80 + "\n")

        return all_valid

    def _extract_prompt_files(self, config: dict) -> Dict[str, Path]:
        """从配置中提取提示词文件路径"""
        prompt_files = {}

        # 遍历配置的提示词路径
        for section, field in self.PROMPT_CONFIGS:
            if section in config and field in config[section]:
                prompt_path = config[section][field]
                if prompt_path:
                    # 使用 section.field 作为显示名称
                    prompt_files[f'{section}.{field}'] = self.root_dir / prompt_path

        return prompt_files

    def _validate_single_file(self, prompt_name: str, prompt_path: Path) -> bool:
        """
        验证单个提示词文件

        Returns:
            bool: True 如果文件有效，False 如果发现问题
        """
        print(f"📄 检查: {prompt_name}")
        print(f"   路径: {prompt_path}")

        if not prompt_path.exists():
            print(f"   ⚠️  文件不存在，跳过\n")
            return True

        # 读取文件
        with open(prompt_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 检测问题
        errors, fixed_content = self._detect_and_fix(content)

        if not errors:
            print(f"   ✅ 未发现问题\n")
            return True

        # 发现问题，生成修复文件
        fix_path = prompt_path.parent / f"{prompt_path.stem}_fix{prompt_path.suffix}"
        with open(fix_path, 'w', encoding='utf-8') as f:
            f.write(fixed_content)

        # 显示详细信息
        print(f"   ❌ 发现 {len(errors)} 个括号问题")
        print(f"   💾 已生成修复文件: {fix_path.name}\n")

        for i, error in enumerate(errors, 1):
            print(f"   问题 {i}:")
            print(f"      位置: 第 {error.line_num} 行")
            print(f"      类型: {error.error_type}")
            print(f"      上下文: ...{error.context}...")
            print(f"      修复: {error.fix}")
            print()

        return False

    def _detect_and_fix(self, content: str) -> Tuple[List[BraceError], str]:
        """
        检测并修复括号问题

        Returns:
            Tuple[List[BraceError], str]: (错误列表, 修复后的内容)
        """
        errors = []
        lines = content.split('\n')
        fixed_lines = []

        for line_num, line in enumerate(lines, 1):
            fixed_line, line_errors = self._fix_line(line, line_num)
            fixed_lines.append(fixed_line)
            errors.extend(line_errors)

        return errors, '\n'.join(fixed_lines)

    def _fix_line(self, line: str, line_num: int) -> Tuple[str, List[BraceError]]:
        """
        修复单行的括号问题

        核心逻辑：
        1. 使用栈跟踪 { } 的配对
        2. 检测每个 } 是否有对应的 {
        3. 对于孤立的 }，根据上下文判断：
           - 如果前面有未闭合的 ( 并且在 tuple 结构中 → 替换为 )
           - 否则 → 转义为 }}
        """
        errors = []
        result = []
        brace_stack = []  # 栈：记录未配对的 { 的位置
        paren_count = 0  # 当前未闭合的 ( 数量

        i = 0
        while i < len(line):
            char = line[i]

            if char == '(':
                paren_count += 1
                result.append(char)

            elif char == ')':
                if paren_count > 0:
                    paren_count -= 1
                result.append(char)

            elif char == '{':
                # 检查是否是转义的 {{
                if i + 1 < len(line) and line[i + 1] == '{':
                    # 已经转义的 {{，保持不变
                    result.append('{{')
                    i += 1  # 跳过下一个 {
                else:
                    brace_stack.append(len(result))  # 记录 { 在 result 中的位置
                    result.append(char)

            elif char == '}':
                # 检查是否是转义的 }}
                if i + 1 < len(line) and line[i + 1] == '}':
                    # 已经转义的 }}，保持不变
                    result.append('}}')
                    i += 1  # 跳过下一个 }
                elif brace_stack:
                    # 有配对的 {，正常闭合
                    brace_stack.pop()
                    result.append(char)
                else:
                    # 孤立的 }，需要判断如何处理
                    context = ''.join(result)

                    # 启发式规则1: 检查是否在类似 ("entity"...) 的结构中
                    in_tuple_structure = self._check_tuple_structure(context)

                    # 启发式规则2: 检查是否有未闭合的 (
                    has_unclosed_paren = paren_count > 0

                    # 启发式规则3: 检查是否在行末或紧跟换行
                    rest_of_line = line[i+1:].strip()
                    at_line_end = not rest_of_line or rest_of_line.startswith('\n')

                    if has_unclosed_paren and in_tuple_structure and at_line_end:
                        # 在 tuple 结构末尾的孤立 }，应该是 )
                        errors.append(BraceError(
                            line_num=line_num,
                            position=i,
                            error_type="孤立的 } 应该是 )",
                            context=context[-50:] if len(context) > 50 else context,
                            fix="} → )"
                        ))
                        result.append(')')
                        paren_count -= 1
                    else:
                        # 应该转义为字面的 }
                        errors.append(BraceError(
                            line_num=line_num,
                            position=i,
                            error_type="孤立的 } 需要转义",
                            context=context[-50:] if len(context) > 50 else context,
                            fix="} → }}"
                        ))
                        result.append('}}')
            else:
                result.append(char)

            i += 1

        return ''.join(result), errors

    def _check_tuple_structure(self, context: str) -> bool:
        """
        检查上下文是否在元组结构中

        识别类似这样的模式：
        - ("entity"{tuple_delimiter}...
        - ("relationship"{tuple_delimiter}...
        """
        # 简化判断：如果包含这些特征，认为在 tuple 结构中
        indicators = [
            '("entity"',
            '("relationship"',
            '{tuple_delimiter}',
            '{record_delimiter}',
        ]
        return any(ind in context for ind in indicators)


def validate_prompts_before_run(settings_path: Path) -> bool:
    """
    在索引运行前验证提示词文件

    Args:
        settings_path: settings.yaml 的路径

    Returns:
        bool: True 如果所有文件有效，False 如果需要修复
    """
    validator = PromptValidator(settings_path)
    return validator.validate_all()


if __name__ == "__main__":
    # 测试代码
    if len(sys.argv) > 1:
        settings_path = Path(sys.argv[1])
    else:
        settings_path = Path("settings.yaml")

    is_valid = validate_prompts_before_run(settings_path)
    sys.exit(0 if is_valid else 1)
