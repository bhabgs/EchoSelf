"""
性格/风格提取模块 - 从聊天记录中提取说话风格和性格特点
Personality Extractor Module - Extract Speaking Style and Personality from Chat History

功能：
1. 解析不同格式的聊天记录（txt, json, csv）
2. 分析用户的说话风格和常用表达
3. 生成性格描述和风格示例
4. 创建用于 LLM 的 System Prompt
"""

import os
import sys
import re
import json
import random
from pathlib import Path
from typing import Optional, Union, List, Dict, Tuple
from collections import Counter
from loguru import logger
import chardet

# 配置 loguru
logger.remove()
logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> - <level>{message}</level>")


class PersonalityExtractor:
    """
    性格提取器类
    从聊天记录中分析和提取用户的说话风格
    """

    def __init__(
        self,
        max_samples: int = 20,
        max_chars: int = 5000
    ):
        """
        初始化性格提取器

        Args:
            max_samples: 用于 few-shot 的最大示例数
            max_chars: 处理的最大字符数
        """
        self.max_samples = max_samples
        self.max_chars = max_chars

        # 提取的数据
        self.messages: List[Dict[str, str]] = []  # 原始消息
        self.style_examples: List[Dict[str, str]] = []  # 对话示例
        self.personality_traits: List[str] = []  # 性格特点
        self.common_expressions: List[str] = []  # 常用表达
        self.speaking_patterns: Dict[str, any] = {}  # 说话模式统计

        logger.info("PersonalityExtractor 初始化完成")

    def load_chat_history(
        self,
        file_path: Union[str, Path],
        user_identifier: Optional[str] = None
    ) -> bool:
        """
        加载聊天记录文件

        Args:
            file_path: 聊天记录文件路径（支持 .txt, .json, .csv）
            user_identifier: 用户标识符（用于区分消息来源）

        Returns:
            bool: 是否加载成功
        """
        file_path = Path(file_path)

        if not file_path.exists():
            logger.error(f"文件不存在: {file_path}")
            return False

        logger.info(f"正在加载聊天记录: {file_path.name}")

        try:
            # 检测文件编码
            with open(file_path, 'rb') as f:
                raw_data = f.read()
                detected = chardet.detect(raw_data)
                encoding = detected.get('encoding', 'utf-8')

            # 根据文件扩展名选择解析方法
            suffix = file_path.suffix.lower()

            if suffix == '.json':
                self._parse_json(file_path, encoding, user_identifier)
            elif suffix == '.csv':
                self._parse_csv(file_path, encoding, user_identifier)
            else:  # 默认按 txt 处理
                self._parse_txt(file_path, encoding, user_identifier)

            logger.info(f"✓ 加载完成，共 {len(self.messages)} 条消息")
            return True

        except Exception as e:
            logger.error(f"加载聊天记录失败: {e}")
            return False

    def _parse_txt(
        self,
        file_path: Path,
        encoding: str,
        user_identifier: Optional[str]
    ):
        """
        解析纯文本格式的聊天记录

        支持的格式：
        1. "用户名: 消息内容"
        2. "[时间] 用户名: 消息内容"
        3. "用户名\n消息内容" (微信导出格式)
        """
        with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
            content = f.read()

        # 尝试多种格式匹配
        patterns = [
            # 格式1: "名字: 消息"
            r'^([^:\n]+?):\s*(.+?)(?=\n[^:\n]+?:|$)',
            # 格式2: "[时间] 名字: 消息"
            r'\[[\d\-\s:]+\]\s*([^:\n]+?):\s*(.+?)(?=\[[\d\-\s:]+\]|$)',
            # 格式3: "名字\n消息" (每两行一组)
            r'^(.+?)\n(.+?)(?=\n.+?\n|$)',
        ]

        for pattern in patterns:
            matches = re.findall(pattern, content, re.MULTILINE | re.DOTALL)
            if matches and len(matches) > 5:  # 至少匹配5条才认为格式正确
                for name, message in matches:
                    name = name.strip()
                    message = message.strip()
                    if message and len(message) > 1:
                        is_user = (user_identifier and user_identifier in name) or \
                                  (not user_identifier)  # 如果没指定，则都算用户消息
                        self.messages.append({
                            "role": "user" if is_user else "other",
                            "name": name,
                            "content": message
                        })
                break

        # 如果上面的模式都没匹配到，按行处理
        if not self.messages:
            lines = content.strip().split('\n')
            for line in lines:
                line = line.strip()
                if line and len(line) > 1:
                    self.messages.append({
                        "role": "user",
                        "name": "user",
                        "content": line
                    })

    def _parse_json(
        self,
        file_path: Path,
        encoding: str,
        user_identifier: Optional[str]
    ):
        """
        解析 JSON 格式的聊天记录

        支持的格式：
        1. [{"role": "user", "content": "..."}]
        2. [{"sender": "名字", "message": "..."}]
        3. {"messages": [...]}
        """
        with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
            data = json.load(f)

        # 如果是字典，尝试提取消息列表
        if isinstance(data, dict):
            for key in ['messages', 'data', 'chats', 'conversations']:
                if key in data:
                    data = data[key]
                    break

        if not isinstance(data, list):
            logger.warning("JSON 格式不正确，无法解析")
            return

        for item in data:
            if isinstance(item, dict):
                # 尝试不同的字段名
                content = item.get('content') or item.get('message') or \
                          item.get('text') or item.get('msg')
                role = item.get('role') or item.get('type') or 'user'
                name = item.get('name') or item.get('sender') or \
                       item.get('from') or item.get('user') or 'user'

                if content:
                    is_user = (user_identifier and user_identifier in str(name)) or \
                              (role in ['user', 'human', 'sent'])
                    self.messages.append({
                        "role": "user" if is_user else "other",
                        "name": str(name),
                        "content": str(content).strip()
                    })

    def _parse_csv(
        self,
        file_path: Path,
        encoding: str,
        user_identifier: Optional[str]
    ):
        """
        解析 CSV 格式的聊天记录
        """
        import csv

        with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # 尝试不同的列名
                content = row.get('content') or row.get('message') or \
                          row.get('text') or row.get('msg') or ''
                name = row.get('name') or row.get('sender') or \
                       row.get('from') or row.get('user') or 'user'

                if content:
                    is_user = user_identifier and user_identifier in str(name)
                    self.messages.append({
                        "role": "user" if is_user else "other",
                        "name": str(name),
                        "content": str(content).strip()
                    })

    def analyze(self) -> Dict:
        """
        分析聊天记录，提取风格特点

        Returns:
            Dict: 分析结果
        """
        if not self.messages:
            logger.warning("没有消息可分析")
            return {}

        logger.info("正在分析聊天风格...")

        # 只分析用户的消息
        user_messages = [m['content'] for m in self.messages if m['role'] == 'user']

        # 1. 基础统计
        self.speaking_patterns = self._compute_statistics(user_messages)

        # 2. 提取常用表达
        self.common_expressions = self._extract_expressions(user_messages)

        # 3. 分析性格特点
        self.personality_traits = self._analyze_personality(user_messages)

        # 4. 生成对话示例
        self.style_examples = self._generate_examples()

        result = {
            "statistics": self.speaking_patterns,
            "expressions": self.common_expressions,
            "traits": self.personality_traits,
            "examples": self.style_examples
        }

        logger.info("✓ 分析完成")
        return result

    def _compute_statistics(self, messages: List[str]) -> Dict:
        """
        计算说话模式统计
        """
        stats = {
            "total_messages": len(messages),
            "avg_length": 0,
            "max_length": 0,
            "min_length": 0,
            "uses_emoji": False,
            "uses_punctuation_repeat": False,  # 如 "！！！"
            "sentence_endings": {},  # 句末习惯
            "question_ratio": 0,  # 问句比例
        }

        if not messages:
            return stats

        lengths = [len(m) for m in messages]
        stats["avg_length"] = sum(lengths) / len(lengths)
        stats["max_length"] = max(lengths)
        stats["min_length"] = min(lengths)

        # 检查是否使用表情符号
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags
            "]+",
            flags=re.UNICODE
        )
        stats["uses_emoji"] = any(emoji_pattern.search(m) for m in messages)

        # 检查标点重复
        punct_repeat = re.compile(r'[!！?？~～]{2,}')
        stats["uses_punctuation_repeat"] = any(punct_repeat.search(m) for m in messages)

        # 句末统计
        endings = []
        for m in messages:
            m = m.strip()
            if m:
                endings.append(m[-1] if len(m) > 0 else '')
        stats["sentence_endings"] = dict(Counter(endings).most_common(5))

        # 问句比例
        question_count = sum(1 for m in messages if '?' in m or '？' in m)
        stats["question_ratio"] = question_count / len(messages)

        return stats

    def _extract_expressions(self, messages: List[str]) -> List[str]:
        """
        提取常用表达和口头禅
        """
        expressions = []

        # 常见口头禅模式
        patterns = [
            r'(哈+)',  # 哈哈哈
            r'(嗯+)',  # 嗯嗯
            r'(噢+|哦+)',  # 噢噢
            r'(好吧|好的|行吧|可以)',
            r'(其实|然后|就是说|反正)',
            r'(hhh+|233+|666+)',  # 网络用语
            r'([~～]+)',  # 波浪号
            r'(啦|呀|呢|吧|哟|咯)$',  # 语气词
        ]

        word_counter = Counter()
        for m in messages:
            for pattern in patterns:
                matches = re.findall(pattern, m, re.IGNORECASE)
                word_counter.update(matches)

        # 取最常见的表达
        expressions = [word for word, count in word_counter.most_common(10)
                       if count >= 2]

        return expressions

    def _analyze_personality(self, messages: List[str]) -> List[str]:
        """
        分析性格特点
        """
        traits = []

        all_text = ' '.join(messages)

        # 基于关键词的简单分析
        trait_patterns = {
            "幽默风趣": [r'哈哈', r'笑', r'搞笑', r'有趣'],
            "热情友好": [r'太好了', r'棒', r'喜欢', r'爱', r'感谢', r'谢谢'],
            "直接干脆": [r'直接', r'简单', r'就这样', r'搞定'],
            "细心体贴": [r'注意', r'小心', r'别忘了', r'记得'],
            "喜欢提问": [r'\?|？'],
            "表达丰富": [r'!|！', r'~|～'],
            "网络用语达人": [r'hhh|233|666|awsl|xswl|yyds'],
        }

        for trait, patterns in trait_patterns.items():
            count = sum(len(re.findall(p, all_text, re.IGNORECASE))
                        for p in patterns)
            if count >= 3:  # 至少出现3次
                traits.append(trait)

        # 基于长度分析
        avg_len = self.speaking_patterns.get("avg_length", 0)
        if avg_len < 10:
            traits.append("言简意赅")
        elif avg_len > 50:
            traits.append("表达详尽")

        return traits if traits else ["自然随和"]

    def _generate_examples(self) -> List[Dict[str, str]]:
        """
        生成对话示例（用于 few-shot）
        """
        examples = []

        # 找出带有上下文的对话对
        for i in range(len(self.messages) - 1):
            current = self.messages[i]
            next_msg = self.messages[i + 1]

            # 找 "别人说 -> 用户回复" 的模式
            if current['role'] != 'user' and next_msg['role'] == 'user':
                examples.append({
                    "input": current['content'],
                    "output": next_msg['content']
                })

        # 如果没有找到对话对，就用用户的单独消息
        if not examples:
            user_msgs = [m['content'] for m in self.messages if m['role'] == 'user']
            for msg in user_msgs[:self.max_samples]:
                examples.append({
                    "input": "(用户主动发言)",
                    "output": msg
                })

        # 随机选择并限制数量
        if len(examples) > self.max_samples:
            examples = random.sample(examples, self.max_samples)

        return examples

    def generate_system_prompt(
        self,
        template: Optional[str] = None
    ) -> str:
        """
        生成用于 LLM 的系统提示词

        Args:
            template: 自定义模板（可选）

        Returns:
            str: 系统提示词
        """
        if not self.personality_traits:
            self.analyze()

        # 默认模板
        if template is None:
            template = """你是一个数字分身AI助手，你需要模仿以下人物的说话风格和性格特点进行对话。

## 人物性格描述：
{personality_description}

## 常用表达习惯：
{expressions}

## 说话风格示例：
{style_examples}

## 重要规则：
1. 保持该人物的语气、用词习惯和表达方式
2. 回复要自然、口语化，像真人聊天一样
3. 避免过于正式或机械的回复
4. 根据上下文适当使用该人物常用的口头禅或表达
5. 回复长度适中，通常1-3句话即可

请开始对话，记住你现在就是这个人的数字分身。"""

        # 构建性格描述
        personality_description = "\n".join(
            f"- {trait}" for trait in self.personality_traits
        )

        # 构建常用表达
        expressions = "\n".join(
            f"- {expr}" for expr in self.common_expressions
        ) or "- 自然表达，无特殊口头禅"

        # 构建风格示例
        style_examples = ""
        for ex in self.style_examples[:5]:  # 最多5个示例
            style_examples += f"\n对方: {ex['input']}\n回复: {ex['output']}\n"

        # 填充模板
        prompt = template.format(
            personality_description=personality_description,
            expressions=expressions,
            style_examples=style_examples
        )

        return prompt

    def export_analysis(self, output_path: Union[str, Path]):
        """
        导出分析结果到文件

        Args:
            output_path: 输出文件路径
        """
        output_path = Path(output_path)

        result = {
            "statistics": self.speaking_patterns,
            "expressions": self.common_expressions,
            "traits": self.personality_traits,
            "examples": self.style_examples,
            "system_prompt": self.generate_system_prompt()
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        logger.info(f"分析结果已导出: {output_path}")


# ============================================================
# 测试代码
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("PersonalityExtractor 测试")
    print("=" * 60)

    # 创建测试数据
    test_chat = """
张三: 早上好呀~
李四: 早！今天天气不错哈哈
张三: 是呀，出去玩吗？
李四: 好呀好呀，去哪里？
张三: 随便，你说了算
李四: 那去公园吧，反正离得近
张三: 行吧，那几点出发？
李四: 10点？不着急哈哈
张三: 可以可以，那待会见
李四: 好的好的，回聊~
"""

    # 写入测试文件
    test_file = Path("/tmp/test_chat.txt")
    with open(test_file, 'w', encoding='utf-8') as f:
        f.write(test_chat)

    # 创建提取器
    extractor = PersonalityExtractor()

    # 加载聊天记录
    print("\n1. 加载聊天记录...")
    if extractor.load_chat_history(test_file, user_identifier="李四"):
        print(f"✓ 加载成功，共 {len(extractor.messages)} 条消息")
    else:
        print("✗ 加载失败")
        exit(1)

    # 分析
    print("\n2. 分析聊天风格...")
    result = extractor.analyze()
    print(f"✓ 分析完成")
    print(f"  - 性格特点: {extractor.personality_traits}")
    print(f"  - 常用表达: {extractor.common_expressions}")

    # 生成系统提示词
    print("\n3. 生成系统提示词...")
    prompt = extractor.generate_system_prompt()
    print("=" * 40)
    print(prompt)
    print("=" * 40)

    # 清理测试文件
    test_file.unlink()

    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)
