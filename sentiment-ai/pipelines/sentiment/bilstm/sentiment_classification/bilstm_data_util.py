"""
预处理公共工具
功能：语言检测、分词、特殊标记
"""
import re
import jieba


SPECIAL_TOKENS = {'<PAD>': 0, '<UNK>': 1, '[CLS]': 2, '[SEP]': 3}


def detect_language(text):
    """检测文本语言类型"""
    zh = len(re.findall(r'[\u4e00-\u9fa5]', text))
    en = len(re.findall(r'[a-zA-Z]', text))
    if zh > 0 and en > 0:
        return 'mixed'
    elif zh > 0:
        return 'zh'
    return 'en'


def tokenize_text(text, zh_mode='char'):
    """
    统一分词入口

    Args:
        text: str 输入文本
        zh_mode: str 中文分词模式 'char' / 'jieba'

    Returns:
        List[str]: 分词后的token列表
    """
    lang = detect_language(text)

    if lang == 'zh':
        if zh_mode == 'char':
            return list(text)
        else:
            return jieba.lcut(text)

    elif lang == 'en':
        return text.split()

    else:  # mixed
        tokens = []
        i = 0
        while i < len(text):
            ch = text[i]
            if '\u4e00' <= ch <= '\u9fa5':
                zh_text = ''
                while i < len(text) and '\u4e00' <= text[i] <= '\u9fa5':
                    zh_text += text[i]
                    i += 1
                if zh_mode == 'char':
                    tokens.extend(list(zh_text))
                else:
                    tokens.extend(jieba.lcut(zh_text))
            elif ch.isalpha() and ch.isascii():
                en_text = ''
                while i < len(text) and text[i].isalpha() and text[i].isascii():
                    en_text += text[i]
                    i += 1
                tokens.append(en_text)
            else:
                if not ch.isspace():
                    tokens.append(ch)
                i += 1
        return tokens