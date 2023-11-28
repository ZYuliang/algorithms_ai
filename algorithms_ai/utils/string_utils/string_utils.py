import re
import unicodedata
from collections import Counter

import regex

from algorithms_ai.utils.string_utils.common_string import CHINESE_PUNCTUATION, ENGLISH_PUNCTUATION
from algorithms_ai.utils.string_utils.longcov import Converter


def get_string_type(c):
    if '\u0030' <= c <= '\u0039':
        return 'num'
    elif c in CHINESE_PUNCTUATION:
        return 'punctuation_zh'
    elif c in ENGLISH_PUNCTUATION:
        return 'punctuation_en'
    elif '\u0041' <= c <= '\u005a':
        return 'char_english_uppercase'
    elif '\u0061' <= c <= '\u007a':
        return 'char_english_lowercase'
    elif '\u0391' <= c <= '\u03a9':
        return 'greek_uppercase'
    elif '\u03b1' <= c <= '\u03c9':
        return 'greek_lowercase'
    elif '\u4e00' <= c <= '\u9fff':
        return 'chinese'
    return 'other'


def get_string_language(c):
    # https://en.wikipedia.org/wiki/Unicode_block
    # 仅识别中英日韩俄法德
    if ('\u0041' <= c <= '\u005a') or ('\u0061' <= c <= '\u007a') or c in ENGLISH_PUNCTUATION:
        return 'en'
    elif ('\u4e00' <= c <= '\u9fff') or ('\u2E80' <= c <= '\u2FDF') or (
            '\u3400' <= c <= '\u4DBF') or c in CHINESE_PUNCTUATION:
        # 第一个简体，XXX,第三个繁体
        return 'zh'
    elif ('\uac00' <= c <= '\ud7ff') or ('\u1100' <= c <= '\u11FF') or ('\u3130' <= c <= '\u318F'):
        return 'ko'
    elif ('\u31F0' <= c <= '\u31FF') or ('\u3040' <= c <= '\u30FF'):
        return 'ja'
    elif ('\u00C0' <= c <= '\u00FF'):
        return 'de/fr'
    elif ('\u0400' <= c <= '\u052F'):
        return 'ru'
    return 'other'


def get_text_language(text, length_limit=100):
    # 仅识别中英日韩俄法德
    language_res = []
    for c in text:
        char_lan = get_string_language(c)
        if char_lan != 'other':
            language_res.append(char_lan)
        if len(language_res) >= length_limit:
            break
    language_res = Counter(language_res)
    if len(language_res) == 1:
        return list(language_res.keys())[0]
    else:
        if 'zh' in language_res:
            if 'ja' in language_res and (language_res['ja'] >= 0.4 * sum(language_res.values())):
                return 'ja'
            else:
                return 'zh'
        elif 'en' in language_res:
            if 'ru' in language_res:
                return 'ru'
            else:  # 法英德默认为英文
                return 'en'
        else:
            if language_res:
                return sorted(language_res.items(), key=lambda x: x[1], reverse=True)[0][0]
            else:
                return 'other'


def zh_convert_chinese(text, flag=1):
    # flag=0 表示简转繁，flag=1表示繁转简
    rule = 'zh-hans' if flag else 'zh-hant'
    return Converter(rule).convert(text)


def en_convert_accent_chars(text):
    # 上面这些可能有些问题
    # refined_texts = [unicodedata.normalize('NFKD', c).encode('ascii', 'ignore').decode("utf-8")
    #                  if get_language(c) in ('en', 'de/fr', 'ru') else c for c in text]
    # return ''.join(refined_texts)
    # return "".join(c for c in unicodedata.normalize("NFD", text).encode('ascii', 'ignore').decode("utf-8") if unicodedata.category(c) != "Mn")
    ###
    #     norm = unicodedata.normalize("NFD", text)
    #     result = "".join(ch for ch in norm if unicodedata.category(ch) != 'Mn')
    #     return unicodedata.normalize("NFC", result)
    ###
    # 把重音英文变成原始英文，比如àéêö等,输入输出不变
    res = regex.sub(r'\p{Mn}', '', unicodedata.normalize('NFKD', text))
    if len(res) == len(text):
        return res
    else:
        return text


def en_convert_full_width(text):
    """全角转半角，全角英文字母占用两个字节（一个汉字的宽度），所以看起来一个全角的字母占位会比半角的字母宽。比如‘ａ’,输入输出长度不变"""
    refined_char = []
    for char in text:
        inside_code = ord(char)
        if inside_code == 12288:  # 全角空格直接转换
            inside_code = 32
        elif 65374 >= inside_code >= 65281:  # 全角字符（除空格）根据关系转化
            inside_code -= 65248
        refined_char.append(chr(inside_code))

    return ''.join(refined_char)


def en_text_lower(text):
    # 有时会出现多一个长度（拉丁文大写转小写）
    lower_text = text.lower()
    if len(lower_text) != len(text):
        lower_text = ''.join([i for i in lower_text if i != '̇'])
    return lower_text


def is_digit(token_text):
    # 字符串是否是数字
    if token_text.isdigit() or (token_text.count('.') == 1 and token_text[0:token_text.find('.')].isdigit()
                                and token_text[token_text.find('.') + 1:].isdigit()):
        return True
    else:
        return False


def get_wordnet_pos(tag):
    from nltk.corpus import wordnet
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def pos_tag(token):
    """ 词性标注
    CC  并列连词          NNS 名词复数        UH 感叹词
    CD  基数词              NNP 专有名词        VB 动词原型
    DT  限定符            NNP 专有名词复数    VBD 动词过去式
    EX  存在词            PDT 前置限定词      VBG 动名词或现在分词
    FW  外来词            POS 所有格结尾      VBN 动词过去分词
    IN  介词或从属连词     PRP 人称代词        VBP 非第三人称单数的现在时
    JJ  形容词            PRP$ 所有格代词     VBZ 第三人称单数的现在时
    JJR 比较级的形容词     RB  副词            WDT 以wh开头的限定词
    JJS 最高级的形容词     RBR 副词比较级      WP 以wh开头的代词
    LS  列表项标记         RBS 副词最高级      WP$ 以wh开头的所有格代词
    MD  情态动词           RP  小品词          WRB 以wh开头的副词
    NN  名词单数           SYM 符号            TO  to
    """
    from nltk import pos_tag
    return pos_tag([token])[-1][-1]  # type:str


def get_lemmatizer():
    # 词性还原（例如：dysfunct -> dysfunctional）,复数变成单数
    # token_text = lemmatizer(token_text, pos=self.get_wordnet_pos(self.pos_tag(token_text)))
    # http://www.nltk.org/nltk_data/
    from nltk.stem.wordnet import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()  # 词形还原
    return lemmatizer.lemmatize  # usage: get_lemmatizer()('token_text')


def get_stemmer(method='poreter'):
    # 词干提取（例如：dysfunctional  -> dysfunct）
    if method == 'lancaster':
        from nltk.stem.lancaster import LancasterStemmer
        stemmer = LancasterStemmer()
    elif method == 'snowball':
        from nltk.stem import SnowballStemmer
        stemmer = SnowballStemmer('english')
    else:
        from nltk.stem.porter import PorterStemmer
        stemmer = PorterStemmer()
    return stemmer.stem  # usage: get_stemmer()('token_text')


def correct_text(text):
    # 拼写校对
    from textblob import TextBlob
    return str(TextBlob(text).correct())


def generate_ngram_token(n, token_list: list):
    # n-gram dict , n=token_length ,n最小为1 ， n>=1
    return [" ".join(token_list[i:i + n]) for i in range(0, len(token_list) - n + 1)]


def other_code(text):
    # 移除非法字符,比如存储openxyl时
    ILLEGAL_CHARACTERS_RE = re.compile(r'[\000-\010]|[\013-\014]|[\016-\037]')
    text = ILLEGAL_CHARACTERS_RE.sub(r'', text)

    # 移除所有不可见字符，包括：'\t', '\n', '\r'
    text = ''.join(t for t in text if t.isprintable())

    return


def find_decimal_and_integer(text):
    # 找到文本中的小数和整数
    res = re.compile("\d+\.?\d+").finditer(text)
    decimals = []
    integers = []
    for i in res:
        if '.' in i.group():
            decimals.append((i.group(), i.start()))
        else:
            integers.append((i.group(), i.start()))
    return decimals, integers


if __name__ == '__main__':
    t = 'dsfβsdfàéêöhelloａ'
    # print(en_convert_full_width(t))
    print(en_convert_accent_chars(t))
