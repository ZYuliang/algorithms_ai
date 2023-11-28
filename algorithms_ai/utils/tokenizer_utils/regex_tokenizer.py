import re


class RegexTokenizer:
    def __init__(self, regex=None, keep_tokens=()):
        # 规则切词，可以保留哪些词不用切分
        # 同一个字符开头满足多个规则，以第一个规则作数
        if not regex:
            from algorithms_ai.utils.string_utils.common_string import PUNCTUATIONS, HTML_ESCAPE_DICT
            # 默认：数字（整数和小数）| 英文单词(英文连字符包含) | 空白符 | 中文 | 希腊大写字母 | 希腊小写字母 | 标点 | 其他字符
            regex = f"\d+\.?\d+|[A-Za-z]+|\s+|[\u4e00-\u9fa5]|[\u0391-\u03a9]|[\u03b1-\u03c9]|[{PUNCTUATIONS}]|[^a-zA-Z0-9{PUNCTUATIONS}\s\u4e00-\u9fa5\u03b1-\u03c9\u0391-\u03a9]"
            html_escape = list(HTML_ESCAPE_DICT.keys())
            regex = '|'.join(html_escape) + '|' + regex
        if keep_tokens:
            regex = '|'.join(keep_tokens) + '|' + regex
        self.find_re = re.compile(regex)

    def run(self, text):
        # 输入文本，输出切分后的文本以及坐标--start-end--- 最后一个字符的坐标，而不是索引
        texts = []
        offsets_mapping = []
        texts_append = texts.append
        offsets_mapping_append = offsets_mapping.append
        tmp_index = 0

        token_index = 0
        for i in self.find_re.finditer(text):
            if tmp_index < i.start():
                if text[tmp_index:i.start()].strip():
                    texts_append(text[tmp_index:i.start()])
                    offsets_mapping_append((tmp_index, i.start() - 1))
                    token_index += 1
            if i.group().strip():
                texts_append(i.group())
                offsets_mapping_append((i.start(), i.end() - 1))
                token_index += 1
            tmp_index = i.end()

        if (tmp_index < len(text)) and (text[tmp_index:len(text) - 1].strip()):
            texts_append(text[tmp_index:len(text) - 1])
            offsets_mapping_append((tmp_index, len(text) - 1))
            token_index += 1
        return {'texts': texts, 'offsets_mapping': offsets_mapping}


if __name__ == '__main__':
    # s4 = "CX (cisplatin 80 mg/m(2) IV Q3W; capecitabine 1000 mg/m(2) P.O. BID for 14 days Q3W) plus intravenous AMG 386 10 mg/kg QW (Arm A"
    # t = SentenceTokenizer(keep_suffix=('s.v',))
    #
    # print(t.run(s4))
    #
    # s = "Oral acalabruti(nib) 100 mg 我i在发啊 <  twic\u00b7e daily  ; was & admi\nnister\ted with or &lt; 13.8kg/m without,α-1 12.😊 "
    # w_t = WordTokenizer(regex=None, keep_words=('在发',))
    # print(w_t.run(s))
    # s2 = 'Recent advances in CRISPR/Cas9-mediated knock-ins in mammalian cells.'
    # s2 = '1H-NMR'
    # re2 = f"[A-Za-z0-9]+|\s+|[\u4e00-\u9fa5]|[\u0391-\u03a9]|[\u03b1-\u03c9]"
    # print(WordTokenizer(regex=re2).run(s2))
    #
    # ENGLISH_PUNCTUATION = '!"#$%&\'()*+/:<=>?@[\\]^`{|}~-'
    # CHINESE_PUNCTUATION = '＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､\u3000、〃〈〉《》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏﹑﹔·！？｡。'
    # PUNCTUATIONS = ENGLISH_PUNCTUATION + CHINESE_PUNCTUATION
    #
    # regex = f"[A-Za-z0-9_.;,]+|\s+|[\u4e00-\u9fa5]|[\u0391-\u03a9]|[\u03b1-\u03c9]|[{PUNCTUATIONS}]"
    # find_re = re.compile(regex)
    # print(list(find_re.finditer('Wnt/β-cateninh,2.2a算法')))
    # print(RegexTokenizer(keep_tokens=('2.2a',)).run('Wnt/β-cateninh, 2.2a%'))
    s = '  Terrible swelβli-IIng 12.3up.\nI went   to d&gt;my GP在发啊  '
    r = ''
    print(RegexTokenizer(r, keep_tokens=('2.2a',)).run(s))

