# 字符串的规范化

from algorithms_ai.utils.string_utils.common_string import HTML_ESCAPE_DICT
from algorithms_ai.utils.string_utils.string_utils import en_convert_accent_chars, en_convert_full_width, \
    zh_convert_chinese, en_text_lower, is_digit
from algorithms_ai.utils.tokenizer_utils.regex_tokenizer import RegexTokenizer


class SampleNormalizer:
    def __init__(self,
                 is_en_convert_accent_chars=True,
                 is_en_convert_full_width=True,
                 is_en_text_lower=False,
                 is_zh_convert_chinese=True,

                 is_convert_html_escape=True,
                 is_convert_digit_to_special_token=False,

                 special_token_mapping=None  # 特殊token的处理
                 ):
        # 不变长度的处理：英文重音、全角，中文中繁体,小写转换
        # 改变长度的处理：转义字符处理，希腊字母处理，数字变成特殊字符,多空格去除，指定字符的变换,
        # 先切词然后对比每个词进行处理

        self.is_en_convert_accent_chars = is_en_convert_accent_chars
        self.is_en_convert_full_width = is_en_convert_full_width
        self.is_en_text_lower = is_en_text_lower
        self.is_zh_convert_chinese = is_zh_convert_chinese

        self.is_convert_html_escape = is_convert_html_escape
        self.html_escape = HTML_ESCAPE_DICT
        self.is_convert_digit_to_special_token = is_convert_digit_to_special_token

        self.special_token_mapping = special_token_mapping

        self.regex_tokenizer = RegexTokenizer().run

    def run(self, sample, ner_keys=()):
        input_text = sample['input_text']
        refined_input_text, index_refined_mapping_raw, index_raw_mapping_refined = self.normalize_text(input_text)

        refined_sample = {
            'input_text': refined_input_text
        }
        ner_res = {i: [[{'text': refined_input_text[index_raw_mapping_refined[_['start_offset']][0]:
                                                    index_raw_mapping_refined[_['end_offset']][-1] + 1],
                         'start_offset': index_raw_mapping_refined[_['start_offset']][0],
                         'end_offset': index_raw_mapping_refined[_['end_offset']][-1],
                         } for _ in j] for j in sample.get(i, [])] for i in ner_keys}
        refined_sample.update(ner_res)
        return refined_sample

    def normalize_text(self, input_text):
        # 规范化处理原始文本，得到处理后的文本,原始文本和处理后文本的互相index映射
        if self.is_en_convert_accent_chars:
            input_text = en_convert_accent_chars(input_text)

        if self.is_en_convert_full_width:
            input_text = en_convert_full_width(input_text)

        if self.is_zh_convert_chinese:
            input_text = zh_convert_chinese(input_text)

        if self.is_en_text_lower:
            input_text = en_text_lower(input_text)

        input_text_tokens = self.regex_tokenizer(input_text)
        mapping_res = self.get_index_mapping(input_text,
                                             input_text_tokens['texts'],
                                             input_text_tokens['offsets_mapping'])
        refined_input_text = mapping_res['refined_input_text']
        index_refined_mapping_raw = mapping_res['index_refined_mapping_raw']
        index_raw_mapping_refined = mapping_res['index_raw_mapping_refined']
        assert len(refined_input_text) == len(index_refined_mapping_raw)
        assert len(input_text) == len(index_raw_mapping_refined)
        return refined_input_text, index_refined_mapping_raw, index_raw_mapping_refined

    def get_index_mapping(self, input_text, tokens_list, offsets_mapping):
        # 给定处理后的token的列表和相应的映射，输出原始文本和处理后的文本的相互映射
        refined_input_text = []

        last_offsets = (-1, -1)
        index_raw_mapping_refined = []  # 给定原文中的坐标可以映射到处理后的文本上，
        index_refined_mapping_raw = []  # 给定处理后的文本，可以映射到原文
        tokens_list_len = len(tokens_list)

        for index in range(tokens_list_len):
            last_end_index = last_offsets[-1] + 1
            offsets = offsets_mapping[index]
            text = self.process_token(tokens_list[index])

            if last_end_index < offsets[0] and index != tokens_list_len - 1 and index != 0:
                refined_input_text.append(' ')
                index_refined_mapping_raw.append(
                    (offsets_mapping[index - 1][-1] + 1, offsets[0] - 1))

            if last_end_index != offsets[0]:
                for _ in range(last_end_index, offsets[0]):
                    index_raw_mapping_refined.append((len(index_refined_mapping_raw) - 1,
                                                      len(index_refined_mapping_raw) - 1))

            refined_input_text.append(text)
            if len(text) == len(tokens_list[index]):
                for _ in range(len(text)):
                    index_refined_mapping_raw.append((offsets[0] + _, offsets[0] + _))
                    index_raw_mapping_refined.append((len(index_refined_mapping_raw) - 1,
                                                      len(index_refined_mapping_raw) - 1))

            else:
                index_refined_mapping_raw.extend([(offsets[0], offsets[-1])] * len(text))
                for _ in range(offsets[1] - offsets[0] + 1):
                    index_raw_mapping_refined.append((len(index_refined_mapping_raw) - len(text),
                                                      len(index_refined_mapping_raw) - 1))

            last_offsets = offsets
        refined_input_text = ''.join(refined_input_text)

        if len(index_raw_mapping_refined) <= len(input_text):
            for _ in range(len(input_text) - len(index_raw_mapping_refined)):
                index_raw_mapping_refined.append((len(index_refined_mapping_raw),
                                                  len(index_refined_mapping_raw)))

        return {
            'refined_input_text': refined_input_text,
            'index_raw_mapping_refined': index_raw_mapping_refined,
            'index_refined_mapping_raw': index_refined_mapping_raw,
        }

    def process_token(self, token_text):
        if self.is_convert_html_escape and token_text in self.html_escape:
            token_text = self.html_escape[token_text]

        if self.is_convert_digit_to_special_token and is_digit(token_text):
            token_text = '[NUM]'

        if self.special_token_mapping and token_text in self.special_token_mapping:
            token_text = self.special_token_mapping[token_text]

        return token_text

    @staticmethod
    def get_label_from_mapping(offsets, offsets_mapping, to_text):
        # 给定映射的mapping和原始文本和需要映射的范围,输出返回的文本和offsets
        # offsets_mapping 注意要从offsets那边的文本到后面那个文本的映射
        # offsets放入的是index位置（-1）而不是索引
        start_offset = offsets_mapping[offsets[0]][0]
        end_offset = offsets_mapping[offsets[-1]][-1]
        return {
            'text': to_text[start_offset:end_offset + 1],
            'start_offset': start_offset,
            'end_offset': end_offset
        }

    def test(self):
        self.is_convert_digit_to_special_token = True
        sample = {
            'input_text': 'Terrible first of all I had diar&gt;rhoea my doseage was increased to 75 and a few days ago my feet and ankles started swelling up.发送多个',
            'NER_ADR': [[{'text': 'diar&gt;rhoea', 'start_offset': 28, 'end_offset': 40}],
                        [{'text': '75', 'start_offset': 70, 'end_offset': 71}],
                        [{'text': 'swelling', 'start_offset': 119, 'end_offset': 126},
                         {'text': 'feet', 'start_offset': 95, 'end_offset': 98}],
                        [{'text': 'ankles', 'start_offset': 104, 'end_offset': 109},
                         {'text': 'swelling', 'start_offset': 119, 'end_offset': 126}]],
        }
        print(self.run(sample, ner_keys=['NER_ADR']))


if __name__ == '__main__':
    SampleNormalizer().test()
