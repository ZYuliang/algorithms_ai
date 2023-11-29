"""
实体识别数据集样本处理
包括：
数据集的切分，句子切分：split_sample_and_label
输入样本的规范化：SampleNormalizer
实体标签结果的规范化：LabelNormalizer

"""
from algorithms_ai.utils.string_utils.common_string import HTML_ESCAPE_DICT
from algorithms_ai.utils.string_utils.string_utils import en_convert_accent_chars, en_convert_full_width, \
    zh_convert_chinese, en_text_lower, is_digit
from algorithms_ai.utils.tokenizer_utils.regex_tokenizer import RegexTokenizer


def find_seq_index(seq, seq_list):
    # 从一组offsets_mapping中找到某个seq的所属，否则返回 -1
    for sub_seq_index, sub_seq in enumerate(seq_list):
        if sub_seq[0] <= seq[0] <= seq[1] <= sub_seq[1]:
            return sub_seq_index
    return -1


def split_sample_and_label(sample, sentence_tokenizer=None, ner_keys=None, keep_features=None):
    # 对一组结果和对应的标签进行分句
    input_text = sample['input_text']

    refined_samples = []
    if sentence_tokenizer:
        if keep_features:
            features = {i: j for i, j in sample.items() if i in keep_features}
        else:
            features = dict()

        sentence_token_result = sentence_tokenizer(input_text)
        offsets_mapping = sentence_token_result['offsets_mapping']
        texts = sentence_token_result['texts']

        if ner_keys:
            ner_keys_res_dict = {i: {'input_text': texts[i], 'run_sentence_id': i} for i in range(len(texts))}
            for i, j in ner_keys_res_dict.items():
                j.update(features)
                j.update({j: [] for j in ner_keys})

            for ner_key in ner_keys:
                ner_key_res = sample.get(ner_key, [])
                for each_entity in ner_key_res:
                    seq_indexs = [find_seq_index((_['start_offset'], _['end_offset']), offsets_mapping)
                                  for _ in each_entity]
                    if -1 in seq_indexs or len(set(seq_indexs)) > 1:  # 确保一个实体只在一个输入（一句话）中出现
                        return []
                    seq_index = seq_indexs[0]
                    refined_entity = [{'text': _['text'],
                                       'start_offset': _['start_offset'] - offsets_mapping[seq_index][0],
                                       'end_offset': _['end_offset'] - offsets_mapping[seq_index][0]}
                                      for _ in each_entity]
                    ner_keys_res_dict[seq_index][ner_key].append(refined_entity)
            refined_samples.extend(ner_keys_res_dict.values())
        return refined_samples
    else:
        return [sample]


class InputNormalizer:
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

    def run(self, input_text, ner_results=None, ner_keys=()):
        refined_input_text, index_refined_mapping_raw, index_raw_mapping_refined = self.normalize_text(input_text)

        if ner_results and ner_keys:
            ner_results = {i: [[{'text': refined_input_text[index_raw_mapping_refined[_['start_offset']][0]:
                                                            index_raw_mapping_refined[_['end_offset']][-1] + 1],
                                 'start_offset': index_raw_mapping_refined[_['start_offset']][0],
                                 'end_offset': index_raw_mapping_refined[_['end_offset']][-1],
                                 } for _ in j] for j in ner_results.get(i, [])] for i in ner_keys}

        return refined_input_text, ner_results, index_refined_mapping_raw, index_raw_mapping_refined

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


class LabelNormalizer:
    def __init__(self,
                 token_start_add=(),
                 token_end_add=(),
                 token_start_remove=(),
                 token_end_remove=(),
                 token_remove=()
                 ):
        # 所有文本都小写，而且每个判断的条件和其他并不会有关联，如，加p> 和 > ,由于是break模式，所以写的时候要写全还要兼顾顺序，写【’p>‘,'>'】
        # 主要处理实体中的一些介词,标点和边界，如果确定不要某些边界就去除，如果需要边界就添加
        self.regex_tokenizer = RegexTokenizer().run  # 通用的切分word，理论上每个label都应该是这个切分的子集

        self.token_start_add = token_start_add  # 在标签前面添加
        self.token_end_add = token_end_add  # 在标签后面添加

        self.token_start_remove = token_start_remove  # 删除标签前缀
        self.token_end_remove = token_end_remove  # 删除标签后缀

        self.token_remove = token_remove  # 删除指定文本

        self.linking_punc = ['-', '_', '-', '-']  # token间的链接符号

    def run(self, input_text, ner_results, ner_keys, entity_repair_mode='complete'):
        # entity_repair_mode表示截断实体是要补全还是去除
        input_text = input_text
        input_text_tokens = self.regex_tokenizer(input_text)
        texts_list = input_text_tokens['texts']
        offsets_mapping = input_text_tokens['offsets_mapping']

        for ner_key in ner_keys:
            label_entities = ner_results.get(ner_key, [])
            refined_label_entities = self.normalize_label_entities(input_text, label_entities,
                                                                   texts_list, offsets_mapping,
                                                                   entity_repair_mode=entity_repair_mode)
            ner_results[ner_key] = refined_label_entities
        return ner_results

    def normalize_label_entities(self, input_text, label_entities, texts_list, offsets_mapping,
                                 entity_repair_mode='complete'):
        # 规范一个标签下的所有实体，交叉重复的实体去除
        entities_index = []
        for entity in label_entities:
            entities_index.append(self.normalize_entity(entity, texts_list, offsets_mapping,
                                                        entity_repair_mode=entity_repair_mode))

        entity_delete = [0 for _ in range(len(entities_index))]
        for each_entity_index in range(len(entities_index) - 1):
            for each_entity_index2 in range(each_entity_index + 1, len(entities_index)):
                include_info = self.judge_entity_included_by_token_index(entities_index[each_entity_index],
                                                                         entities_index[each_entity_index2])
                if include_info == 'entity_1_include_2':
                    entity_delete[each_entity_index2] = 1
                elif include_info == 'entity_2_include_1':
                    entity_delete[each_entity_index] = 1
                elif include_info == 'include_each_other':
                    entity_delete[each_entity_index2] = 1
        entities_index = [i for i, j in zip(entities_index, entity_delete) if j == 0]

        return [[{
            'text': input_text[offsets_mapping[s][0]:offsets_mapping[e][1] + 1],
            'start_offset': offsets_mapping[s][0],
            'end_offset': offsets_mapping[e][1]
        } for s, e in each_entity_index] for each_entity_index in entities_index
        ]

    @staticmethod
    def judge_entity_included_by_token_index(entity_1, entity_2):
        """
        判断两个实体是否互相包含：互不包含，1包含2，2包含1，互相包含
        b =[(21, 22),(27,29),(2,4)]
        e2 = [(21, 22), (27, 30)]
        print(judge_entity_included(b, e2))
        """
        entity_1_include_2_matrix = [[0] * len(entity_2) for _ in range(len(entity_1))]  # 1包含2的计数
        entity_2_include_1_matrix = [[0] * len(entity_1) for _ in range(len(entity_2))]  # 2包含1的计数
        for entity_1_part_index, entity_1_part in enumerate(entity_1):
            for entity_2_part_index, entity_2_part in enumerate(entity_2):
                if entity_2_part[0] <= entity_1_part[0] <= entity_1_part[1] <= entity_2_part[1]:
                    entity_2_include_1_matrix[entity_2_part_index][entity_1_part_index] = 1
                if entity_1_part[0] <= entity_2_part[0] <= entity_2_part[1] <= entity_1_part[1]:
                    entity_1_include_2_matrix[entity_1_part_index][entity_2_part_index] = 1

        entity_1_sum_max = sum([max([(entity_2_include_1_matrix[j][i]) for j in range(len(entity_2))])
                                for i in range(len(entity_1))])
        if entity_1_sum_max == len(entity_1):
            # 2包含1
            entity_2_include_1 = True
        else:
            entity_2_include_1 = False

        entity_2_sum_max = sum([max([(entity_1_include_2_matrix[j][i]) for j in range(len(entity_1))])
                                for i in range(len(entity_2))])
        if entity_2_sum_max == len(entity_2):
            # 1包含2
            entity_1_include_2 = True
        else:
            entity_1_include_2 = False

        if entity_1_include_2 and entity_2_include_1:
            return 'include_each_other'
        elif entity_1_include_2 and not entity_2_include_1:
            return 'entity_1_include_2'
        elif entity_2_include_1 and not entity_1_include_2:
            return 'entity_2_include_1'
        else:
            return 'not_include'

    def normalize_entity(self, entity, texts_list, offsets_mapping, entity_repair_mode='complete'):
        # 规范化一个实体,规范化单独的part，交叉重复的part去除,输出这个实体由哪几个token组成，输出token的index
        entity_index = []
        for entity_part in entity:
            start_end_offset_index = self.normalize_entity_part(entity_part, texts_list,
                                                                offsets_mapping,
                                                                entity_repair_mode=entity_repair_mode)
            if start_end_offset_index:
                entity_index.append(start_end_offset_index)
        entity_index = sorted(entity_index, key=lambda x: (x[0], x[-1]))

        # 不同实体片段的合并，处理嵌套片段
        if len(entity_index) > 1:
            refined_entity_index = []
            last_entity_index = entity_index[0]
            for current_entity_index in entity_index[1:]:
                if current_entity_index[0] <= last_entity_index[1] or \
                        last_entity_index[1] + 1 == current_entity_index[0]:
                    last_entity_index = (last_entity_index[0], current_entity_index[1])
                elif current_entity_index[0] > last_entity_index[1] + 1:
                    refined_entity_index.append(last_entity_index)
                    last_entity_index = current_entity_index
            refined_entity_index.append(last_entity_index)
        else:
            refined_entity_index = entity_index

        return refined_entity_index

    def normalize_entity_part(self, entity_part, texts_list, offsets_mapping, entity_repair_mode='complete'):
        # 规范化实体的部分,补全或删除

        refined_start_offset_index = -1

        refined_end_offset_index = -1
        entity_start_offset = entity_part['start_offset']
        entity_end_offset = entity_part['end_offset']

        for i, offsets in enumerate(offsets_mapping):
            if refined_start_offset_index >= 0 and refined_end_offset_index >= 0:
                break

            if refined_start_offset_index == -1:
                if offsets[0] <= entity_start_offset <= offsets[1]:
                    if entity_repair_mode == 'delete':
                        if i != len(offsets_mapping) - 1:
                            refined_start_offset_index = i + 1
                        else:
                            refined_start_offset_index = i
                    else:
                        refined_start_offset_index = i
                elif i != 0 and offsets_mapping[i - 1][1] < entity_start_offset < offsets[0]:
                    refined_start_offset_index = i
                elif i == 0 and 0 <= entity_start_offset < offsets[0]:
                    refined_start_offset_index = i

            if refined_start_offset_index >= 0 and refined_end_offset_index == -1:
                if offsets[0] <= entity_end_offset <= offsets[1]:
                    if entity_repair_mode == 'delete':
                        if i != 0:
                            refined_end_offset_index = i - 1
                        else:
                            refined_end_offset_index = i
                    else:
                        refined_end_offset_index = i
                elif i != len(offsets_mapping) - 1 and offsets[1] < entity_end_offset < offsets_mapping[i + 1][0]:
                    refined_end_offset_index = i
                elif i == len(offsets_mapping) - 1 and offsets[1] < entity_end_offset:
                    refined_end_offset_index = i
        if refined_end_offset_index >= refined_start_offset_index >= 0:
            start_offset_index, end_offset_index = self.process_entity_part(texts_list,
                                                                            refined_start_offset_index,
                                                                            refined_end_offset_index,
                                                                            offsets_mapping)
            if end_offset_index >= start_offset_index >= 0:
                return start_offset_index, end_offset_index
            else:
                return None
        else:
            return None

    def process_entity_part(self, texts_list, entity_start_index, entity_end_index, offsets_mapping):
        # 对抽取出来的一段实体进行边界的确定处理，包括增删token或直接删除，一般对介词进行处理
        while entity_start_index > 0 and \
                texts_list[entity_start_index - 1].lower() in self.token_start_add:
            entity_start_index = entity_start_index - 1

        while entity_start_index < len(texts_list) - 1 and \
                texts_list[entity_start_index].lower() in self.token_start_remove:
            entity_start_index = entity_start_index + 1

        while entity_end_index < len(texts_list) - 1 and \
                texts_list[entity_end_index + 1].lower() in self.token_end_add:
            entity_end_index = entity_end_index + 1

        while entity_end_index > 0 and \
                texts_list[entity_end_index].lower() in self.token_end_remove:
            entity_end_index = entity_end_index - 1

        # 使用'-'相邻的进行合并
        while entity_start_index > 1 and texts_list[entity_start_index - 1] in self.linking_punc and \
                (offsets_mapping[entity_start_index][0] - offsets_mapping[entity_start_index - 1][1] == 1) and \
                (offsets_mapping[entity_start_index - 1][0] - offsets_mapping[entity_start_index - 2][1] == 1):
            entity_start_index = entity_start_index - 2
        while len(texts_list) - 1 > entity_start_index > 0 and texts_list[entity_start_index] in self.linking_punc and \
                (offsets_mapping[entity_start_index + 1][0] - offsets_mapping[entity_start_index][1] == 1) and \
                (offsets_mapping[entity_start_index][0] - offsets_mapping[entity_start_index - 1][1] == 1):
            entity_start_index = entity_start_index - 1

        while entity_end_index < len(texts_list) - 2 and texts_list[entity_end_index + 1] in self.linking_punc and \
                (offsets_mapping[entity_end_index + 1][0] - offsets_mapping[entity_end_index][1] == 1) and \
                (offsets_mapping[entity_end_index + 2][0] - offsets_mapping[entity_end_index + 1][1] == 1):
            entity_end_index = entity_end_index + 2
        while 0 < entity_end_index < len(texts_list) - 1 and texts_list[entity_end_index] in self.linking_punc and \
                (offsets_mapping[entity_end_index][0] - offsets_mapping[entity_end_index - 1][1] == 1) and \
                (offsets_mapping[entity_end_index + 1][0] - offsets_mapping[entity_end_index][1] == 1):
            entity_end_index = entity_end_index + 1

        # 使用括号进行合并或去除
        sub_text_list = texts_list[entity_start_index:entity_end_index + 1]
        for left_punc, right_punc in [('(', ')'), ('[', ']')]:
            if sub_text_list.count(left_punc) == 1 and sub_text_list.count(right_punc) == 0 and \
                    entity_end_index < len(texts_list) - 1 and texts_list[entity_end_index + 1] == right_punc:
                if texts_list[entity_start_index] != left_punc:
                    entity_end_index = entity_end_index + 1

            if sub_text_list.count(right_punc) == 1 and sub_text_list.count(left_punc) == 0 and \
                    entity_start_index > 0 and texts_list[entity_start_index - 1] == left_punc:
                if texts_list[entity_end_index] != right_punc:
                    entity_start_index = entity_start_index - 1

        if entity_start_index == entity_end_index and texts_list[entity_start_index] in self.token_remove:
            entity_start_index = entity_start_index + 1

        return entity_start_index, entity_end_index

    def test(self):
        e = {
            'NER_DRUG': [
                [{'start_offset': 72, 'end_offset': 84, 'text': 'Terrible muscle'}],
                [{'start_offset': 103, 'end_offset': 138, 'text': 'Burning sensations in neck shoulders'}],
                [{'start_offset': 74, 'end_offset': 77, 'text': 'ible'},
                 {'start_offset': 86, 'end_offset': 99, 'text': 'and joint pain'}],
                [{'start_offset': 74, 'end_offset': 84, 'text': 'ible muscle'},
                 {'start_offset': 86, 'end_offset': 99, 'text': 'and joint pain'}],
                [{'start_offset': 103, 'end_offset': 123, 'text': 'Burning sensations in'},
                 {'start_offset': 144, 'end_offset': 154, 'text': 'upper chest'}],
                [{'start_offset': 140, 'end_offset': 142, 'text': 'and'},
                 {'start_offset': 150, 'end_offset': 154, 'text': 'chest'}],
                [{'start_offset': 70, 'end_offset': 77, 'text': 'Terrible'},
                 {'start_offset': 86, 'end_offset': 88, 'text': 'and'}],
            ],
        }
        input_text = 'I was in deniel that the statins would cause ME side effects - I have Terrible muscle and joint pain , Burning sensations in neck shoulders and upper chest .'
        self.token_remove = ['and']
        self.token_end_remove = ['in']
        print(self.run(input_text=input_text, ner_results=e, ner_keys=['NER_DRUG']))


class NERNormalizer:
    def __init__(self, input_normalizer: InputNormalizer = None, label_normalizer: LabelNormalizer = None):
        if input_normalizer:
            self.input_normalizer = input_normalizer
        else:
            self.input_normalizer = InputNormalizer()

        if label_normalizer:
            self.label_normalizer = label_normalizer
        else:
            self.label_normalizer = LabelNormalizer()

    def run(self, ner_sample, ner_keys, sentence_tokenizer=None, keep_features=None):
        if isinstance(ner_sample, list):
            for i, j in enumerate(ner_sample): j['run_sample_id'] = i
            if not keep_features:
                keep_features = ['run_sample_id']
            else:
                keep_features.append('run_sample_id')
            if sentence_tokenizer:
                ner_samples = []
                for i in ner_sample:
                    all_samples = split_sample_and_label(sample=i,
                                                         sentence_tokenizer=sentence_tokenizer,
                                                         ner_keys=ner_keys,
                                                         keep_features=keep_features)
                    for k, v in enumerate(all_samples): v['run_sentence_id'] = k
                    ner_samples.extend(all_samples)
                keep_features.append('run_sentence_id')
            else:
                ner_samples = ner_sample
        else:
            if sentence_tokenizer:
                ner_samples = split_sample_and_label(sample=ner_sample,
                                                     sentence_tokenizer=sentence_tokenizer,
                                                     ner_keys=ner_keys,
                                                     keep_features=keep_features)
                for i, j in enumerate(ner_samples): j['run_sentence_id'] = i
                if not keep_features:
                    keep_features = ['run_sentence_id']
                else:
                    keep_features.append('run_sentence_id')
            else:
                ner_samples = [ner_sample]

        from algorithms_ai.utils.parallel_process_utils.mpire_utils import apply_mpire
        refined_ner_samples = apply_mpire(func=self.process_one_sample,
                                          data_list=ner_samples,
                                          job_num=4,
                                          ner_keys=ner_keys)
        if keep_features:
            for i in range(len(refined_ner_samples)):
                for j in keep_features:
                    if j in ner_samples[i]:
                        refined_ner_samples[i][j] = ner_samples[i][j]

        return refined_ner_samples

    def process_one_sample(self, ner_sample, ner_keys):
        input_text = ner_sample['input_text']
        ner_results = {i: j for i, j in ner_sample.items() if i in ner_keys}
        refined_input_text, refined_ner_results, refined_mapping_raw, raw_mapping_refined = \
            self.input_normalizer.run(input_text=input_text,
                                      ner_results=ner_results,
                                      ner_keys=ner_keys)
        refined_ner_results = self.label_normalizer.run(input_text=refined_input_text,
                                                        ner_results=refined_ner_results,
                                                        ner_keys=ner_keys)
        refined_ner_sample = {'input_text': refined_input_text}
        refined_ner_sample.update(refined_ner_results)
        return refined_ner_sample

    def test(self):
        sample = {
            'input_text': 'Terrible first of all I had diarrhoea my doseage was increased to 75 and a few days ago my feet and ankles started swelling up.\nI went to my GP and the results of my blood test showed there was something wrong with my liver and I am a non-drinker.\nMy blood pressure had gone sky high so my that is why the swelling.\nI was told to stop taking Arthrotec and am now using a natural product that seems to work it is called Pernamax.\n',
            'NER_ADR': [[{'text': 'diarrhoea', 'start_offset': 28, 'end_offset': 36}],
                        [{'text': 'swelling', 'start_offset': 115, 'end_offset': 122},
                         {'text': 'feet', 'start_offset': 91, 'end_offset': 94}],
                        [{'text': 'wrong with my liver', 'start_offset': 204, 'end_offset': 222}],
                        [{'text': 'blood pressure had gone sky high',
                          'start_offset': 251,
                          'end_offset': 282}],
                        [{'text': 'ankles', 'start_offset': 100, 'end_offset': 105},
                         {'text': 'swelling', 'start_offset': 115, 'end_offset': 122}],
                        [{'text': 'swelling', 'start_offset': 306, 'end_offset': 313}]],
            'NER_Drug': [[{'text': 'Arthrotec', 'start_offset': 342, 'end_offset': 350}],
                         [{'text': 'Pernamax', 'start_offset': 419, 'end_offset': 426}]],
            'features': 1
        }
        sentence_tokenizer = RegexTokenizer(regex='\n').run
        print(self.run(sample, ['NER_ADR', 'NER_Drug'], keep_features=['features']))

        print(self.run([sample] * 2, ['NER_ADR', 'NER_Drug'], sentence_tokenizer=sentence_tokenizer,
                       keep_features=['features']))


if __name__ == '__main__':
    NERNormalizer().test()
