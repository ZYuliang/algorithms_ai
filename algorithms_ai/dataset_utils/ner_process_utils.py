"""
实体识别数据集样本处理
包括：
数据集的切分，句子切分：split_sample_and_label
输入样本的规范化：SampleNormalizer
实体标签结果的规范化：LabelNormalizer

"""
from copy import deepcopy

from loguru import logger

from algorithms_ai.utils.parallel_process_utils.mpire_utils import apply_mpire
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


def split_sample_to_sentence_samples(sample, sentence_tokenizer=None, entity_types=None, keep_features=None):
    # 对一组结果和对应的标签进行分句
    input_text = sample['input_text']

    sentence_samples = []
    if sentence_tokenizer:
        if keep_features:
            features = {i: j for i, j in sample.items() if i in keep_features}
        else:
            features = dict()

        sentence_token_result = sentence_tokenizer(input_text)
        offsets_mapping = sentence_token_result['offsets_mapping']
        texts = sentence_token_result['texts']

        if entity_types:
            sentence_samples_dict = {i: {'input_text': texts[i], 'sentence_split_offsets': offsets_mapping[i]}
                                     for i in range(len(texts))}
            for _, sentence_sample in sentence_samples_dict.items():
                sentence_sample.update(features)
                sentence_sample.update({entity_type: [] for entity_type in entity_types})

            for entity_type in entity_types:
                entities = sample.get(entity_type, [])
                for entity in entities:
                    seq_indexs = [find_seq_index((entity_part['start_offset'], entity_part['end_offset']),
                                                 offsets_mapping) for entity_part in entity]
                    if -1 in seq_indexs or len(set(seq_indexs)) > 1:  # 确保一个实体只在一个输入（一句话）中出现
                        return []
                    seq_index = seq_indexs[0]
                    entity = [{'text': entity_part['text'],
                               'start_offset': entity_part['start_offset'] - offsets_mapping[seq_index][0],
                               'end_offset': entity_part['end_offset'] - offsets_mapping[seq_index][0]}
                              for entity_part in entity]
                    sentence_samples_dict[seq_index][entity_type].append(entity)
            sentence_samples.extend(sentence_samples_dict.values())
        return sentence_samples
    else:
        return [sample]


def split_sample_with_sliding_window(sample, model_tokenizer=None, sliding_window=0.2, entity_types=None,
                                     keep_features=None):
    """
    根据模型tokenizer使用sliding-window进行处理
    返回samples
    """
    if not model_tokenizer:
        return [sample]
    input_text = sample["input_text"]
    tokenized_info = model_tokenizer(input_text, return_offsets_mapping=True)
    if model_tokenizer.model_max_length > 100000 or len(
            tokenized_info['input_ids']) <= model_tokenizer.model_max_length:
        return [sample]

    char_state = ['O' for _ in range(len(input_text))]  # 存储每个字符是否可以作为开头或结尾
    for _, offsets in model_tokenizer._tokenizer.pre_tokenizer.pre_tokenize_str(input_text):
        char_state[offsets[0]] = 's'
        if char_state[offsets[-1] - 1] == 's':
            char_state[offsets[-1] - 1] = 's/e'
        else:
            char_state[offsets[-1] - 1] = 'e'

    # if entity_types:
    #     all_entity_span = []
    #     for entity_type in entity_types:
    #         for entity in sample.get(entity_type, []):
    #             all_entity_span.append((entity[0]['start_offset'], entity[-1]['end_offset']))
    #     if len(all_entity_span) > 1:
    #         all_entity_span = sorted(all_entity_span, key=lambda x: (x[0], x[1]))
    #         refined_all_entity_span = []
    #         last_entity_span = all_entity_span[0]
    #         for s, e in all_entity_span[1:]:
    #             if s > last_entity_span[1]:
    #                 refined_all_entity_span.append(last_entity_span)
    #                 last_entity_span = (s, e)
    #             elif s <= last_entity_span[1]:
    #                 last_entity_span = (last_entity_span[0], max(last_entity_span[1], e))
    #         refined_all_entity_span.append(last_entity_span)
    #         all_entity_span = refined_all_entity_span
    #     else:
    #         all_entity_span = all_entity_span
    #     for entity_start, entity_end in all_entity_span:
    #         char_state[entity_start:entity_end + 1] = ['O'] * (entity_end - entity_start + 1)
    #         char_state[entity_start] = 's'
    #         if char_state[entity_end] == 's':
    #             char_state[entity_end] = 's/e'
    #         else:
    #             char_state[entity_end] = 'e'

    token_spans = [i for i in tokenized_info['offset_mapping'] if i != (0, 0)]
    max_token_length = model_tokenizer.model_max_length - len(model_tokenizer.all_special_tokens)  # 有效token数
    sliding_window_token_length = max(int(model_tokenizer.model_max_length * sliding_window), 1)

    all_token_span_sample = []
    token_span_sample = []
    for each_token_span in token_spans:
        token_span_sample.append(each_token_span)
        if len(token_span_sample) > max_token_length:
            while token_span_sample and ('e' not in char_state[token_span_sample[-1][0]:token_span_sample[-1][1]] and
                                         's/e' not in char_state[token_span_sample[-1][0]:token_span_sample[-1][1]]):
                token_span_sample = token_span_sample[0:-1]

            while token_span_sample and ('s' not in char_state[token_span_sample[-1][0]:token_span_sample[-1][1]] and
                                         's/e' not in char_state[token_span_sample[-1][0]:token_span_sample[-1][1]]):
                token_span_sample = token_span_sample[1:]
            all_token_span_sample.append(deepcopy(token_span_sample))
            token_span_sample = token_span_sample[-sliding_window_token_length - 1:-1]

    if token_span_sample:
        all_token_span_sample.append(token_span_sample)

    refined_samples = []
    for token_span_sample in all_token_span_sample:
        if not token_span_sample:
            continue
        start = token_span_sample[0][0]
        end = token_span_sample[-1][-1]

        sub_sample = dict()
        sub_sample['input_text'] = input_text[start:end]
        sub_sample['sliding_window_offsets'] = (start, end - 1)
        if entity_types:
            for entity_type in entity_types:
                sub_sample[entity_type] = [
                    [{'start_offset': entity_part['start_offset'] - start,
                      'end_offset': entity_part['end_offset'] - start,
                      'text': entity_part['text']}
                     for entity_part in entity]
                    for entity in sample.get(entity_type, [])
                    if (start <= entity[0]['start_offset'] <= entity[-1]['end_offset'] <= end)
                ]
        if keep_features:
            for i in keep_features: sub_sample[i] = sample.get(i)
        refined_samples.append(sub_sample)
    return refined_samples


def add_prefixes_to_model_samples(model_samples, entity_types, prefixes, keep_features=None):
    refined_model_samples = []
    assert len(model_samples) == len(prefixes)
    for model_sample, prefix in zip(model_samples, prefixes):
        input_text = model_sample['input_text']
        entities_result = {entity_type: model_sample.get(entity_type, []) for entity_type in entity_types}
        entities_result = {
            entity_type: [
                [{'start_offset': entity_part['start_offset'] + len(prefix),
                  'end_offset': entity_part['end_offset'] + len(prefix),
                  'text': entity_part['text']} for entity_part in entity]
                for entity in entities
            ]
            for entity_type, entities in entities_result.items()
        }
        refined_model_sample = {'input_text': prefix + input_text}
        refined_model_sample['prefix'] = prefix
        refined_model_sample.update(entities_result)
        if keep_features:
            refined_model_sample.update({i: model_sample[i] for i in keep_features if i in model_sample})
        refined_model_samples.append(refined_model_sample)
    return refined_model_samples


def update_keep_features(keep_features, add_feature):
    if not keep_features:
        keep_features = [add_feature]
    else:
        keep_features.append(add_feature)
    keep_features = list(set(keep_features))
    return keep_features


def order_entity_part_in_one_sample(sample, entity_types):
    if not entity_types:
        return sample
    for entity_type in entity_types:
        sample[entity_type] = [
            sorted(entity, key=lambda entity_part: (entity_part['start_offset'], entity_part['end_offset']))
            for entity in sample.get(entity_type, [])]


def check_sample(sample, entity_types=None):
    # 检查样本是否实体正确
    if not entity_types:
        return True

    input_text = sample['input_text']
    for entity_type in entity_types:
        for entity in sample.get(entity_type, []):
            for entity_part in entity:
                if input_text[entity_part['start_offset']:entity_part['end_offset'] + 1] != entity_part['text']:
                    return False
    return True


class InputTextNormalizer:
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

    def run(self, input_text, entities_result=None, entity_types=()):
        refined_input_text, index_refined_mapping_raw, index_raw_mapping_refined = self.normalize_text(input_text)

        if entities_result and entity_types:
            entities_result = {entity_type: [[{'text': refined_input_text[
                                                       index_raw_mapping_refined[entity_part['start_offset']][0]:
                                                       index_raw_mapping_refined[entity_part['end_offset']][-1] + 1],
                                               'start_offset': index_raw_mapping_refined[entity_part['start_offset']][
                                                   0],
                                               'end_offset': index_raw_mapping_refined[entity_part['end_offset']][-1],
                                               } for entity_part in entity]
                                             for entity in entities_result.get(entity_type, [])]
                               for entity_type in entity_types}

        return refined_input_text, entities_result, index_refined_mapping_raw, index_raw_mapping_refined

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

            if (last_end_index < offsets[0] and index != tokens_list_len - 1 and index != 0) \
                    or (index == tokens_list_len - 1 and last_end_index < offsets[0]):
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
        print(self.run(input_text=sample['input_text'], entities_result=sample, entity_types=['NER_ADR']))


class EntitiesResultNormalizer:
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

    def run(self, input_text, entities_result, entity_types, entity_repair_mode='complete'):
        # entity_repair_mode表示截断实体是要补全还是去除
        input_text = input_text
        input_text_tokens = self.regex_tokenizer(input_text)
        texts_list = input_text_tokens['texts']
        offsets_mapping = input_text_tokens['offsets_mapping']

        for entity_type in entity_types:
            entities_result[entity_type] = self.normalize_entities(input_text,
                                                                   entities_result.get(entity_type, []),
                                                                   texts_list,
                                                                   offsets_mapping,
                                                                   entity_repair_mode=entity_repair_mode)
        return entities_result

    def normalize_entities(self, input_text, entities, texts_list, offsets_mapping,
                           entity_repair_mode='complete'):
        # 规范一个标签下的所有实体，交叉重复的实体去除
        entities_index = []
        for entity in entities:
            normalized_entity = self.normalize_entity(entity, texts_list, offsets_mapping,
                                                      entity_repair_mode=entity_repair_mode)
            if normalized_entity:
                entities_index.append(normalized_entity)

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
                        if entity_start_offset == offsets[0]:
                            refined_start_offset_index = i
                        elif entity_start_offset > offsets[0]:
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
                        if entity_end_offset == offsets[1]:
                            refined_end_offset_index = i
                        elif entity_end_offset < offsets[1]:
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
                    entity_end_index < len(texts_list) - 1 and texts_list[entity_end_index + 1] == right_punc and \
                    entity_start_index < len(texts_list) - 1:
                if texts_list[entity_start_index] == left_punc:
                    entity_start_index = entity_start_index + 1

            if sub_text_list.count(right_punc) == 1 and sub_text_list.count(left_punc) == 0 and \
                    entity_start_index > 0 and texts_list[
                entity_start_index - 1] == left_punc and entity_end_index >= 1:
                if texts_list[entity_end_index] == right_punc:
                    entity_end_index = entity_end_index - 1

            if sub_text_list.count(right_punc) == 1 and sub_text_list.count(left_punc) == 1 and \
                    texts_list[entity_start_index] == left_punc and texts_list[entity_end_index] == right_punc:
                entity_end_index = entity_end_index - 1
                entity_start_index = entity_start_index + 1

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

        r = {
            'input_text': 'Up to 10 PTCL patients will receive nanatinostat 20 mg orally once daily, days 1-4 per week.',
            'NER_clinical_trial.indications': [[{'start_offset': 9, 'end_offset': 11, 'text': 'PTC'}]]}
        e = r
        input_text = r['input_text']
        # self.token_remove = ['and']
        # self.token_end_remove = ['in']
        print(self.run(
            input_text=input_text, entities_result=e, entity_types=['NER_clinical_trial.indications'],
            entity_repair_mode='complete'
        ))


class SamplesNormalizer:
    def __init__(self,
                 input_text_normalizer: InputTextNormalizer = None,
                 entities_result_normalizer: EntitiesResultNormalizer = None,
                 sentence_tokenizer=None,
                 model_tokenizer=None,
                 sliding_window=0.2):
        if input_text_normalizer:
            self.input_text_normalizer = input_text_normalizer
        else:
            self.input_text_normalizer = InputTextNormalizer()

        if entities_result_normalizer:
            self.entities_result_normalizer = entities_result_normalizer
        else:
            self.entities_result_normalizer = EntitiesResultNormalizer()
        self.sentence_tokenizer = sentence_tokenizer
        self.model_tokenizer = model_tokenizer
        self.sliding_window = sliding_window

    def run(self, samples, entity_types, keep_features=None, return_addition_info=False):
        if not isinstance(samples, list):
            samples = [samples]

        is_right_sample = [check_sample(sample=sample, entity_types=entity_types) for sample in samples]  # 检查样本是否正确
        run_samples = []
        for sample_index, sample in enumerate(samples):
            if is_right_sample[sample_index]:
                sample['raw_sample_id'] = sample_index
                run_samples.append(sample)
            else:
                logger.error(f'error sample index:{sample_index},input_text:{sample["input_text"]}')
        keep_features = update_keep_features(keep_features, 'raw_sample_id')

        model_sampless = apply_mpire(func=self.process_raw_sample,
                                     data_list=run_samples,
                                     job_num=4,
                                     entity_types=entity_types,
                                     sentence_tokenizer=self.sentence_tokenizer,
                                     model_tokenizer=self.model_tokenizer,
                                     sliding_window=self.sliding_window,
                                     keep_features=keep_features,
                                     return_addition_info=return_addition_info
                                     )
        return [j for i in model_sampless for j in i]

    def process_raw_sample(self, sample, entity_types, sentence_tokenizer=None, model_tokenizer=None,
                           sliding_window=0.2, keep_features=None, return_addition_info=False):
        """
        # 输入原始文本或原始文本+结果，输出处理后的文本+处理后的结果
        # 1. 先切句---分成若干样本, run_sample_id,sentence_split_offsets
        # 2. 再进一步切model-max-length和结果,model
        # 3. 再预处理数据和结果--input_normalizer
        """
        order_entity_part_in_one_sample(sample, entity_types)
        sentence_samples = split_sample_to_sentence_samples(sample=sample,
                                                            sentence_tokenizer=sentence_tokenizer,
                                                            entity_types=entity_types,
                                                            keep_features=keep_features)
        keep_features = update_keep_features(keep_features, 'sentence_split_offsets')

        model_samples = []
        for sentence_sample in sentence_samples:
            model_sample = split_sample_with_sliding_window(sample=sentence_sample,
                                                            model_tokenizer=model_tokenizer,
                                                            sliding_window=sliding_window,
                                                            entity_types=entity_types,
                                                            keep_features=keep_features)
            model_samples.extend(model_sample)
        if model_tokenizer:
            keep_features = update_keep_features(keep_features, 'sliding_window_offsets')

        model_samples = [self.process_model_sample(model_sample=model_sample,
                                                   entity_types=entity_types,
                                                   keep_features=keep_features,
                                                   return_addition_info=return_addition_info,
                                                   entity_repair_mode='complete'
                                                   ) for model_sample in model_samples]
        return model_samples

    def process_model_sample(self, model_sample, entity_types, keep_features=None, return_addition_info=False,
                             entity_repair_mode='complete'):
        # 处理要输入模型的单个样本
        input_text = model_sample['input_text']
        entities_result = {i: j for i, j in model_sample.items() if i in entity_types}
        model_input_text, model_entities_result, refined_mapping_raw, raw_mapping_refined = \
            self.input_text_normalizer.run(input_text=input_text,
                                           entities_result=entities_result,
                                           entity_types=entity_types)

        model_entities_result = self.entities_result_normalizer.run(input_text=model_input_text,
                                                                    entities_result=model_entities_result,
                                                                    entity_types=entity_types,
                                                                    entity_repair_mode=entity_repair_mode)
        refined_model_sample = {'input_text': model_input_text}
        refined_model_sample.update(model_entities_result)
        if keep_features:
            refined_model_sample.update({i: model_sample.get(i) for i in keep_features})
        if return_addition_info:
            refined_model_sample.update(
                {
                    'refined_mapping_raw': refined_mapping_raw,
                    'raw_mapping_refined': raw_mapping_refined
                }
            )

        return refined_model_sample

    def postprocess_model_samples(self, raw_samples, model_samples, entity_types, keep_feature=None):
        # 输入原始文本和处理后的文本和模型预测的结果，输出原始文本上的标签
        # 模型的预测结果，不仅包含模型的预测，还要包含原始的一些信息，把原始信息覆盖到模型结果中
        assert 'refined_mapping_raw' in model_samples[0]
        assert 'raw_sample_id' in model_samples[0]
        raw_samples_dict = dict()  # 对于每个原始样本，收集模型结果的信息
        for model_sample in model_samples:
            refined_mapping_raw = model_sample['refined_mapping_raw']

            if model_sample['raw_sample_id'] not in raw_samples_dict:
                raw_samples_dict[model_sample['raw_sample_id']] = {entity_type: [] for entity_type in entity_types}
            for entity_type in entity_types:
                for entity in model_sample.get(entity_type, []):
                    sliding_window_start = model_sample.get('sliding_window_offsets')
                    if not sliding_window_start:
                        sliding_window_start = 0
                    else:
                        sliding_window_start = sliding_window_start[0]
                    sentence_split_start = model_sample.get('sentence_split_offsets')
                    if not sentence_split_start:
                        sentence_split_start = 0
                    else:
                        sentence_split_start = sentence_split_start[0]

                    raw_start_end = [(refined_mapping_raw[entity_part['start_offset']][
                                          0] + sliding_window_start + sentence_split_start,
                                      refined_mapping_raw[entity_part['end_offset']][
                                          -1] + sliding_window_start + sentence_split_start
                                      ) for entity_part in entity]
                    if raw_start_end not in raw_samples_dict[model_sample['raw_sample_id']][entity_type]:
                        raw_samples_dict[model_sample['raw_sample_id']][entity_type].append(raw_start_end)

        refined_samples = []
        for sample_index, raw_sample in enumerate(raw_samples):
            sample = {'input_text': raw_sample['input_text']}

            for entity_type in entity_types:
                entities = raw_samples_dict.get(sample_index, dict()).get(entity_type, [])
                entities = [[{'text': raw_sample['input_text'][entity_part[0]:entity_part[1] + 1],
                              'start_offset': entity_part[0],
                              'end_offset': entity_part[1]}
                             for entity_part in entity] for entity in entities]
                sample[entity_type] = entities

            if keep_feature:
                for i in keep_feature: sample[i] = raw_sample.get(i)
            refined_samples.append(sample)
        return refined_samples

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

        from algorithms_ai.deep_learning.ner_bio.run_ner_bio import get_tokenizer

        tokenizer = get_tokenizer(
            # checkpoint="/large_files/pretrained_pytorch/microsoft_BiomedNLP-BiomedBERT-large-uncased-abstract",
            checkpoint='/large_files/pretrained_pytorch/pubmed_bert_base_cased/',
            model_max_length=50
        )
        # sample = {'input_text': sample['input_text']}
        self.input_text_normalizer = InputTextNormalizer(is_en_text_lower=True)

        # print(self.preprocess_one_raw_sample(sample, ['NER_ADR', 'NER_Drug'],
        #                                      sentence_tokenizer=sentence_tokenizer,
        #                                      model_tokenizer=tokenizer,
        #                                      keep_features=['features']))
        ner_samples = [sample] * 2
        ner_samples.append(
            {'input_text': 'dasgadg'}
        )
        for i in ner_samples: order_entity_part_in_one_sample(i, ['NER_ADR', 'NER_Drug'])

        self.sentence_tokenizer = sentence_tokenizer
        self.model_tokenizer = tokenizer
        self.sliding_window = 0.2

        ner_sample_for_model = self.run(ner_samples, ['NER_ADR', 'NER_Drug'],
                                        keep_features=['features'],
                                        return_addition_info=True
                                        )
        print(ner_sample_for_model)

        res = self.postprocess_model_samples(
            raw_samples=ner_samples,
            model_samples=ner_sample_for_model,
            entity_types=['NER_ADR', 'NER_Drug'],
            keep_feature=None
        )
        print(res)


if __name__ == '__main__':
    InputTextNormalizer().test()
    EntitiesResultNormalizer().test()
    SamplesNormalizer().test()
