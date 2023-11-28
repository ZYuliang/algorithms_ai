# 规范化label
# 对于得到的一些标签结果，规范化标签，主要处理边界，对token进行处理

from algorithms_ai.utils.tokenizer_utils.regex_tokenizer import RegexTokenizer


class LabelNormalizer:
    def __init__(self,
                 token_start_add=(),
                 token_end_add=(),
                 token_start_remove=(),
                 token_end_remove=(),
                 token_remove=()
                 ):
        # 所有文本都小写，而且每个判断的条件和其他并不会有关联，如，加p> 和 > ,由于是break模式，所以写的时候要写全还要兼顾顺序，写【’p>‘,'>'】
        # 主要处理实体中的一些介词和边界，
        self.regex_tokenizer = RegexTokenizer().run  # 通用的切分word，理论上每个label都应该是这个切分的子集

        self.token_start_add = token_start_add  # 在标签前面添加
        self.token_end_add = token_end_add  # 在标签后面添加

        self.token_start_remove = token_start_remove  # 删除标签前缀
        self.token_end_remove = token_end_remove  # 删除标签后缀

        self.token_remove = token_remove  # 删除指定文本

        self.linking_punc = ['-', '_', '-', '-']  # token间的链接符号

    def run(self, input_text, ner_results, ner_labels):
        input_text = input_text
        input_text_tokens = self.regex_tokenizer(input_text)
        texts_list = input_text_tokens['texts']
        offsets_mapping = input_text_tokens['offsets_mapping']

        for ner_label in ner_labels:
            label_entities = ner_results.get(ner_label, [])
            refined_label_entities = self.normalize_label_entities(input_text, label_entities,
                                                                   texts_list, offsets_mapping)
            ner_results[ner_label] = refined_label_entities
        return ner_results

    def normalize_label_entities(self, input_text, label_entities, texts_list, offsets_mapping):
        # 规范一个标签下的所有实体，交叉重复的实体去除
        entities_index = []
        for entity in label_entities:
            entities_index.append(self.normalize_entity(entity, texts_list, offsets_mapping))

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

    def normalize_entity(self, entity, texts_list, offsets_mapping):
        # 规范化一个实体,规范化单独的part，交叉重复的part去除,输出这个实体由哪几个token组成，输出token的index
        entity_index = []
        for entity_part in entity:
            start_end_offset_index = self.normalize_entity_part(entity_part, texts_list,
                                                                offsets_mapping,
                                                                entity_repair_mode='complete')
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
        print(self.run(input_text=input_text, ner_results=e, ner_labels=['NER_DRUG']))


if __name__ == '__main__':
    LabelNormalizer().test()
