"""
原始数据的处理，变成标注格式

"""
from information_extraction.utils_ie import sequence_padding


def search_token_index(offset_mapping, char_index):
    # 根据原文本中字符坐标找到input_id中的坐标
    for refined_index, i in enumerate(offset_mapping):
        if i[0] != i[1]:
            if i[0] <= char_index < i[1]:
                return refined_index
    return -1


def process_one_sample(sample, tokenizer, entity2id=None, relation2id=None):
    # 处理标注格式的数据
    text = sample["input_text"]
    encoder_text = tokenizer(text, return_offsets_mapping=True, truncation=True)
    input_ids_mask = [0 if i in tokenizer.all_special_ids else 1 for i in encoder_text['input_ids']]
    offset_mapping = encoder_text['offset_mapping']

    entity_labels = []

    relation_labels = []
    head_labels = []
    tail_labels = []
    if entity2id:
        entity2id = sorted(entity2id.items(), key=lambda x: x[-1])
        entity_labels = [set() for _ in range(len(entity2id))]
        for sort_id, entity_info in enumerate(entity2id):
            entity_type, entity_id = entity_info[0], entity_info[1]
            assert entity_id == sort_id
            entity_res = sample.get(entity_type)
            for each_res in entity_res:
                token_index = set()
                for each_mention in each_res:
                    entity_index_start = search_token_index(offset_mapping, each_mention['start_offset'])
                    entity_index_end = search_token_index(offset_mapping, each_mention['end_offset'])
                    if entity_index_end != -1 and entity_index_end != -1:
                        token_index.update(list(range(entity_index_start, entity_index_end + 1)))
                if len(token_index) == 1:  # 一个token就是单独的一个实体
                    token_index = list(token_index)
                    entity_labels[entity_id].add((token_index[0], token_index[0]))

                elif len(token_index) >= 2:  # 需要有序
                    token_index = list(token_index)
                    token_index = sorted(token_index)
                    for each_index_id in range(0, len(token_index) - 1):
                        for each_index_id2 in range(each_index_id + 1, len(token_index)):
                            if token_index[each_index_id] < token_index[each_index_id2]:
                                entity_labels[entity_id].add((token_index[each_index_id], token_index[each_index_id2]))

        # 用（0，0）补足
        for each_label in entity_labels:
            if not each_label:
                each_label.add((0, 0))
        entity_labels = sequence_padding([list(i) for i in entity_labels]) if entity2id else []

    if relation2id:
        relation_labels = [set() for _ in range(2)]  # 关系约束
        head_labels = [set() for _ in range(len(relation2id))]
        tail_labels = [set() for _ in range(len(relation2id))]

        for i in sample.get('relations', []):
            subject_head = search_token_index(offset_mapping, i['subject']['char_span'][0])
            subject_tail = search_token_index(offset_mapping, i['subject']['char_span'][1])
            object_head = search_token_index(offset_mapping, i['object']['char_span'][0])
            object_tail = search_token_index(offset_mapping, i['object']['char_span'][1])
            if subject_head == -1 or subject_tail == -1 or object_head == -1 or object_tail == -1:
                continue

            if entity2id:
                entity_labels[entity2id[i['subject']['label']]].add((subject_head, subject_tail))
                entity_labels[entity2id[i['object']['label']]].add((object_head, object_tail))
            relation_labels[0].add((subject_head, subject_tail))
            relation_labels[1].add((object_head, object_tail))
            head_labels[relation2id[i['predicate']]].add((subject_head, object_head))
            tail_labels[relation2id[i['predicate']]].add((subject_tail, object_tail))

        for each_label in relation_labels + head_labels + tail_labels:
            if not each_label:
                each_label.add((0, 0))
        relation_labels = sequence_padding([list(i) for i in relation_labels]) if relation2id else []
        head_labels = sequence_padding([list(i) for i in head_labels]) if relation2id else []
        tail_labels = sequence_padding([list(i) for i in tail_labels]) if relation2id else []

    return {
        'input_text': text,
        # 'entities': {i:j for i,j in sample.items() if i.lower().startswith('ner')},
        # 'relations': sample.get('relations', []),

        'input_ids': encoder_text["input_ids"],
        'input_ids_mask': input_ids_mask,
        'attention_mask': encoder_text["attention_mask"],
        'token_type_ids': encoder_text["token_type_ids"],
        'offset_mapping': encoder_text["offset_mapping"],
        'labels': [entity_labels, relation_labels, head_labels, tail_labels],
    }
