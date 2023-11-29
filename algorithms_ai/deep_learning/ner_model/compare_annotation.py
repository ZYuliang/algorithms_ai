"""
对比两组模型结果的差异，主要用于判断当前模型对比上一版本的模型的改变

"""
from collections import Counter

import os


def compare_annotation_for_ner(last_annotations, current_annotations, entity_types, sample_ids=None, log_dir='./log'):
    """
    输入两个标注结果进行对比，必须含文本和实体
    :param last_annotations:
    :param current_annotations:
    :param entity_types:
    :param sample_ids:
    :param log_dir:
    :return:
    l_n = [[{'text': 'a', 'label': 'b'}, {'text': 'a', 'label': 'a'}, {'text': 'a', 'label': 'b'},
            {'text': 'a', 'label': 'b'}],
           [{'text': 'a', 'label': 'b'}, {'text': 'a', 'label': 'a'}, {'text': 'a', 'label': 'b'},
            {'text': 'a', 'label': 'b'}]]
    c_n = [[{'text': 'cd', 'label': 'b'}, {'text': 'a2', 'label': 'a'}, {'text': 'a', 'label': 'b'},
            {'text': 'd', 'label': 'a'}],
           [{'text': 'a', 'label': 'b'}, {'text': 'a', 'label': 'a'}, {'text': 'a', 'label': 'b'},
            {'text': 'a', 'label': 'b'}]]
    compare_annotation_for_ner(l_n, c_n, entity_types=['a', 'b'])
    """
    results = dict()
    for entity_type in entity_types:
        results[entity_type] = {
            'increased_entities': [],
            'missing_entities': [],
            'same_entities': []
        }
    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)
    detail_file = os.path.join(log_dir, 'detail.txt')
    summary_file = os.path.join(log_dir, 'summary.txt')
    write_detail = open(detail_file, 'w', encoding='utf8').write
    write_summary = open(summary_file, 'w', encoding='utf8').write
    write_detail(f'样本id\t实体类型\t错误类型\t具体实体\n')
    write_summary(f'实体类型\t错误类型\t具体实体\t数量\n')

    if not sample_ids:
        sample_ids = list(range(len(last_annotations)))

    for last_annotation, current_annotation, sample_id in zip(last_annotations, current_annotations, sample_ids):
        for entity_type in entity_types:
            sub_last = [i['text'] for i in last_annotation if i['label'] == entity_type]
            sub_current = [i['text'] for i in current_annotation if i['label'] == entity_type]

            sub_missing = []
            for i in sub_last:
                if i not in sub_current:
                    sub_missing.append(i)
                    results[entity_type]['missing_entities'].append(i)
                else:
                    results[entity_type]['same_entities'].append(i)
            sub_increased = []
            for i in sub_current:
                if i not in sub_last:
                    sub_increased.append(i)
                    results[entity_type]['increased_entities'].append(i)
            if sub_missing:
                write_detail(f'{sample_id}\t{entity_type}\t漏标\t{set(sub_missing)}\n')
            if sub_increased:
                write_detail(f'{sample_id}\t{entity_type}\t新增\t{set(sub_increased)}\n')

    for entity_type in entity_types:
        summary = f'实体类型：{entity_type}, 同样：{len(results[entity_type]["same_entities"])} ,' \
                  f'漏标：{len(results[entity_type]["missing_entities"])}, ' \
                  f'新增：{len(results[entity_type]["increased_entities"])}\n'
        write_summary(summary)

        missing_entities = sorted(dict(Counter(results[entity_type]["missing_entities"])).items(), key=lambda x: x[1],
                                  reverse=True)
        increased_entities = sorted(dict(Counter(results[entity_type]["increased_entities"])).items(),
                                    key=lambda x: x[1], reverse=True)

        for i in missing_entities:
            write_summary(f'{entity_type}\t漏标\t{i[0]}\t{i[1]}\n')

        for i in increased_entities:
            write_summary(f'{entity_type}\t新增\t{i[0]}\t{i[1]}\n')

    return results


if __name__ == '__main__':
    l_n = [[{'text': 'a', 'label': 'b'}, {'text': 'a', 'label': 'a'}, {'text': 'a', 'label': 'b'},
            {'text': 'a', 'label': 'b'}],
           [{'text': 'a', 'label': 'b'}, {'text': 'a', 'label': 'a'}, {'text': 'a', 'label': 'b'},
            {'text': 'a', 'label': 'b'}]]
    c_n = [[{'text': 'cd', 'label': 'b'}, {'text': 'a2', 'label': 'a'}, {'text': 'a', 'label': 'b'},
            {'text': 'd', 'label': 'a'}],
           [{'text': 'a', 'label': 'b'}, {'text': 'a', 'label': 'a'}, {'text': 'a', 'label': 'b'},
            {'text': 'a', 'label': 'b'}]]
    r = compare_annotation_for_ner(l_n, c_n, entity_types=['a', 'b'])
    print(r)
