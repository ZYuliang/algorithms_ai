"""
实体识别数据集的分析： 分不同实体类型，
1. 对样本实体的分析，包括实体是否存在嵌套，非连续的情况，以及样本实体数的分布,以及存在的最常见的和不常见的实体是什么
2. 常见的token和不常见的token

"""

from collections import Counter

import os
import pandas as pd

from algorithms_ai.dataset_utils.dataset_analysis.base_analysis import analyze_unordered_numerical_array


def compare_ner_results(old_results, new_results, entities, old_keep_features, result_xlsx='./results_compared.xlsx',
                        eval_mode=False):
    """
    对比两组模型结果的差异，主要用于判断当前模型对比上一版本的模型的改变：
    1. 改变主要在于每个样本的预测情况：新增和缺失和正确的实体，
    2. 整体样本的情况，新增和缺失实体所占的个数。排序
    :param old_results: 老版本
    :param new_results: 新版本
    :param entities: 要处理的实体类型
    :param old_keep_features:需要保留的特征，用于定位
    :return: 
    """

    assert len(old_results) == len(new_results)

    all_info = []

    if not old_keep_features:
        old_keep_features = ['tmp_sample_id']
        for i, _ in enumerate(old_results): _['tmp_sample_id'] = i

    all_missing_entities = {e: [] for e in entities}
    all_increased_entities = {e: [] for e in entities}

    for entity in entities:
        for old_res, new_res in zip(old_results, new_results):
            sub_info = {i: old_res.get(i) for i in old_keep_features}

            old_res = [[(each_part['start_offset'], each_part['end_offset'], each_part['text'])
                        for each_part in each_entity] for each_entity in old_res.get(entity, [])]
            new_res = [[(each_part['start_offset'], each_part['end_offset'], each_part['text'])
                        for each_part in each_entity] for each_entity in new_res.get(entity, [])]

            missing_entities = []
            increased_entities = []
            right_entities = []

            for i in old_res:
                if i not in new_res:
                    missing_entities.append(i)
                else:
                    right_entities.append(i)

            for i in new_res:
                if i not in old_res:
                    increased_entities.append(i)

            sub_info['entity_type'] = entity
            sub_info['missing_entities'] = missing_entities
            sub_info['increased_entities'] = increased_entities
            sub_info['right_entities'] = right_entities

            if not increased_entities and not missing_entities:
                is_right = True
            else:
                is_right = False
            sub_info['is_right'] = is_right

            all_info.append(sub_info)

            for i in missing_entities:
                all_missing_entities[entity].append('|'.join([j[-1].lower() for j in i]))

            for i in increased_entities:
                all_increased_entities[entity].append('|'.join([j[-1].lower() for j in i]))

    summary_info = []
    for entity in entities:
        missing_summary = sorted(Counter(all_missing_entities[entity]).items(), key=lambda x: x[-1], reverse=True)

        for i in missing_summary:
            summary_info.append(
                {
                    'entity_type': entity,
                    'entity_error': 'missing',
                    'entity_text': i[0],
                    'entity_num': i[-1]
                }
            )

        increased_summary = sorted(Counter(all_increased_entities[entity]).items(), key=lambda x: x[-1], reverse=True)
        for i in increased_summary:
            summary_info.append(
                {
                    'entity_type': entity,
                    'entity_error': 'increased',
                    'entity_text': i[0],
                    'entity_num': i[-1]
                }
            )

    all_info = pd.DataFrame(all_info)
    summary_info = pd.DataFrame(summary_info)

    if not os.path.exists(os.path.dirname(result_xlsx)):
        os.mkdir(os.path.dirname(result_xlsx))

    with pd.ExcelWriter(result_xlsx) as writer:
        all_info.to_excel(writer, sheet_name="detail", index=False)
        summary_info.to_excel(writer, sheet_name="entity_summary", index=False)

        if eval_mode:
            from algorithms_ai.deep_learning.metric.ner_metric import entity_recognition_metric_for_data
            eval_res = entity_recognition_metric_for_data(old_results, new_results, entities,
                                                          return_score_key=['micro_precision', 'micro_recall',
                                                                            'micro_f1',
                                                                            'micro_score', 'accuracy'])
            eval_res = pd.DataFrame([{'evaluation_standard': i, 'score': j} for i, j in eval_res.items()])
            eval_res.to_excel(writer, sheet_name="score", index=False)


def ner_analysis(ner_samples=None, tokenizer=None, entities=None):
    entity_info = collect_entity_info(ner_samples=ner_samples, tokenizer=tokenizer, entities=entities)

    for entity in entities:
        print('#' * 50)
        print(f'分析实体：{entity}')
        print(
            f'嵌套实体样本占比:{round(sum(entity_info[entity]["has_nested"]) / len(entity_info[entity]["has_nested"]), 4)}')
        print(
            f'非连续实体样本占比:{round(sum(entity_info[entity]["has_discontinuous"]) / len(entity_info[entity]["has_discontinuous"]), 4)}')
        print(f'样本实体数量的分布：')
        analyze_unordered_numerical_array(entity_info[entity]["entity_num"])

        all_entities = [j.lower() for i in entity_info[entity]["entities"] for j in i]

        all_entities_sorted = sorted(Counter(all_entities).items(), key=lambda x: x[-1], reverse=True)
        print(f'最常用前20个实体:{all_entities_sorted[0:20]}')

        ##
        if tokenizer:
            all_tokens = [k.lower() for i in entity_info[entity]["entity_tokens"] for j in i for k in j]
            all_tokens_sorted = sorted(Counter(all_tokens).items(), key=lambda x: x[-1], reverse=True)
            print(f'最常用前20个token:{all_tokens_sorted[0:20]}')

            all_tokens_start = [j[0].lower() for i in entity_info[entity]["entity_tokens"] for j in i]
            all_tokens_start_sorted = sorted(Counter(all_tokens_start).items(), key=lambda x: x[-1], reverse=True)
            print(f'最常用起始前20个token:{all_tokens_start_sorted[0:20]}')

            all_tokens_end = [j[-1].lower() for i in entity_info[entity]["entity_tokens"] for j in i]
            all_tokens_end_sorted = sorted(Counter(all_tokens_end).items(), key=lambda x: x[-1], reverse=True)
            print(f'最常用结尾前20个token:{all_tokens_end_sorted[0:20]}')


def collect_entity_info(ner_samples=None, tokenizer=None, entities=None):
    entity_info = {
        i: {'entity_num': [], 'has_nested': [], 'has_discontinuous': [], 'entities': [], 'entity_tokens': []} for i
        in entities}

    for entity in entities:
        for ner_sample in ner_samples:

            entity_res = ner_sample.get(entity, [])

            has_discontinuous = 0
            for i in entity_res:
                if len(i) > 1:
                    has_discontinuous = 1
            entity_info[entity]['has_discontinuous'].append(has_discontinuous)

            entity_info[entity]['entity_num'].append(len(entity_res))

            has_nested = 0
            sorted_entity_res = sorted([j for i in entity_res for j in i],
                                       key=lambda x: (x['start_offset'], x['end_offset']))
            for i in range(len(sorted_entity_res) - 1):
                if sorted_entity_res[i]['end_offset'] >= sorted_entity_res[i + 1]['start_offset']:
                    has_nested = 1

            entity_info[entity]['has_nested'].append(has_nested)

            entities = ['|'.join([j['text'] for j in i]) for i in entity_res]
            entity_info[entity]['entities'].append(entities)

            if tokenizer:
                entity_tokens = []

                for i in entities:
                    tokened_res = tokenizer(i)
                    if 'texts' in tokened_res:
                        entity_tokens.append(tokened_res['texts'])
                    else:
                        if 'input_ids' in tokened_res:
                            entity_tokens.append(tokenizer.convert_ids_to_tokens(tokened_res['input_ids']))

                entity_info[entity]['entity_tokens'].append(entity_tokens)

    return entity_info


if __name__ == '__main__':
    d = [
        {'clinical_trial.patient_labels':
             [[{'start_offset': 192, 'end_offset': 223, 'text': 'Progressed on Standard Treatment'},
               {'start_offset': 139, 'end_offset': 147, 'text': 'Recurrent'}],
              [{'start_offset': 139, 'end_offset': 147, 'text': 'Recurrent'}]],
         'clinical_trial.phase': [[{'start_offset': 2, 'end_offset': 9, 'text': 'Phase II'}]],
         'clinical_trial.indications': [[{'start_offset': 163, 'end_offset': 186, 'text': 'Nasopharyngeal Carcinoma'}]],
         'clinical_trial.clinical_stage': [[{'start_offset': 122, 'end_offset': 137, 'text': 'Locally Advanced'}],
                                           [{'start_offset': 152, 'end_offset': 161, 'text': 'Metastatic'}]],
         'clinical_trial.pathology': [[{'start_offset': 79, 'end_offset': 103, 'text': 'Moderately Differentiated'}],
                                      [{'start_offset': 105, 'end_offset': 120, 'text': 'Undifferentiated'}]],
         'input_text': 'A Phase II, Open-label, Randomized Controlled Study of PDR001 in Patients With Moderately Differentiated/Undifferentiated Locally Advanced Recurrent or Metastatic Nasopharyngeal Carcinoma Who Progressed on Standard Treatment',
         'features': {'field': 'OfficialTitle', 'nct_id': 'NCT02605967', 'phaselist': ['Phase 2'],
                      'conditionlist': ['Nasopharyngeal Carcinoma']}, 'nct_id': 'NCT02605967',
         'field': 'OfficialTitle'}, \
        {'clinical_trial.patient_labels': [[{'start_offset': 44, 'end_offset': 52, 'text': 'Pediatric'}]],
         'clinical_trial.therapy_labels': [[{'start_offset': 95, 'end_offset': 104, 'text': 'First-Line'}]],
         'clinical_trial.indications': [[{'start_offset': 78, 'end_offset': 83, 'text': 'Glioma'}]],
         'clinical_trial.pathology': [[{'start_offset': 68, 'end_offset': 76, 'text': 'Low-Grade'}]],
         'input_text': 'DAY101 vs. Standard of Care Chemotherapy in Pediatric Patients With Low-Grade Glioma Requiring First-Line Systemic Therapy (LOGGIC/FIREFLY-2)',
         'features': {'field': 'BriefTitle',
                      'nct_id': 'NCT05566795', 'phaselist': ['Phase 3'], 'conditionlist': ['Low-grade Glioma']},
         'nct_id': 'NCT05566795', 'field': 'BriefTitle'}, \
        {'clinical_trial.phase': [[{'start_offset': 20, 'end_offset': 26, 'text': 'Phase 3'}]],
         'clinical_trial.patient_labels': [[{'start_offset': 149, 'end_offset': 157, 'text': 'Pediatric'}]],
         'clinical_trial.bio_labels': [
             [{'start_offset': 176, 'end_offset': 213, 'text': 'Harboring an Activating RAF Alteration'}]],
         'clinical_trial.therapy_labels': [[{'start_offset': 225, 'end_offset': 234, 'text': 'First-Line'}]],
         'clinical_trial.indications': [[{'start_offset': 169, 'end_offset': 174, 'text': 'Glioma'}]],
         'clinical_trial.pathology': [[{'start_offset': 159, 'end_offset': 167, 'text': 'Low-Grade'}]],
         'input_text': 'LOGGIC/FIREFLY-2: A Phase 3, Randomized, International Multicenter Trial of DAY101 Monotherapy Versus Standard of Care Chemotherapy in Patients With Pediatric Low-Grade Glioma Harboring an Activating RAF Alteration Requiring First-Line Systemic Therapy',
         'features': {'field': 'OfficialTitle', 'nct_id': 'NCT05566795', 'phaselist': ['Phase 3'],
                      'conditionlist': ['Low-grade Glioma']}, 'nct_id': 'NCT05566795', 'field': 'OfficialTitle'}, \
        {'clinical_trial.phase': [[{'start_offset': 62, 'end_offset': 68, 'text': 'Phase 3'}]],
         'clinical_trial.patient_labels': [[{'start_offset': 209, 'end_offset': 217, 'text': 'pediatric'}]],
         'clinical_trial.therapy_labels': [[{'start_offset': 326, 'end_offset': 335, 'text': 'front-line'}]],
         'clinical_trial.indications': [[{'start_offset': 229, 'end_offset': 234, 'text': 'glioma'}]],
         'clinical_trial.pathology': [[{'start_offset': 219, 'end_offset': 227, 'text': 'low-grade'}]],
         'clinical_trial.bio_labels': [[{'start_offset': 242, 'end_offset': 314,
                                         'text': 'harboring an activating rapidly accelerated fibrosarcoma (RAF) alteration'}]],
         'input_text': 'This is a 2-arm, randomized, open-label, multicenter, global, Phase 3 trial to evaluate the efficacy, safety, and tolerability of DAY101 monotherapy versus standard of care (SoC) chemotherapy in patients with pediatric low-grade glioma (LGG) harboring an activating rapidly accelerated fibrosarcoma (RAF) alteration requiring front-line systemic therapy.',
         'features': {'field': 'BriefSummary', 'nct_id': 'NCT05566795', 'phaselist': ['Phase 3'],
                      'conditionlist': ['Low-grade Glioma']}, 'nct_id': 'NCT05566795', 'field': 'BriefSummary'}
    ]

    from algorithms_ai.utils.tokenizer_utils.regex_tokenizer import RegexTokenizer

    ner_analysis(ner_samples=d, tokenizer=RegexTokenizer(), entities=['clinical_trial.patient_labels',
                                                                      'clinical_trial.indications',
                                                                      'clinical_trial.clinical_stage'
                                                                      ])
    #
    # old_results = [{
    #     'input_text': 'sdf',
    #     'id': 2,
    #     'entity_a': [
    #         [{'start_offset': 1, 'end_offset': 4, 'text': 'har'}, {'start_offset': 8, 'end_offset': 12, 'text': 'har'}]
    #     ],
    #     'entity_b': [
    #         [{'start_offset': 1, 'end_offset': 4, 'text': 'har'}, {'start_offset': 8, 'end_offset': 12, 'text': 'har'}]
    #     ]
    #
    # },
    #     {
    #         'input_text': 'adfh',
    #
    #         'id': 0,
    #         'entity_a': [
    #             [{'start_offset': 1, 'end_offset': 4, 'text': 'har'},
    #              {'start_offset': 8, 'end_offset': 12, 'text': 'har'}]
    #         ],
    #         'entity_b': [
    #             [{'start_offset': 2, 'end_offset': 4, 'text': 'har'}]
    #         ]
    #     }
    # ]
    #
    # new_results = [
    #     {
    #         'entity_a': [
    #             [{'start_offset': 1, 'end_offset': 4, 'text': 'har'},
    #              {'start_offset': 8, 'end_offset': 12, 'text': 'har'}]
    #         ],
    #         'entity_b': [
    #             [{'start_offset': 1, 'end_offset': 4, 'text': 'har'},
    #              {'start_offset': 8, 'end_offset': 12, 'text': 'har'}]
    #         ]
    #     },
    #     {
    #         'entity_a': [
    #             [{'start_offset': 1, 'end_offset': 4, 'text': 'har'},
    #              {'start_offset': 8, 'end_offset': 12, 'text': 'har'}]
    #         ],
    #         'entity_b': [
    #             [{'start_offset': 1, 'end_offset': 4, 'text': 'har'},
    #              {'start_offset': 8, 'end_offset': 12, 'text': 'har'}]
    #         ]
    #     }
    # ]
    # entities = ['entity_a', 'entity_b']
    # old_keep_features = ['input_text', 'id']
    # compare_ner_results(old_results, new_results, entities, old_keep_features, result_xlsx='./r.xlsx')
