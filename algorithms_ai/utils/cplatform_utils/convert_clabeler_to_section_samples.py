from loguru import logger
from tqdm import tqdm


def convert_clabeler_to_section_samples(es_data, field_mapping_from_names, anno_user='Published'):
    # 把clabeler中的es数据转为模型要用的格式,选择哪些人的标注,只选择一个人的标注，如果不同人有不同的标注，就在外面设置优先级
    # field_mapping_from_names 字段--字段中要抽取的from_nams: [ (filed_name,from_names)]
    # field_name:es数据中第一层级的字段名字，from_names es数据中annotations中要抽取的实体名称
    # 输出的字段：_id，input_text，text_section，一些from_names对应的结果
    section_samples = []

    for article_sample in tqdm(es_data, desc='process es'):
        all_annotations = [i for i in article_sample['system_cstudio'] if i['user_name'] == anno_user]
        if len(all_annotations) == 0:
            logger.info(f'es_data:{article_sample["_id"]} not has annotation')
            continue
        elif len(all_annotations) > 1:
            logger.info(f'es_data:{article_sample["_id"]} has one more annotations')
            continue
        annotations = all_annotations[0]['anno_result']

        for field_name, from_names in field_mapping_from_names:
            if from_names:
                from_name_annotations = [annotation for annotation in annotations if
                                         annotation['from_name'] in from_names]
            else:
                from_name_annotations = annotations

            input_texts = article_sample.get(field_name)
            if not input_texts:
                continue

            if isinstance(input_texts, str):
                text_section = f'["{field_name}"]'
                section_sample = extract_section_sample_from_annotation(from_name_annotations,
                                                                        text_section,
                                                                        input_texts)
                if section_sample:
                    section_sample.update({'_id': article_sample.get('_id')})
                    section_samples.append(section_sample)
                else:
                    logger.error(f'error id:{article_sample.get("_id")}')
            elif isinstance(input_texts, list):
                for index, sub_field in enumerate(input_texts):
                    for sub_field_name, sub_field_text in sub_field.items():
                        text_section = f'["{field_name}"][{index}]["{sub_field_name}"]'
                        section_sample = extract_section_sample_from_annotation(from_name_annotations,
                                                                                text_section,
                                                                                sub_field_text)
                        if section_sample:
                            section_sample.update({'_id': article_sample.get('_id')})
                            section_samples.append(section_sample)
                        else:
                            logger.error(f'error id:{article_sample.get("_id")}')
    return section_samples


def extract_section_sample_from_annotation(annotations, text_section, input_text):
    # 从一组标注中根据字段和文本提取实体
    section_sample = dict()
    section_sample['input_text'] = input_text
    section_sample['text_section'] = text_section
    try:
        section_annotations = [annotation for annotation in annotations
                               if text_section in annotation['value']['start_section']
                               and text_section in annotation['value']['end_section']]

        linked_annotations = {i['id']: i for i in section_annotations if i['value']['ignored']}  # 跳跃的实体标注
        true_annotations = [i for i in section_annotations if not i['value']['ignored']]
        for true_annotation in true_annotations:
            entity = true_annotation['value']

            if not entity['linked_regions']:
                assert input_text[entity['startOffset']:entity['endOffset']] == entity['text']
                refined_entities = [{
                    'start_offset': entity['startOffset'],
                    'end_offset': entity['endOffset'] - 1,
                    'text': input_text[entity['startOffset']:entity['endOffset']]
                }]
            else:
                refined_entities = []
                for linked_id in entity['linked_regions']:
                    if linked_id == true_annotation['id']:
                        refined_entities.append({
                            'start_offset': entity['startOffset'],
                            'end_offset': entity['endOffset'] - 1,
                            'text': input_text[entity['startOffset']:entity['endOffset']]
                        })
                    else:
                        entity = linked_annotations[linked_id]['value']
                        refined_entities.append({
                            'start_offset': entity['startOffset'],
                            'end_offset': entity['endOffset'] - 1,
                            'text': input_text[entity['startOffset']:entity['endOffset']]
                        })
            if true_annotation['from_name'] not in section_sample:
                section_sample[true_annotation['from_name']] = [refined_entities]
            else:
                section_sample[true_annotation['from_name']].append(refined_entities)
        return section_sample
    except Exception as e:
        logger.info(f'error extract_section_sample_from_annotation,error type:{e},annotations:{annotations},'
                    f'text_section:{text_section}, input_text:{input_text}')
        return None


if __name__ == '__main__':
    #     field_entitiy = [
    #         ('BriefTitle', ['clinical_trial.patient_labels', 'clinical_trial.indications']),
    #         ('OfficialTitle', ['clinical_trial.patient_labels', 'clinical_trial.indications']),
    #         ('BriefSummary', ['clinical_trial.patient_labels', 'clinical_trial.indications']),
    #         ('ArmsAndInterventions', ['clinical_trial.arms.arm_detail.arm_drug',
    #                                   'clinical_trial.arms.arm_detail.arm_label_character',
    #                                   'clinical_trial.arms.arm_detail.arm_label_indication']),
    #         ('InterventionList', ['clinical_trial.arms.arm_detail.arm_drug',
    #                               'clinical_trial.arms.arm_detail.arm_label_character',
    #                               'clinical_trial.arms.arm_detail.arm_label_indication'])
    #
    #     ]
    #     model_res = extract_ner_from_cplatform(es_data=es_data,
    #                                            field_entitiy=field_entitiy,
    #                                            anno_users='Published',
    #                                            keep_features=['_id'])

    a = [{'to_name': 'doc',
          'meta': [{'field': 'name', 'index': 'discover_indication', 'text': '实体瘤', 'id': '133', 'type': 'es_table'}],
          'id': 'H_8Xfiwp1d', 'from_name': 'clinical_trial.indications', 'type': 'labels',
          'value': {'start_section': '["OfficialTitle"]', 'ignored': False, 'endOffset': 195,
                    'end_section': '["OfficialTitle"]', 'linked_regions': [], 'startOffset': 183,
                    'text': 'Solid Tumors', 'labels': ['适应症'], 'pdf_position': None}}, {'to_name': 'doc', 'meta': [
        {'field': 'name', 'index': 'discover_indication', 'text': '实体瘤', 'id': '133', 'type': 'es_table'}],
                                                                                        'id': 'jbyaz-O_AS',
                                                                                        'from_name': 'clinical_trial.indications',
                                                                                        'type': 'labels', 'value': {
            'start_section': '["BriefSummary"]', 'ignored': False, 'endOffset': 190, 'end_section': '["BriefSummary"]',
            'linked_regions': [], 'startOffset': 178, 'text': 'solid tumors', 'labels': ['适应症'], 'pdf_position': None}},
         {'to_name': 'doc', 'meta': [], 'id': 'C8dXWl6E-E', 'from_name': 'clinical_trial.patient_labels',
          'type': 'labels', 'value': {'start_section': '["OfficialTitle"]', 'ignored': False, 'endOffset': 159,
                                      'end_section': '["OfficialTitle"]', 'linked_regions': [], 'startOffset': 152,
                                      'text': 'Chinese', 'labels': ['患者基线特征'], 'pdf_position': None}},
         {'to_name': 'doc', 'meta': [], 'id': 'YLxlcpkxQa', 'from_name': 'clinical_trial.patient_labels',
          'type': 'labels', 'value': {'start_section': '["BriefSummary"]', 'ignored': False, 'endOffset': 150,
                                      'end_section': '["BriefSummary"]', 'linked_regions': [], 'startOffset': 143,
                                      'text': 'Chinese', 'labels': ['患者基线特征'], 'pdf_position': None}}]
    b = '["BriefSummary"]'
    c = 'This was a dose verification, pharmacokinetic (PK) assessment of products derived from two manufacturing processes and scales (500L-FMP and 2000L-FMP; FMP: Final Manufacturing Process) and indication expansion clinical study of monoclonal antibody conducted in Chinese subjects with advanced solid tumors, with a purpose of exploring the safety, tolerability, pharmacokinetics and preliminary efficacy.'
    id = 'NCT04068519'
    print(extract_section_sample_from_annotation(a, b, c))
