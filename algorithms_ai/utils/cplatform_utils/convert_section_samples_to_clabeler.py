import uuid
from datetime import datetime


def convert_section_samples_to_clabeler(section_samples, from_name_config,
                                        source_id, project_id,
                                        model_version='0.0.0', user_name='rule_tagger'):
    # 每个元素是一个section的结果
    assert 'input_text' in section_samples[0] and 'text_section' in section_samples[0]
    article_samples = dict()
    for section_sample in section_samples:
        if section_sample['_id'] not in article_samples:
            article_samples[section_sample['_id']] = [section_sample]
        else:
            article_samples[section_sample['_id']].append(section_sample)

    push_to_clabeler = []
    for _id, article_sample in article_samples.items():
        annotation_results = convert_article_sample(article_sample, from_name_config)
        push_to_clabeler.append(
            {
                "_id": f"{model_version}_{user_name}_{_id}_{project_id}_{source_id}",
                "doc_id": str(_id),
                "source_id": source_id,
                "project_id": project_id,
                "version": model_version,
                "user_name": user_name,
                "anno_type": "prediction",
                "anno_result": annotation_results,
                "create_time": datetime.utcnow().isoformat() + 'Z',
            }

        )
    return push_to_clabeler


def convert_article_sample(article_sample, from_name_config):
    annotation_results = []
    for section_sample in article_sample:
        for from_name, from_name_info in from_name_config.items():
            entities = section_sample.get(from_name_info['entity_type'], [])
            for entity in entities:
                if len(entity) == 1:
                    anno = convert_entity_part(
                        entity_part=entity[0],
                        text_section=section_sample['text_section'],
                        from_name=from_name,
                        label_zh_name=from_name_info['label_zh_name'],
                        linked_regions=[],
                        section_text=section_sample['input_text'],
                        ignored=False)
                    if anno: annotation_results.append(anno)
                else:
                    annos = []
                    checked_true = True
                    for entity_part in entity:
                        anno = convert_entity_part(
                            entity_part=entity_part,
                            text_section=section_sample['text_section'],
                            from_name=from_name,
                            label_zh_name=from_name_info['label_zh_name'],
                            linked_regions=[],
                            section_text=section_sample['input_text'],
                            ignored=True)
                        if anno:
                            annos.append(anno)
                        else:
                            checked_true = False
                    if checked_true:
                        linked_regions = [i['id'] for i in annos]
                        for i in annos: i['value']['linked_regions'] = linked_regions
                        annos[0]['value']['ignored'] = False
                        annos[0]['value']['text'] = ' '.join([i['value']['text'] for i in annos])
                        annotation_results.extend(annos)
    return annotation_results


def convert_entity_part(entity_part, text_section, from_name, label_zh_name,
                        linked_regions, section_text=None, ignored=False):
    if section_text:
        if section_text[entity_part['start_offset']:entity_part['end_offset'] + 1] != entity_part['text']:
            return None
    value = {
        "start_section": text_section,
        "end_section": text_section,
        "startOffset": entity_part['start_offset'],
        "endOffset": entity_part['end_offset'] + 1,
        "text": entity_part['text'],
        "labels": [label_zh_name],
        "ignored": ignored,
        "linked_regions": linked_regions,
        "pdf_position": None
    }
    return {
        "id": f"prediction.{uuid.uuid4()}",
        "from_name": from_name,
        "to_name": "doc",
        "type": "labels",
        "value": value,
        'meta': []
    }


if __name__ == '__main__':
    anno = [
        {
            'to_name': 'doc',
            'meta': [
                {
                    'field': 'name',
                    'index': 'discover_indication',
                    'text': '肾细胞癌',
                    'id': '157',
                    'type': 'es_table'
                }
            ],
            'id': 'NOSMjd4IdI',
            'from_name': 'clinical_trial.indications',
            'type': 'labels',
            'value': {
                'start_section': '["BriefTitle"]',
                'ignored': False,
                'endOffset': 95,
                'end_section': '["BriefTitle"]',
                'linked_regions': [
                ],
                'startOffset': 75,
                'text': 'Renal Cell Carcinoma',
                'labels': [
                    '适应症'
                ],
                'pdf_position': None
            }
        },
        {
            'to_name': 'doc',
            'meta': [
            ],
            'id': 'OkeS4l008Q',
            'from_name': 'clinical_trial.indications',
            'type': 'labels',
            'value': {
                'start_section': '["OfficialTitle"]',
                'ignored': False,
                'endOffset': 122,
                'end_section': '["OfficialTitle"]',
                'linked_regions': [
                ],
                'startOffset': 96,
                'text': 'Renal Cell Carcinoma (RCC)',
                'labels': [
                    '适应症'
                ],
                'pdf_position': None
            }
        },
        {
            'to_name': 'doc',
            'meta': [
            ],
            'id': '5eDEHgMdbE',
            'from_name': 'clinical_trial.indications',
            'type': 'labels',
            'value': {
                'start_section': '["BriefSummary"]',
                'ignored': False,
                'endOffset': 208,
                'end_section': '["BriefSummary"]',
                'linked_regions': [
                ],
                'startOffset': 182,
                'text': 'renal cell carcinoma (RCC)',
                'labels': [
                    '适应症'
                ],
                'pdf_position': None
            }
        },
        {
            'to_name': 'doc',
            'meta': [
            ],
            'id': 'hwSesEgU0a',
            'from_name': 'clinical_trial.indications',
            'type': 'labels',
            'value': {
                'start_section': '["BriefSummary"]',
                'ignored': False,
                'endOffset': 316,
                'end_section': '["BriefSummary"]',
                'linked_regions': [
                    'hwSesEgU0a',
                    'roKOPBs5A8'
                ],
                'startOffset': 306,
                'text': 'clear cell renal cell carcinoma',
                'labels': [
                    '适应症'
                ],
                'pdf_position': None
            }
        },
        {
            'to_name': 'doc',
            'meta': [
            ],
            'id': 'A_ibWopa2U',
            'from_name': 'clinical_trial.indications',
            'type': 'labels',
            'value': {
                'start_section': '["BriefSummary"]',
                'ignored': False,
                'endOffset': 355,
                'end_section': '["BriefSummary"]',
                'linked_regions': [
                ],
                'startOffset': 320,
                'text': 'non-clear cell renal cell carcinoma',
                'labels': [
                    '适应症'
                ],
                'pdf_position': None
            }
        },
        {
            'to_name': 'doc',
            'meta': [
            ],
            'id': 'roKOPBs5A8',
            'from_name': 'clinical_trial.indications',
            'type': 'labels',
            'value': {
                'start_section': '["BriefSummary"]',
                'ignored': True,
                'endOffset': 355,
                'end_section': '["BriefSummary"]',
                'linked_regions': [
                    'hwSesEgU0a',
                    'roKOPBs5A8'
                ],
                'startOffset': 335,
                'text': 'renal cell carcinoma',
                'labels': [
                    '适应症'
                ],
                'pdf_position': None
            }
        },
        {
            'to_name': 'doc',
            'meta': [
            ],
            'id': 'kCehYNNGE-',
            'from_name': 'clinical_trial.indications',
            'type': 'labels',
            'value': {
                'start_section': '["BriefSummary"]',
                'ignored': False,
                'endOffset': 362,
                'end_section': '["BriefSummary"]',
                'linked_regions': [
                ],
                'startOffset': 357,
                'text': 'ccRCC',
                'labels': [
                    '适应症'
                ],
                'pdf_position': None
            }
        },
        {
            'to_name': 'doc',
            'meta': [
            ],
            'id': 'ekiqlcTD7m',
            'from_name': 'clinical_trial.indications',
            'type': 'labels',
            'value': {
                'start_section': '["BriefSummary"]',
                'ignored': False,
                'endOffset': 372,
                'end_section': '["BriefSummary"]',
                'linked_regions': [
                ],
                'startOffset': 366,
                'text': 'nccRCC',
                'labels': [
                    '适应症'
                ],
                'pdf_position': None
            }
        },
        {
            'to_name': 'doc',
            'meta': [
            ],
            'id': '0ckGMbhwJm',
            'from_name': 'clinical_trial.indications',
            'type': 'labels',
            'value': {
                'start_section': '["BriefSummary"]',
                'ignored': False,
                'endOffset': 616,
                'end_section': '["BriefSummary"]',
                'linked_regions': [
                ],
                'startOffset': 577,
                'text': 'clear cell renal cell carcinoma (ccRCC)',
                'labels': [
                    '适应症'
                ],
                'pdf_position': None
            }
        }
    ]
    raw_es_data = [{'system_cstudio': [{
        'anno_result': anno,
        'update_time': '2023-11-13T01:07:04.771282Z', 'anno_type': 'annotation',
        'create_time': '2023-11-13T01:07:04.771282Z', 'project_id': 1, 'user_name': 'Published', 'ground_truth': True,
        'source_id': 2, 'id': 'guyev', 'doc_id': 'NCT05256472', 'version': '0.1.0'
    }],
        'BriefSummary': 'This is a Phase II, open-label trial to evaluate the efficacy and safety of AK104 monotherapy or AK104 in combination with axitinib as a first-line treatment for advanced/metastatic renal cell carcinoma (RCC). There are two parts in this trial. In part 1 of this study, subjects with unresectable advanced clear cell or non-clear cell renal cell carcinoma (ccRCC or nccRCC) who had not received systemic therapy for advanced disease will be enrolled to randomly received three different dosage of AK104 monotherapy. In part 2 of this study, subjects with unresectable advanced clear cell renal cell carcinoma (ccRCC) who had not received systemic therapy for advanced disease will be enrolled to receive AK104 plus Axitinib. All subjects will receive treatment until disease progression, development of unacceptable toxicity, death, a decision by the physician or patient to withdraw from the trial. The primary endpoint is ORR per RECIST v1.1 as assessed by investigators.',
        'BriefTitle': 'A Study of AK104 Monotherapy or AK104 Plus Axitinib in Advanced/Metastatic Renal Cell Carcinoma',
        '_id': 'NCT05256472'
    }]

    from process_cplatform_es_data import extract_ner_from_cplatform

    field_entity = [
        ('BriefTitle', ['clinical_trial.patient_labels', 'clinical_trial.indications']),
        ('OfficialTitle', ['clinical_trial.patient_labels', 'clinical_trial.indications']),
        ('BriefSummary', ['clinical_trial.patient_labels', 'clinical_trial.indications']),
        ('ArmsAndInterventions', ['clinical_trial.arms.arm_detail.arm_drug',
                                  'clinical_trial.arms.arm_detail.arm_label_character',
                                  'clinical_trial.arms.arm_detail.arm_label_indication']),
        ('InterventionList', ['clinical_trial.arms.arm_detail.arm_drug',
                              'clinical_trial.arms.arm_detail.arm_label_character',
                              'clinical_trial.arms.arm_detail.arm_label_indication'])
    ]

    section_samples = extract_ner_from_cplatform(raw_es_data, field_entity, anno_user='Published')
    # from_name_
    from_name_config_ = {
        'clinical_trial.indications': {
            'entity_type': 'clinical_trial.indications',
            'label_zh_name': '适应症'
        }

    }

    cplatform_config = [
        # (from_name,label_zh_name,label_id, link_index, link_key,)

    ]

    to_push = convert_section_sample_to_clabeler(section_samples=section_samples,
                                                 from_name_config=from_name_config_,
                                                 model_version='0.0.0', user_name='rule_tagger',
                                                 source_id=2,
                                                 project_id=1,
                                                 )
    from zyl_utils.utils.elastic_search_utils.elasticsearch8_utils import es_bulk_upload_data
    from zyl_utils.utils.elastic_search_utils.es_config import huawei_prod_client

    es_bulk_upload_data(es_client=huawei_prod_client(),
                        index_name='clabeler_annotations_new_new',
                        data=to_push,
                        action='index',
                        id_key='_id',
                        batch_size=500)

    # delete_by_id(es_client=huawei_prod_client(), index_name='clabeler_annotations_new_new',
    #              id="0.0.0_rule_tagger_NCT05256472_1_2")

    print(2)
