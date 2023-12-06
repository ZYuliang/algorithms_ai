def search_entity_index(entity, offsets_mapping, entity_repair_mode='complete'):
    # 根据原始坐标时映射的实体的offset_mapping，找到index

    start_offset_index = -1
    end_offset_index = -1

    start_offset = entity['start_offset']
    end_offset = entity['end_offset']

    for i, offsets in enumerate(offsets_mapping):
        if start_offset_index >= 0 and end_offset_index >= 0:
            break

        if start_offset_index == -1:
            if offsets[0] <= start_offset <= offsets[1]:
                if entity_repair_mode == 'delete':
                    if i != len(offsets_mapping) - 1:
                        start_offset_index = i + 1
                    else:
                        start_offset_index = i
                else:
                    start_offset_index = i
            elif i != 0 and offsets_mapping[i - 1][1] < start_offset < offsets[0]:
                start_offset_index = i
            elif i == 0 and 0 <= start_offset < offsets[0]:
                start_offset_index = i

        if start_offset_index >= 0 and end_offset_index == -1:
            if offsets[0] <= end_offset <= offsets[1]:
                if entity_repair_mode == 'delete':
                    if i != 0:
                        end_offset_index = i - 1
                    else:
                        end_offset_index = i
                else:
                    end_offset_index = i
            elif i != len(offsets_mapping) - 1 and offsets[1] < end_offset < offsets_mapping[i + 1][0]:
                end_offset_index = i
            elif i == len(offsets_mapping) - 1 and offsets[1] < end_offset:
                end_offset_index = i
    return start_offset_index, end_offset_index


def process_one_sample(sample: dict, pre_tokenizer, entities, label2id, keep_features=None):
    """
    # 'ner_tags': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 8, 8, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0],
    # 'tokens': ['@paulwalk', 'It', "'s", 'the', 'view', 'from', 'where', 'I', "'m", 'living', 'for', 'two', 'weeks', '.',
    #            'Empire', 'State', 'Building', '=', 'ESB', '.', 'Pretty', 'bad', 'storm', 'here', 'last', 'evening', '.']

    """
    input_text = sample['input_text']
    token_res = pre_tokenizer(input_text)
    tokens = token_res['texts']
    offsets_mapping = token_res['offsets_mapping']
    ner_tags = ['O'] * len(tokens)

    return_res = {
        'input_text': input_text,
        'tokens': tokens,
        # 'offsets_mapping':offsets_mapping
    }

    is_delete = False
    if entities:
        for entity in entities:

            for _ in sample.get(entity,[]):
                if len(_)>=2:
                    is_delete=True

            entity_res = [__ for _ in sample.get(entity, []) for __ in _]
            for each_entity_res in entity_res:
                start_offset_index, end_offset_index = search_entity_index(each_entity_res, offsets_mapping)
                if start_offset_index != -1 and end_offset_index != -1:
                    ner_tags[start_offset_index] = f'B-{entity}'
                    for entity_index in range(start_offset_index + 1, end_offset_index + 1):
                        ner_tags[entity_index] = f'I-{entity}'

        ner_tags = [label2id[i] for i in ner_tags]
        return_res['ner_tags'] = ner_tags

    if keep_features:
        return_res.update({i: j for i, j in sample.items() if i in keep_features})

    if not is_delete:
        return return_res
    else:
        return None


def tokenize_and_align_labels(examples, tokenizer=None):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True,return_offsets_mapping=True)

    if 'ner_tags' in examples:
        labels = []
        for i, label in enumerate(examples[f"ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:  # Set the special tokens to -100.
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            labels.append(label_ids)
        tokenized_inputs["labels"] = labels

    return tokenized_inputs


if __name__ == '__main__':
    s = {'input_text': '[OfficialTitle]:FONTANA: A Modular Phase I/IIa, Open-label, '
                       'Multi-center Study to Assess the Safety, Tolerability, Pharmacokinetics, '
                       'and Preliminary Efficacy of Ascending Doses of AZD5335 Monotherapy and in Combination With '
                       'Anti-cancer Agents in Participants With Solid Tumors',
         'NER_clinical_trial.phase': [[{'end_offset': 45, 'start_offset': 35, 'text': 'Phase I/IIa'}]],
         'NER_clinical_trial.bio_labels': [], 'NER_clinical_trial.clinical_stage': [],
         'NER_clinical_trial.pathology': [], 'NER_clinical_trial.patient_labels': [],
         'NER_clinical_trial.indications': [[{'end_offset': 275, 'start_offset': 264, 'text': 'Solid Tumors'}]],
         'NER_clinical_trial.therapy_labels': [], 'NER_clinical_trial.drug': [],
         'nct_id': 'NCT05797168', 'field': 'OfficialTitle', 'run_sample_id': 1, 'run_sentence_id': 0}

    from zyl_utils.utils.tokenizer_utils.regex_tokenizer import RegexTokenizer

    pre_tokenizer = RegexTokenizer().run

    # from transformers import AutoTokenizer
    # model_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    entities = ['NER_clinical_trial.indications', 'NER_clinical_trial.drug']
    label2id = {"O": 0}
    for i, j in enumerate(entities):
        label2id[f'B-{j}'] = 2 * (i + 1) - 1
        label2id[f'I-{j}'] = 2 * (i + 1)

    print(process_one_sample(s, pre_tokenizer, entities, label2id, keep_features=['nct_id']))
