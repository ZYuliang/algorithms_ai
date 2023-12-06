def get_entity_by_bio(bio_res):
    last_label = 'O'

    all_entities = []
    each_entity = []

    entity_type = None
    for index, current_label in enumerate(bio_res):
        if last_label == 'O':
            if current_label[0] == 'B':

                each_entity.append(index)
                entity_type = current_label[2:]
            elif current_label[0] == 'I':
                pass
        elif last_label[0] == 'B':
            if current_label == 'O':
                if each_entity:
                    all_entities.append((tuple(each_entity), entity_type))
                each_entity = []
            elif current_label[0] == 'B':
                if each_entity:
                    all_entities.append((tuple(each_entity), entity_type))
                each_entity = []
                each_entity.append(index)
                entity_type = current_label[2:]
            else:
                if current_label[2:] == last_label[2:]:
                    each_entity.append(index)
                else:
                    if each_entity:
                        all_entities.append((tuple(each_entity), entity_type))
                    each_entity = []
        else:
            if current_label == 'O':
                if each_entity:
                    all_entities.append((tuple(each_entity), entity_type))
                each_entity = []
            elif current_label[0] == 'B':  # 另一个实体
                if each_entity:
                    all_entities.append((tuple(each_entity), entity_type))
                each_entity = []
                each_entity.append(index)
                entity_type = current_label[2:]
            else:  # 'I'
                if current_label[2:] == last_label[2:]:
                    if each_entity:
                        each_entity.append(index)
                else:
                    if each_entity:
                        all_entities.append((tuple(each_entity), entity_type))
                    each_entity = []
        last_label = current_label

    if each_entity:
        all_entities.append((tuple(each_entity), entity_type))

    return all_entities


if __name__ == '__main__':
    s = ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
         'B-NER_clinical_trial.indications', 'I-NER_clinical_trial.indications', 'I-NER_clinical_trial.indications',
         'I-NER_clinical_trial.indications', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
         'B-NER_clinical_trial.indications', 'I-NER_clinical_trial.indications', 'I-NER_clinical_trial.indications',
         'I-NER_clinical_trial.indications', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
         'O', 'O', 'O', 'I-NER_clinical_trial.indications', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
         'O', 'O', 'O', 'O', 'O', 'B-NER_clinical_trial.indications', 'I-NER_clinical_trial.indications',
         'I-NER_clinical_trial.indications', 'I-NER_clinical_trial.indications', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
         'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-NER_clinical_trial.indications', 'I-NER_clinical_trial.indications',
         'I-NER_clinical_trial.indications', 'I-NER_clinical_trial.indications', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
         'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'I-NER_clinical_trial.indications',
         'I-NER_clinical_trial.indications', 'I-NER_clinical_trial.indications', 'O', 'O', 'O']

    print(get_entity_by_bio(s))
