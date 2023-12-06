import numpy as np

from algorithms_ai.deep_learning.information_extraction.logits_process_ie import process_entity_logits, group_entities
from algorithms_ai.deep_learning.metric.ner_metric import entity_recognition_metric

from algorithms_ai.utils.time_utils.time_utils import set_timeout

def after_timeout():  # 超时后的处理函数
    print('time out!')
    return {'sum_micro_score':0,'sum_micro_f1':0,'accuracy':0}

@set_timeout(20, after_timeout)
def metric_for_ie(eval_pred, entity2id, relation2id=None, threshold=0):
    # eval_pred 若include_inputs_for_metrics=True则包含inputs
    # eval_pred:inputs 输入的input_ids(b,max_seq),
    # label_ids：输入的数据的标签,data_collator时构建,在 list[label1[b,1,*,2],label2,label3,label4]，
    # predictions模型输出的除了loss外的其他东西（tuple）,和label_ids的格式一一对应

    # ignored_token_ids 需要忽略的token，不在inputs的原始文本中，是额外加的token

    all_score_dict = dict()

    if entity2id:
        input_ids_mask = eval_pred.predictions[-1]
        ignored_token_ids = [0, -100]
        ignored_token_index_dict = dict()
        for i, j in zip(*np.where(np.isin(input_ids_mask, ignored_token_ids))):
            if i not in ignored_token_index_dict:
                ignored_token_index_dict[i] = {j}
            else:
                ignored_token_index_dict[i].add(j)

        entity_pred_logits = eval_pred.predictions[0]
        entity_labels = eval_pred.label_ids[0]
        entity_scores_dict = evaluate_entity_score(entity_pred_logits, entity_labels, ignored_token_index_dict,
                                                   threshold,
                                                   entity2id,
                                                   return_score_key=('micro_score','micro_f1', 'accuracy'))
        all_score_dict.update(entity_scores_dict)

    if relation2id:
        pass

    return all_score_dict


def evaluate_entity_score(entity_pred_logits, entity_labels, ignored_token_index_dict, threshold, entity2id,
                          return_score_key=('micro_score', 'accuracy')):
    pred_entities = process_entity_logits(entity_pred_logits, ignored_token_index_dict, threshold=threshold)

    max_sample_num, max_entity_num, _, _ = entity_labels.shape
    label_entities = group_entities(entity_labels, max_sample_num, max_entity_num, ignored_token_index_dict)

    id2entity = {j: i for i, j in entity2id.items()}
    entity_scores_dict = dict()
    for i in range(max_entity_num):
        print(f'\n entity_type:{id2entity[i]} :')
        each_entity_type_preds = []
        each_entity_type_labels = []
        for j in range(max_sample_num):
            each_entity_type_preds.append(pred_entities[j][i])
            each_entity_type_labels.append(label_entities[j][i])
        entity_scores_dict[id2entity[i]] = entity_recognition_metric(y_true=each_entity_type_labels,
                                                                     y_pred=each_entity_type_preds,
                                                                     return_score_key=return_score_key)
    entity_scores_dict['sum'] = dict()
    for k in return_score_key:
        entity_scores_dict['sum'][k] = round(
            sum([entity_scores_dict[i][k] for i, j in entity2id.items()]) / len(id2entity), 4)
    entity_scores_dict = {i + '_' + p: q for i, j in entity_scores_dict.items() for p, q in j.items()}
    return entity_scores_dict
