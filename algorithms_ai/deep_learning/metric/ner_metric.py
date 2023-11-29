def get_f1(true_positive, false_negative, false_positive):
    precision = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    recall = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1


def entity_recognition_metric(y_true: list, y_pred: list, pos_neg_ratio: str = None, return_score_key=None):
    """the metric of entity_recognition, reference: https://docs.qq.com/doc/DYXRYQU1YbkVvT3V2
    实体识别指标，输入一个list，list中每个元素是一个对象（可以是一句或一段或一篇）的实体结果，每个对象的实体结果是一个set，
    这个set中每个实体用唯一值来映射,一般坐标,坐标要排序,若要指定实体类别，前面加id-
    这个f1的计算是仅考虑有实体的样本的准确度，如果原始句子中没有实体而且预测句子中没有实体，它所有相关指标是不会变化的
    Args:
        y_true: list[set[str]],the list of true target texts,each element is a set
        y_pred: list[set[str]],the list of pred target texts,each element is a set
        pos_neg_ratio: the ratio of positive and negative sample importance, default: the ratio of positive and
                       negative sample sizes, you can set it,like"7:3"

    Returns:
        show report and res
    """

    neg_data = 0
    neg_correct_dt = 0
    neg_wrong_dt = 0
    neg_redundant_entities = 0

    pos_data = 0
    pos_correct_dt = 0
    pos_wrong_dt = 0
    pos_correct_entities = 0
    pos_wrong_entities = 0
    pos_omitted_entities = 0
    pos_redundant_entities = 0

    all_macro_scores = []

    true_positive = []
    false_negative = []
    false_positive = []

    macro_precision = []
    macro_recall = []
    macro_f1 = []

    for i, j in zip(y_true, y_pred):
        # sub_total = len(i | j)
        sub_true_positive = len(j & i)
        true_positive.append(sub_true_positive)
        sub_false_negative = len(j - i)
        false_negative.append(sub_false_negative)
        sub_false_positive = len(i - j)
        false_positive.append(sub_false_positive)
        # sub_true_negative = (sub_total - sub_true_positive - sub_false_negative - sub_false_positive)
        sub_macro_precision, sub_macro_recall, sub_macro_f1 = get_f1(sub_true_positive, sub_false_negative,
                                                                     sub_false_positive)

        macro_precision.append(sub_macro_precision)
        macro_recall.append(sub_macro_recall)
        macro_f1.append(sub_macro_f1)

        if i == set():
            neg_data += 1
            if j == set():
                neg_correct_dt += 1
                all_macro_scores.append(1)
            else:
                neg_wrong_dt += 1
                neg_redundant_entities += len(j)
                all_macro_scores.append(0)
        else:
            pos_data += 1
            true_pred = len(i & j)
            pos_correct_entities += true_pred

            if i == j:
                pos_correct_dt += 1
                all_macro_scores.append(1)
            elif len(i) >= len(j):
                pos_wrong_dt += 1
                pos_wrong_entities += (len(j) - true_pred)
                pos_omitted_entities += (len(i) - len(j))
                all_macro_scores.append(true_pred / len(i))
            else:
                pos_wrong_dt += 1
                pos_redundant_entities += (len(j) - len(i))
                pos_wrong_entities += (len(i) - true_pred)
                all_macro_scores.append(true_pred / len(j))

    score_precision = 4

    all_pos_entities = pos_correct_entities + pos_wrong_entities + pos_omitted_entities + pos_redundant_entities
    if neg_data == 0:
        neg_metric = 0
    else:
        neg_metric = round(neg_correct_dt / (neg_correct_dt + neg_redundant_entities), score_precision)
    if pos_data == 0:
        pos_metric = 0
    else:
        pos_metric = round(pos_correct_entities / all_pos_entities, score_precision)

    micro_score = round((pos_correct_entities + neg_correct_dt) / (
            neg_correct_dt + neg_redundant_entities + all_pos_entities), score_precision)
    macro_score = round(sum(all_macro_scores) / len(y_true), score_precision)

    # 正负样本的比重
    if pos_neg_ratio:
        pos_all = float(pos_neg_ratio.split(':')[0])
        neg_all = float(pos_neg_ratio.split(':')[1])
        pos_ratio = pos_all / (pos_all + neg_all)
        neg_ratio = neg_all / (pos_all + neg_all)
    else:
        pos_ratio = pos_data / (pos_data + neg_data)
        neg_ratio = neg_data / (pos_data + neg_data)

    weighted_score = round(pos_ratio * pos_metric + neg_ratio * neg_metric, score_precision)

    tp = pos_correct_dt
    fn = pos_wrong_dt
    fp = neg_wrong_dt
    tn = neg_correct_dt

    accuracy = round((tp + tn) / (tp + fn + fp + tn), score_precision)

    macro_precision = round(sum(macro_precision) / len(y_true), score_precision)
    macro_recall = round(sum(macro_recall) / len(y_true), score_precision)
    macro_f1 = round(sum(macro_f1) / len(y_true), score_precision)

    micro_precision, micro_recall, micro_f1 = get_f1(sum(true_positive), sum(false_negative), sum(false_positive))
    micro_precision = round(micro_precision, score_precision)
    micro_recall = round(micro_recall, score_precision)
    micro_f1 = round(micro_f1, score_precision)

    results = [
        ['positive_data', pos_data, pos_correct_dt, pos_wrong_dt, pos_correct_entities,
         pos_wrong_entities, pos_omitted_entities, pos_redundant_entities, pos_metric],
        ['negative_data', neg_data, neg_correct_dt, neg_wrong_dt, '-', '-', '-', neg_redundant_entities, neg_metric],
        ['all_data', pos_data + neg_data, neg_correct_dt + pos_correct_dt, neg_wrong_dt + pos_wrong_dt,
         pos_correct_entities, pos_wrong_entities, pos_omitted_entities,
         pos_redundant_entities + neg_redundant_entities,
         ''],
        ['micro_precision', '', '', '', '', '', '', '', micro_precision],
        ['micro_recall', '', '', '', '', '', '', '', micro_recall],
        ['micro_f1', '', '', '', '', '', '', '', micro_f1],

        ['macro_precision', '', '', '', '', '', '', '', macro_precision],
        ['macro_recall', '', '', '', '', '', '', '', macro_recall],
        ['macro_f1', '', '', '', '', '', '', '', macro_f1],

        ['micro_score', '', '', '', '', '', '', '', micro_score],
        ['macro_score', '', '', '', '', '', '', '', macro_score],
        ['weighted_score', '', '', '', '', '', '', '', weighted_score],
        ['accuracy', '', '', '', '', '', '', '', accuracy],
    ]

    index = [' ' * 17, '| data_num ', '| correct_data ', '| wrong_data ', '| correct_entities ', '| wrong_entities ',
             '| omitted_entities ', '| redundant_entities ', '|  score']

    print(''.join(index))
    for r in results:
        if return_score_key:
            if r[0] in return_score_key or r[0] in ['positive_data', 'negative_data', 'all_data']:
                print(''.join(
                    [str(i).center(len(j)) if 'score' not in j else str(i).rjust(len(j)) for i, j in zip(r, index)]))
        else:
            print(''.join(
                [str(i).center(len(j)) if 'score' not in j else str(i).rjust(len(j)) for i, j in zip(r, index)]))

    # print(
    #     f"正样本集得分为：{pos_correct_entities} / ({pos_correct_entities}+{pos_wrong_entities}+{pos_omitted_entities}+{pos_redundant_entities}) = {pos_metric}，"
    #     f"负样本集得分为：{neg_correct_dt} / ({neg_correct_dt} + {neg_redundant_entities})={neg_metric}，",
    #     f"总体得分为： ({pos_correct_entities} + {neg_correct_dt}) / ({all_pos_entities}+{neg_correct_dt + neg_redundant_entities})={micro_score}",
    #     f"样本准确率：{round(accuracy, score_precision)}",
    # )
    all_score = {'pos_metric': pos_metric, 'neg_metric': neg_metric,
                 'micro_precision': micro_precision, 'micro_recall': micro_recall, 'micro_f1': micro_f1,
                 'macro_precision': macro_precision, 'macro_recall': macro_recall, 'macro_f1': macro_f1,
                 'micro_score': micro_score, 'macro_score': macro_score, 'weighted_score': weighted_score,
                 'accuracy': accuracy}
    if return_score_key:
        all_score = {i: all_score[i] for i in return_score_key if i in all_score.keys()}

    return all_score


def entity_recognition_metric_for_data(y_true, y_pred, all_entity_type, show_each_entity_score=False,
                                       return_score_key=None):
    # 对于原始数据格式进行的评估
    # 每个实体的指标需要有，整体的指标需要有
    # pos_neg_ratio: str = None, return_score_key = None
    assert len(y_true) == len(y_pred)

    def get_refined_res(res):
        refined_res = []
        for each_sample_res in res:
            refined_e_res = dict()
            for each_entity_type in all_entity_type:
                refined_entity_res = set()
                for each_entity_res in each_sample_res.get(each_entity_type, []):
                    refined_entity_res.add(
                        '|'.join([
                            f'{each_entity_part["start_offset"]},{each_entity_part["end_offset"]},{each_entity_part["text"]}'
                            for each_entity_part in sorted(each_entity_res, key=lambda x: x['start_offset'])]))
                refined_e_res[each_entity_type] = refined_entity_res
            refined_res.append(refined_e_res)
        return refined_res

    refined_y_true = get_refined_res(y_true)
    refined_y_pred = get_refined_res(y_pred)

    if show_each_entity_score:
        for entity_type in all_entity_type:
            sub_true = [i[entity_type] for i in refined_y_true]
            sub_pred = [i[entity_type] for i in refined_y_pred]
            print(f'实体：{entity_type} 的得分统计：')
            entity_recognition_metric(y_true=sub_true, y_pred=sub_pred, return_score_key=return_score_key)

    refined_y_true = [{k + ":" + j for k, v in i.items() for j in v} for i in refined_y_true]
    refined_y_pred = [{k + ":" + j for k, v in i.items() for j in v} for i in refined_y_pred]
    print(f'所有实体的得分统计：')
    res = entity_recognition_metric(y_true=refined_y_true, y_pred=refined_y_pred, return_score_key=return_score_key)
    return res


if __name__ == '__main__':
    # a = [{'0,1', '2,3', '3,4', '5,6|7,9'}, {'0,1', '2,3', '3,4', '5,6|7,9'}, {'3,2'},set()]
    # b = [{'0,2', '3,4', '5,6|7,9'}, {'3,2'}, {'3,2'},{'2,3'}]

    # a = [set(), {'d'}, set()]
    # b = [{'1', '2', '4'}, {'d'}, set()]
    # entity_recognition_metric(y_true=a, y_pred=b)

    a = [
        {
            'entity_type_1': [
                [{'start_offset': 1, 'end_offset': 2, 'text': 'ab'},
                 {'start_offset': 6, 'end_offset': 10, 'text': 'dfgwa'}],
                [{'start_offset': 4, 'end_offset': 7, 'text': 'dgsg'}],
            ],
            'entity_type_2': [
                [{'start_offset': 1, 'end_offset': 2, 'text': 'ab'}],
                [{'start_offset': 4, 'end_offset': 7, 'text': 'dgsg'}],
            ],
            'entity_type_3': [
            ],

        },
        {
            'entity_type_1': [
                [{'start_offset': 1, 'end_offset': 2, 'text': 'ab'},
                 {'start_offset': 6, 'end_offset': 10, 'text': 'dfgwa'}],
                [{'start_offset': 4, 'end_offset': 7, 'text': 'dgsg'}],
                [{'start_offset': 4, 'end_offset': 7, 'text': 'dgsg'}],
            ],
            'entity_type_2': [
                [{'start_offset': 4, 'end_offset': 7, 'text': 'dgsg'}],
            ],
        },

    ]
    b = [
        {
            'input_text': 's',
            'entity_type_1': [
                [{'start_offset': 1, 'end_offset': 2, 'text': 'ab'},
                 {'start_offset': 6, 'end_offset': 10, 'text': 'dfgwa'}],
                [{'start_offset': 4, 'end_offset': 7, 'text': 'dgsg'}],
            ],
            'entity_type_2': [
                [{'start_offset': 1, 'end_offset': 2, 'text': 'ab'}],
                [{'start_offset': 4, 'end_offset': 7, 'text': 'dgsg'}],
            ],
            'entity_type_3': [
            ],
        },
        {
            'entity_type_1': [
                [{'start_offset': 6, 'end_offset': 10, 'text': 'dfgwa'},
                 {'start_offset': 1, 'end_offset': 2, 'text': 'ab'}, ],
                [{'start_offset': 4, 'end_offset': 8, 'text': 'dgsgd'}],
                [{'start_offset': 2, 'end_offset': 5, 'text': 'dgsg'}],
            ],
            'entity_type_2': [
                [{'start_offset': 4, 'end_offset': 7, 'text': 'dgsg'}],
            ],
        },

    ]
    entity_recognition_metric_for_data(y_true=a, y_pred=b,
                                       all_entity_type=['entity_type_1', 'entity_type_2', 'entity_type_3'],
                                       show_each_entity_score=True)
