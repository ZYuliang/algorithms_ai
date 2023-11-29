def entity_recognition_metric(y_true: list, y_pred: list, pos_neg_ratio: str = None):
    """the metric of entity_recognition, reference: https://docs.qq.com/doc/DYXRYQU1YbkVvT3V2
    实体识别指标，输入一个list，list中每个元素是一个对象（可以是一句或一段或一篇）的实体结果，每个对象的实体结果是一个set，
    这个set中每个实体用唯一值来映射,一般坐标,坐标要排序,若要指定实体类别，前面加id-
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
    for i, j in zip(y_true, y_pred):
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

    # pos_precision = pos_correct_dt / (neg_correct_dt + pos_correct_dt)
    # recall = pos_correct_dt / pos_data
    tp = pos_correct_dt
    fn = pos_wrong_dt
    fp = neg_wrong_dt
    tn = neg_correct_dt

    accuracy = (tp + tn) / (tp + fn + fp + tn)
    # precision = tp / (tp + fp)
    # recall = tp / (tp + fn)
    # f1 = 2 / (1 / precision + 1 / recall)
    results = [
        ['positive data', pos_data, pos_correct_dt, pos_wrong_dt, pos_correct_entities,
         pos_wrong_entities, pos_omitted_entities, pos_redundant_entities, pos_metric],
        ['negative data', neg_data, neg_correct_dt, neg_wrong_dt, '-', '-', '-', neg_redundant_entities, neg_metric],
        ['all data', pos_data + neg_data, neg_correct_dt + pos_correct_dt, neg_wrong_dt + pos_wrong_dt,
         pos_correct_entities, pos_wrong_entities, pos_omitted_entities,
         pos_redundant_entities + neg_redundant_entities,
         ''],
        # ['precision','', '', '', '', '', '', '', precision],
        # ['recall','', '', '', '', '', '', '', recall],
        # ['f1 score','', '', '', '', '', '', '', (2 * precision * recall) / (precision + recall)],
        # ['accuracy score','', '', '', '', '', '', '', (neg_correct_dt + pos_correct_dt) / (pos_data + neg_data)],
        ['micro score', '', '', '', '', '', '', '', micro_score],
        ['macro score', '', '', '', '', '', '', '', macro_score],
        ['weighted score', '', '', '', '', '', '', '', weighted_score],
    ]

    index = [' ' * 17, '| data_num ', '| correct_data ', '| wrong_data ', '| correct_entities ', '| wrong_entities ',
             '| omitted_entities ', '| redundant_entities ', '|  score']

    print(''.join(index))
    for r in results:
        print(''.join([str(i).center(len(j)) if 'score' not in j else str(i).rjust(len(j)) for i, j in zip(r, index)]))

    print(
        f"正样本集得分为：{pos_correct_entities} / ({pos_correct_entities}+{pos_wrong_entities}+{pos_omitted_entities}+{pos_redundant_entities}) = {pos_metric}，"
        f"负样本集得分为：{neg_correct_dt} / ({neg_correct_dt} + {neg_redundant_entities})={neg_metric}，",
        f"总体得分为： ({pos_correct_entities} + {neg_correct_dt}) / ({all_pos_entities}+{neg_correct_dt + neg_redundant_entities})={micro_score}",
        f"样本准确率：{round(accuracy, score_precision)}",
    )
    return {
        'pos_metric': pos_metric,
        'neg_metric': neg_metric,
        'micro_score': micro_score,
        'macro_score': macro_score,
        'weighted_score': weighted_score
    }


if __name__ == '__main__':
    a = [{'0,1', '2,3', '3,4', '5,6|7,9'}, {'0,1', '2,3', '3,4', '5,6|7,9'}, {'3,2'}]
    b = [{'0,2', '3,4', '5,6|7,9'}, {'3,2'}, {'3,2'}]
    entity_recognition_metric(a, b)
