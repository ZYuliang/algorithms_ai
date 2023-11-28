from loguru import logger
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix, multilabel_confusion_matrix
import numpy as np

def metric_for_single_label_classification(y_true, y_pred, id2label=None, show_confusion_matrix=False,
                                           return_metric=None, show_report=False, digits=3):
    """ 单标签多分类的评估
    return_metric:是否指定返回key的指标

    y_true = [3, 4, 1, 3, 4]
    y_pred = [2, 3, 4, 3, 4]
    id2_label = {
        1: 'a',
        2: 'b',
        3: 'c',
        4: 'd',
    }
    print(metric_for_cls(y_true, y_pred, id2_label, show_confusion_matrix=True,
                         return_metric=None))
    ##
    2023-04-21 15:40:30.825 | INFO     | __main__:metric_for_cls:11 - label:['a', 'b', 'c', 'd']
    2023-04-21 15:40:30.827 | INFO     | __main__:metric_for_cls:13 - confusion_matrix:
    [[0 0 0 1]
     [0 0 0 0]
     [0 1 1 0]
     [0 0 1 1]]
    {'a_precision': 0.0, 'a_recall': 0.0, 'a_f1-score': 0.0, 'a_support': 1,
    'b_precision': 0.0, 'b_recall': 0.0, 'b_f1-score': 0.0, 'b_support': 0,
    'c_precision': 0.5, 'c_recall': 0.5, 'c_f1-score': 0.5, 'c_support': 2,
    'd_precision': 0.5, 'd_recall': 0.5, 'd_f1-score': 0.5, 'd_support': 2,
    'accuracy': 0.4,
    'macro avg_precision': 0.25, 'macro avg_recall': 0.25, 'macro avg_f1-score': 0.25, 'macro avg_support': 5,
    'weighted avg_precision': 0.4, 'weighted avg_recall': 0.4, 'weighted avg_f1-score': 0.4, 'weighted avg_support': 5}
    """

    if id2label:
        y_true = [id2label[i] for i in y_true]
        y_pred = [id2label[i] for i in y_pred]

    if show_confusion_matrix:
        # 上方表示预测值，左边表示实际值
        label_index = list(id2label.values())
        # logger.info(f'label:{label_index}')
        c_m = confusion_matrix(y_true, y_pred, labels=label_index)
        c_m_r = [['预测\真实']+label_index]
        print('混淆矩阵：')
        for i,j in zip(label_index,c_m):
            c_m_r.append([i]+list(j))
        for i in c_m_r:
            for j in i:
                print(f'{str(j):>10}',end='')
            print('\n')

    if show_report:
        logger.info(
            f'classification_report:\n{classification_report(y_true, y_pred, output_dict=False, zero_division=0, digits=digits)}')

    res = classification_report(y_true, y_pred, output_dict=True, zero_division=0, digits=digits)
    refined_res = dict()
    for i, j in res.items():
        if isinstance(j, dict):
            for p, q in j.items():
                refined_res[i + '_' + p] = q
        else:
            refined_res[i] = j

    if return_metric:
        return {i: round(j, digits) for i, j in refined_res.items() if i in return_metric}
    else:
        return {i: round(j, digits) for i, j in refined_res.items()}


def metric_for_multi_label_classification(y_true, y_pred,
                                          labels=None,
                                          show_confusion_matrix=False,
                                          show_report=False,
                                          return_metric=None,
                                          digits=3
                                          ):
    """ 从样本角度和类别角度分别计算指标,一般return_metric用hamming_score
    y_true = [[0, 1, 0, 1],
                       [0, 1, 1, 0],
                       [1, 0, 1, 1]]

    y_pred = [[0, 1, 1, 0],
                       [0, 1, 1, 0],
                       [0, 1, 0, 1]]
    labels = ['a','b','c','d']
    print(metric_for_multi_label_classification(y_true, y_pred,
                                                labels=labels,
                                                show_confusion_matrix=True,
                                                show_report=True,
                                                return_metric=None
                                        ))

    ###
    2023-04-23 11:26:33.396 | INFO     | __main__:metric_for_multi_label_classification:134 - labels:['a', 'b', 'c', 'd']
    2023-04-23 11:26:33.398 | INFO     | __main__:metric_for_multi_label_classification:137 - confusion_matrix for labels:
    [[[2 0]
      [1 0]]

     [[0 1]
      [0 2]]

     [[0 1]
      [1 1]]

     [[1 0]
      [1 1]]]
    2023-04-23 11:26:33.403 | INFO     | __main__:metric_for_multi_label_classification:139 - multi_label_classification_report:
                  precision    recall  f1-score   support

               0      0.000     0.000     0.000         1
               1      0.667     1.000     0.800         2
               2      0.500     0.500     0.500         2
               3      1.000     0.500     0.667         2

       micro avg      0.667     0.571     0.615         7
       macro avg      0.542     0.500     0.492         7
    weighted avg      0.619     0.571     0.562         7
     samples avg      0.667     0.611     0.633         7

    {'exact_accuracy': 0.333, 'hamming_score': 0.528,
    '0_precision': 0.0, '0_recall': 0.0, '0_f1-score': 0.0,
    '0_support': 1, '1_precision': 0.667, '1_recall': 1.0, '1_f1-score': 0.8,
    '1_support': 2, '2_precision': 0.5, '2_recall': 0.5, '2_f1-score': 0.5, '2_support': 2,
    '3_precision': 1.0, '3_recall': 0.5, '3_f1-score': 0.667, '3_support': 2,
    'micro avg_precision': 0.667, 'micro avg_recall': 0.571, 'micro avg_f1-score': 0.615, 'micro avg_support': 7,
    'macro avg_precision': 0.542, 'macro avg_recall': 0.5, 'macro avg_f1-score': 0.492, 'macro avg_support': 7,
    'weighted avg_precision': 0.619, 'weighted avg_recall': 0.571, 'weighted avg_f1-score': 0.562, 'weighted avg_support': 7,
    'samples avg_precision': 0.667, 'samples avg_recall': 0.611, 'samples avg_f1-score': 0.633, 'samples avg_support': 7}
    ###


    """

    if labels:
        logger.info(f'labels:{labels}')
    if show_confusion_matrix:
        # 分多个类类别，上方表示预测值，左边表示实际值
        logger.info(f'confusion_matrix for labels:\n{multilabel_confusion_matrix(y_true, y_pred, samplewise=False)}')
    if show_report:
        logger.info(
            f'multi_label_classification_report:\n{classification_report(y_true, y_pred, output_dict=False, zero_division=0, digits=digits)}')

    cls_metric_res = classification_report(y_true, y_pred, output_dict=True, zero_division=0, digits=digits)

    refined_res = {
        'exact_accuracy': accuracy_score(y_true, y_pred),
        # 'zero_one_loss':zero_one_loss(y_true, y_pred),
        # 'hamming_loss':hamming_loss(y_true,y_pred),
        'hamming_score': caculate_hamming_score(y_true, y_pred)
    }

    for i, j in cls_metric_res.items():
        if isinstance(j, dict):
            for p, q in j.items():
                refined_res[i + '_' + p] = q
        else:
            refined_res[i] = j
    if return_metric:
        return {i: round(j, digits) for i, j in refined_res.items() if i in return_metric}
    else:
        return {i: round(j, digits) for i, j in refined_res.items()}


def caculate_hamming_score(y_true, y_pred, digits=3):
    hamming_score = 0
    for true, pred in zip(y_true, y_pred):
        if np.array_equal(true, pred):
            sub_hamming_score = 1
        else:
            sub_hamming_score_up = 0
            sub_hamming_score_down = 0
            for i, j in zip(true, pred):
                if i == 1:
                    sub_hamming_score_down += 1
                    if j == 1:
                        sub_hamming_score_up += 1
                else:
                    if j == 1:
                        sub_hamming_score_down += 1
            sub_hamming_score = sub_hamming_score_up / sub_hamming_score_down
        hamming_score += sub_hamming_score
    hamming_score = round(hamming_score / len(y_true), digits)
    return hamming_score


def compute_metrics_for_single_label_classification(eval_pred, id2label,show_confusion_matrix=False,return_metric=None,
                                                    show_report=False):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    res = metric_for_single_label_classification(labels, predictions, id2label=id2label, show_confusion_matrix=show_confusion_matrix,
                                                 return_metric=return_metric,show_report=show_report)
    return res


if __name__ == '__main__':
    y_true = [3, 4, 1, 3, 4]
    y_pred = [2, 3, 4, 3, 4]
    id2_label = {
        1: 'a',
        2: 'b',
        3: 'c',
        4: 'd',
    }
    print(metric_for_single_label_classification(y_true, y_pred, id2_label, show_confusion_matrix=True,
                                                 return_metric=None))

    #############
    # import numpy as np
    #
    # y_true = [[0, 1, 0, 1],
    #           [0, 1, 1, 0],
    #           [1, 0, 1, 1]]
    #
    # y_pred = [[0, 1, 1, 0],
    #           [0, 1, 1, 0],
    #           [0, 1, 0, 1]]
    # labels = ['a', 'b', 'c', 'd']
    # print(metric_for_multi_label_classification(y_true, y_pred,
    #                                             labels=labels,
    #                                             show_confusion_matrix=True,
    #                                             show_report=True,
    #                                             return_metric=None
    #                                             ))
