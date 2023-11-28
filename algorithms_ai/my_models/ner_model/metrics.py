from collections import defaultdict
from collections import namedtuple


SeqLabelBlock = namedtuple(
    'SeqLabelBlock',
    ['block_type', 'start_offset', 'end_offset']
)


def compute_precision_recall(correct, actual, possible):
    """
    Args:
        correct (:obj:`int`):
            Number of correct predictions.
        actual (:obj:`int`):
            Number of predicted blocks.
        possible (:obj:`int`):
            Number of ground truth blocks.
    Returns:
        :obj:`tuple`:
        precision (:obj:`float`)
        recall (:obj:`int`)
        f1 (:obj:`int`)
    """
    precision = correct / actual if actual > 0 else 0.0
    recall = correct / possible if possible > 0 else 0.0
    f1 = 2.0 * precision * recall / (precision + recall + 1e-6)
    return precision, recall, f1


def collect_seqlabel_blocks(labels):
    """
    Args:
        labels (:obj:`List[str]`):
            BIO sequence labels
    Returns:
        :obj:`List[SeqLabelBlock]`
    """
    blocks = []
    start_offset, end_offset, block_type = None, None, None

    for offset, lab in enumerate(labels):
        if lab == 'O':
            if (block_type is not None and
                    start_offset is not None):
                # last block ends
                end_offset = offset - 1
                blocks.append(
                    SeqLabelBlock(
                        block_type, start_offset, end_offset)
                )
                start_offset = None
                end_offset = None
                block_type = None
        elif block_type is None:
            # start of a block
            block_type = lab[2:]
            start_offset = offset
        elif block_type != lab[2:] or \
                (block_type == lab[2:] and lab[0] == 'B'):
            # last block ends and new block starts
            end_offset = offset - 1
            blocks.append(
                SeqLabelBlock(
                    block_type, start_offset, end_offset)
            )
            block_type = lab[2:]
            start_offset = offset
            end_offset = None

    if (block_type and
            start_offset is not None and
            end_offset is None):
        blocks.append(
            SeqLabelBlock(
                block_type, start_offset, len(labels) - 1)
        )

    return blocks


def compute_token_level_metrics(gold_blocks, pred_blocks):
    """
    Args:
        gold_blocks (:obj:`List[SeqLabelBlock]`)
        pred_blocks (:obj:`List[SeqLabelBlock]`)
    Returns:
        :obj:`Dict`
    """
    metrics = {
        'correct': 0,
        'actual': sum(
            [b.end_offset + 1 - b.start_offset for b in pred_blocks]),
        'possible': sum(
            [b.end_offset + 1 - b.start_offset for b in gold_blocks])
    }
    for pred in pred_blocks:
        for gold in gold_blocks:
            if pred.block_type == gold.block_type:
                pred_range = range(pred.start_offset, pred.end_offset + 1)
                gold_range = range(gold.start_offset, gold.end_offset + 1)
                overlap_range = find_overlap(gold_range, pred_range)
                metrics['correct'] += len(overlap_range)
    return metrics


def compute_muc_metircs(gold_blocks, pred_blocks):
    """
    Args:
        gold_blocks (:obj:`List[SeqLabelBlock]`)
        pred_blocks (:obj:`List[SeqLabelBlock]`)
    Returns:
        :obj:`Dict`
    """
    metrics = {
        'correct': 0,
        'actual': 2 * len(pred_blocks),
        'possible': 2 * len(gold_blocks)
    }
    for pred in pred_blocks:
        if pred in gold_blocks:
            metrics['correct'] += 2
        else:
            for gold in gold_blocks:
                pred_range = range(pred.start_offset, pred.end_offset + 1)
                gold_range = range(gold.start_offset, gold.end_offset + 1)
                if (gold.start_offset == pred.start_offset and
                    gold.end_offset == pred.end_offset and
                        gold.block_type != pred.block_type):
                    metrics['correct'] += 1
                    break
                elif (find_overlap(gold_range, pred_range) and
                        gold.block_type == pred.block_type):
                    metrics['correct'] += 1

    return metrics


def compute_exact_match_metrics(gold_blocks, pred_blocks):
    """ http://www.davidsbatista.net/blog/2018/05/09/Named_Entity_Evaluation/
    Args:
        gold_blocks (:obj:`List[SeqLabelBlock]`)
        pred_blocks (:obj:`List[SeqLabelBlock]`)
    Returns:
        :obj:`Dict`
    """
    metrics = {
        'correct': 0,
        'incorrect': 0,
        'missed': 0,
        'spurious': 0,
        'actual': 0,
        'possible': 0
    }
    metrics['possible'] = len(gold_blocks)
    metrics['actual'] = len(pred_blocks)

    for pred in pred_blocks:
        # Scenario I: Exact match between gold and pred
        if pred in gold_blocks:
            metrics['correct'] += 1
        else:
            overlapped = False
            for gold in gold_blocks:
                pred_range = range(pred.start_offset, pred.end_offset + 1)
                gold_range = range(gold.start_offset, gold.end_offset + 1)

                # Scenario IV: Offsets match, but type is wrong
                if (gold.start_offset == pred.start_offset and
                    gold.end_offset == pred.end_offset and
                        gold.block_type != pred.block_type):
                    metrics['incorrect'] += 1
                    overlapped = True
                    break
                # Scenario V: Wrong boundaries
                elif find_overlap(gold_range, pred_range):
                    metrics['incorrect'] += 1
                    overlapped = True
                    break
            # Scenario II: Blocks are spurious
            if not overlapped:
                metrics['spurious'] += 1

    # Scenario III: Block was missed entirely.
    metrics['missed'] = metrics['possible'] - metrics['correct'] - \
        metrics['incorrect']

    return metrics


def find_overlap(true_range, pred_range):
    """Find the overlap between two ranges
    Find the overlap between two ranges. Return the overlapping values if
    present, else return an empty set().
    """

    true_set = set(true_range)
    pred_set = set(pred_range)

    overlaps = true_set.intersection(pred_set)

    return overlaps


def calc_pair(preds, golds):
    correct_samples = 0
    total_samples = 0
    results_exact_match = defaultdict(int)
    results_MUC = defaultdict(int)
    results_token_level = defaultdict(int)


    for pred,gold in zip(preds,golds):
        pred = collect_seqlabel_blocks(pred)
        gold = collect_seqlabel_blocks(gold)
        exact_match = compute_exact_match_metrics(gold, pred)
        muc = compute_muc_metircs(gold, pred)
        token_level = compute_token_level_metrics(gold, pred)

        if (exact_match['correct'] == exact_match['possible'] and
                exact_match['correct'] == exact_match['actual']):
            correct_samples += 1

        for metric in exact_match:
            results_exact_match[metric] += exact_match[metric]
        for metric in muc:
            results_MUC[metric] += muc[metric]
        for metric in token_level:
            results_token_level[metric] += token_level[metric]

    print(results_exact_match,results_MUC,results_token_level)
    token_p, token_r, token_f1 = \
        compute_precision_recall(
            results_token_level['correct'],
            results_token_level['actual'],
            results_token_level['possible']
        )

    muc_p, muc_r, muc_f1 = \
        compute_precision_recall(
            results_MUC['correct'],
            results_MUC['actual'],
            results_MUC['possible']
        )

    exact_p, exact_r, exact_f1 = \
        compute_precision_recall(
           results_exact_match['correct'],
            results_exact_match['actual'],
            results_exact_match['possible']
        )
    print( token_p, token_r, token_f1)
    print( muc_p, muc_r, muc_f1)
    print(exact_p, exact_r, exact_f1)



# def v2(predictions,references,
#        suffix: bool = False,
#         scheme = None,
#         mode = None,
#         sample_weight = None,
#         zero_division = "warn",):
#     report = classification_report(
#         y_true=references,
#         y_pred=predictions,
#         suffix=suffix,
#         output_dict=True,
#         scheme=scheme,
#         mode=mode,
#         sample_weight=sample_weight,
#         zero_division=zero_division,
#     )
#     report.pop("macro avg")
#     report.pop("weighted avg")
#     overall_score = report.pop("micro avg")
#
#     scores = {
#         type_name: {
#             "precision": score["precision"],
#             "recall": score["recall"],
#             "f1": score["f1-score"],
#             "number": score["support"],
#         }
#         for type_name, score in report.items()
#     }
#     scores["overall_precision"] = overall_score["precision"]
#     scores["overall_recall"] = overall_score["recall"]
#     scores["overall_f1"] = overall_score["f1-score"]
#     scores["overall_accuracy"] = accuracy_score(y_true=references, y_pred=predictions)
#     return scores

if __name__ == '__main__':
    la = [['O', 'O', 'O', 'B-arm_option', 'O', 'B-arm_option', 'I-shared_arm_option', 'I-shared_arm_option', 'I-shared_arm_option', 'I-shared_arm_option', 'I-shared_arm_option', 'I-shared_arm_option', 'I-shared_arm_option', 'I-shared_arm_option', 'I-shared_arm_option', 'I-shared_arm_option', 'I-shared_arm_option', 'I-shared_arm_option', 'I-shared_arm_option', 'O'], ['O', 'O', 'O', 'O', 'B-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'O'], ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-shared_arm_option', 'I-shared_arm_option', 'I-shared_arm_option', 'I-shared_arm_option', 'I-shared_arm_option', 'I-shared_arm_option', 'I-shared_arm_option', 'I-shared_arm_option', 'I-shared_arm_option', 'I-shared_arm_option', 'I-shared_arm_option', 'I-shared_arm_option', 'I-shared_arm_option', 'I-shared_arm_option', 'I-shared_arm_option', 'I-arm_option', 'O', 'B-arm_option', 'I-shared_arm_option', 'I-shared_arm_option', 'I-shared_arm_option', 'I-shared_arm_option', 'I-shared_arm_option', 'I-shared_arm_option', 'I-shared_arm_option', 'I-shared_arm_option', 'I-shared_arm_option', 'I-shared_arm_option', 'I-shared_arm_option', 'I-shared_arm_option', 'I-shared_arm_option', 'I-shared_arm_option', 'I-shared_arm_option', 'I-shared_arm_option', 'I-shared_arm_option', 'I-shared_arm_option', 'I-shared_arm_option', 'O'], ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-shared_arm_option', 'I-shared_arm_option', 'I-shared_arm_option', 'I-shared_arm_option', 'I-shared_arm_option', 'I-shared_arm_option', 'I-shared_arm_option', 'I-shared_arm_option', 'I-shared_arm_option', 'I-arm_option', 'I-arm_option', 'O', 'B-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'O'], ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'O', 'B-arm_option', 'I-arm_option', 'I-arm_option', 'O', 'O', 'O', 'O', 'O', 'O'], ['O', 'O', 'B-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'O', 'O', 'B-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'O'], ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'O', 'B-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-shared_arm_option', 'I-shared_arm_option', 'I-shared_arm_option', 'I-shared_arm_option', 'I-shared_arm_option', 'I-shared_arm_option', 'I-shared_arm_option', 'I-shared_arm_option', 'I-shared_arm_option', 'I-shared_arm_option', 'I-shared_arm_option', 'I-shared_arm_option', 'I-shared_arm_option', 'I-shared_arm_option', 'I-shared_arm_option', 'I-shared_arm_option', 'I-shared_arm_option', 'I-shared_arm_option', 'I-shared_arm_option', 'I-shared_arm_option', 'I-shared_arm_option', 'I-shared_arm_option', 'I-shared_arm_option', 'I-shared_arm_option', 'I-shared_arm_option', 'I-shared_arm_option', 'I-shared_arm_option', 'I-shared_arm_option', 'I-shared_arm_option', 'I-shared_arm_option', 'I-shared_arm_option', 'I-shared_arm_option', 'I-shared_arm_option', 'I-shared_arm_option', 'I-shared_arm_option', 'I-shared_arm_option', 'I-shared_arm_option', 'I-shared_arm_option', 'I-shared_arm_option', 'I-shared_arm_option', 'I-shared_arm_option', 'I-shared_arm_option', 'I-shared_arm_option', 'I-shared_arm_option', 'I-shared_arm_option', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'], ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'O', 'O', 'B-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-shared_arm_option', 'I-shared_arm_option', 'I-shared_arm_option', 'I-shared_arm_option', 'I-shared_arm_option', 'I-shared_arm_option', 'O']]
    pr =  [['O', 'O', 'B-arm_option', 'B-arm_option', 'O', 'B-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'O', 'O', 'O', 'I-shared_arm_option', 'I-shared_arm_option', 'I-shared_arm_option', 'O'], ['O', 'O', 'O', 'O', 'B-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'O'], ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-arm_option', 'I-shared_arm_option', 'I-shared_arm_option', 'I-shared_arm_option', 'I-shared_arm_option', 'I-shared_arm_option', 'I-shared_arm_option', 'I-shared_arm_option', 'I-shared_arm_option', 'I-shared_arm_option', 'I-shared_arm_option', 'I-shared_arm_option', 'I-shared_arm_option', 'I-shared_arm_option', 'I-shared_arm_option', 'I-shared_arm_option', 'O', 'B-arm_option', 'I-shared_arm_option', 'I-shared_arm_option', 'I-shared_arm_option', 'I-shared_arm_option', 'I-shared_arm_option', 'I-shared_arm_option', 'I-shared_arm_option', 'I-shared_arm_option', 'I-shared_arm_option', 'I-shared_arm_option', 'I-shared_arm_option', 'I-shared_arm_option', 'I-shared_arm_option', 'I-shared_arm_option', 'I-shared_arm_option', 'I-shared_arm_option', 'I-shared_arm_option', 'I-shared_arm_option', 'I-shared_arm_option', 'O'], ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'O', 'O', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'O'], ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'O', 'B-arm_option', 'I-arm_option', 'I-arm_option', 'O', 'I-shared_arm_option', 'I-shared_arm_option', 'I-shared_arm_option', 'I-shared_arm_option', 'O'], ['O', 'O', 'B-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'B-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'O'], ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'O', 'B-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-shared_arm_option', 'I-shared_arm_option', 'I-shared_arm_option', 'O', 'I-shared_arm_option', 'O', 'I-shared_arm_option', 'O', 'O', 'I-shared_arm_option', 'O', 'O', 'O', 'I-shared_arm_option', 'I-shared_arm_option', 'O', 'I-shared_arm_option', 'I-shared_arm_option', 'I-shared_arm_option', 'I-shared_arm_option', 'I-shared_arm_option', 'I-shared_arm_option', 'I-shared_arm_option', 'I-shared_arm_option', 'I-shared_arm_option', 'I-shared_arm_option', 'I-shared_arm_option', 'I-shared_arm_option', 'I-shared_arm_option', 'I-shared_arm_option', 'O', 'I-shared_arm_option', 'I-shared_arm_option', 'I-shared_arm_option', 'I-shared_arm_option', 'I-shared_arm_option', 'I-shared_arm_option', 'O', 'O', 'I-shared_arm_option', 'O', 'I-shared_arm_option', 'I-shared_arm_option', 'O', 'I-shared_arm_option', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'], ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'O', 'O', 'B-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-shared_arm_option', 'I-shared_arm_option', 'I-shared_arm_option', 'I-shared_arm_option', 'I-shared_arm_option', 'I-shared_arm_option', 'O']]
    """
    {'eval_loss': 0.2784838080406189, 
    'eval_arm_option': {'precision': 0.625, 'recall': 0.6666666666666666, 'f1': 0.6451612903225806, 'number': 15}, 
    'eval_shared_arm_option': {'precision': 0.13333333333333333, 'recall': 0.3333333333333333, 'f1': 0.19047619047619044, 'number': 6}, 
    'eval_overall_precision': 0.3870967741935484, 
    'eval_overall_recall': 0.5714285714285714, 
    'eval_overall_f1': 0.4615384615384615, 
    'eval_overall_accuracy': 0.9016393442622951, 
    'eval_runtime': 0.4133, 
    'eval_samples_per_second': 19.356, 
    'eval_steps_per_second': 2.419, 
    'epoch': 9.87}
    """
    calc_pair(pr,la)
    # import evaluate
    # seqeval = evaluate.load("seqeval")
    # print(seqeval.compute(predictions=pr, references=la))
    from seqeval.metrics import classification_report

    # print(classification_report(la, pr,
    #                             digits=2,
    #                             suffix=True,
    #                             output_dict=False,
    #                             mode=None,
    #                             sample_weight=None,
    #                             zero_division='warn',
    #                             scheme=None
    #                             ))


    print(classification_report(la, pr,
                                digits=2,
                                suffix=False,
                                output_dict=False,
                                mode=None,
                                sample_weight=None,
                                zero_division='warn',
                                scheme=None
                                ))

    print(classification_report(la, pr,
                            digits=2,
                            suffix=False,
                            output_dict=False,
                            mode="strict",
                            sample_weight=None,
                            zero_division='warn',
                            scheme=None
                            ))

    print(classification_report(la, pr,
                                digits=2,
                                suffix=False,
                                output_dict=True,
                                mode="strict",
                                sample_weight=None,
                                zero_division='warn',
                                scheme=None
                                ))