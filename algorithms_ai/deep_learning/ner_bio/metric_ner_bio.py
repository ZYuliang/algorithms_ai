import evaluate
import numpy as np

seqeval = evaluate.load("seqeval")


def compute_metrics_for_ner(eval_pred, id2label):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=2)  # logits变成label_id

    true_predictions = [
        [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [id2label[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return results
