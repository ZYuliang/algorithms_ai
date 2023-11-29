# https://data.csiro.au/collection/csiro:10948?v=3&d=true
# cadec 2
import os
import re

import datasets

from algorithms_ai.dataset_utils.dataset_create.data_config import cadec2_raw_data, cadec2_ner_results, \
    cadec2_dataset_dict
from algorithms_ai.dataset_utils.dataset_utils import split_train_dev_test, save_and_load_dataset_dict
from algorithms_ai.dataset_utils.ner_process_utils import NERNormalizer
from algorithms_ai.utils.file_utils.file_utils import save_data, load_data
from algorithms_ai.utils.tokenizer_utils.regex_tokenizer import RegexTokenizer


def deal_with_cadec2():
    def extract_indices_from_brat_annotation(indices: str):
        indices = re.findall(r"\d+", indices)
        indices = [int(i) for i in indices]
        return indices

    input_ann = os.path.join(cadec2_raw_data, 'original')
    input_text_dir = os.path.join(cadec2_raw_data, 'text')

    all_samples = []

    for document in os.listdir(input_ann):
        with open(os.path.join(input_ann, document), "r") as in_f:
            document = document.replace(".ann", "")
            text = open(os.path.join(input_text_dir, "%s.txt" % document)).read()
            if not text:
                continue

            each_sample = {
                'input_text': text
            }
            ner_results = dict()
            is_right = True
            for line in in_f:
                line = line.strip()
                if line[0] != "T": continue
                sp = line.strip().split("\t")
                assert len(sp) == 3
                mention = sp[2]
                sp = sp[1].split(" ")
                label = 'NER_' + sp[0]
                if label not in ner_results:
                    ner_results[label] = []

                indices = extract_indices_from_brat_annotation(" ".join(sp[1:]))
                if ' '.join([text[indices[i]:indices[i + 1]] for i in range(0, len(indices), 2)]) == mention:
                    sub_ner = []
                    for i in range(0, len(indices), 2):
                        sub_ner.append(
                            {
                                'text': text[indices[i]:indices[i + 1]],
                                'start_offset': indices[i],
                                'end_offset': indices[i + 1] - 1,
                            }
                        )
                    ner_results[label].append(sub_ner)
                else:
                    is_right = False
            if is_right:
                each_sample.update(ner_results)

            all_samples.append(each_sample)

    save_data(all_samples, cadec2_ner_results)
    return all_samples


@save_and_load_dataset_dict(dataset_dir=cadec2_dataset_dict)
def get_dataset_dict():
    all_samples = load_data(cadec2_ner_results)
    train_data, dev_data, test_data = split_train_dev_test(all_samples)

    sentence_tokenizer = RegexTokenizer(regex='\n').run
    ner_norm = NERNormalizer()

    all_ner_keys = set()
    for i in train_data: all_ner_keys.update([k for k, _ in i.items() if k.startswith('NER')])

    refined_train_data = ner_norm.run(train_data, ner_keys=all_ner_keys,
                                      sentence_tokenizer=sentence_tokenizer, keep_features=None)
    refined_test_data = ner_norm.run(test_data, ner_keys=all_ner_keys,
                                     sentence_tokenizer=sentence_tokenizer, keep_features=None)
    refined_dev_data = ner_norm.run(dev_data, ner_keys=all_ner_keys,
                                    sentence_tokenizer=sentence_tokenizer, keep_features=None)

    dataset_dict = {
        'raw_train': datasets.Dataset.from_list(train_data),
        'raw_dev': datasets.Dataset.from_list(dev_data),
        'raw_test': datasets.Dataset.from_list(test_data),
        'train': datasets.Dataset.from_list(refined_train_data),
        'dev': datasets.Dataset.from_list(refined_dev_data),
        'test': datasets.Dataset.from_list(refined_test_data),
    }
    return datasets.DatasetDict(dataset_dict)


if __name__ == '__main__':
    # deal_with_cadec2()
    a = get_dataset_dict(use_file_cache=True)
    print(2)
