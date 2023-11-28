"""
数据的collator

"""

import numpy as np
import torch

from information_extraction.utils_ie import sequence_padding


class CollatorForIE:
    def __init__(self, entity2id=None, relation2id=None, add_raw_info=None, mode='train'):
        self.entity2id = entity2id
        self.relation2id = relation2id
        self.add_raw_info = add_raw_info
        self.mode = mode

    def __call__(self, features):
        batch_token_ids, batch_mask_ids, batch_token_type_ids = [], [], []
        batch_entity_labels, batch_head_labels, batch_tail_labels, batch_relation_labels = [], [], [], []
        batch_input_ids_mask = []
        raw_info = []
        for item in features:
            if self.mode != 'predict':
                if self.entity2id:
                    batch_entity_labels.append(np.array(item['labels'][0]))

                if self.relation2id:
                    batch_relation_labels.append(np.array(item['labels'][1]))
                    batch_head_labels.append(np.array(item['labels'][2]))
                    batch_tail_labels.append(np.array(item['labels'][3]))

            batch_token_ids.append(item['input_ids'])
            batch_mask_ids.append(item['attention_mask'])
            batch_token_type_ids.append(item['token_type_ids'])
            batch_input_ids_mask.append(item['input_ids_mask'])

            if self.add_raw_info:
                raw_info.append(dict())
                for each_info in self.add_raw_info:
                    raw_info[-1][each_info] = item.get(each_info, None)

        batch_token_ids = torch.tensor(sequence_padding(batch_token_ids)).long()
        batch_mask_ids = torch.tensor(sequence_padding(batch_mask_ids)).float()
        batch_token_type_ids = torch.tensor(sequence_padding(batch_token_type_ids)).long()
        batch_input_ids_mask = torch.tensor(sequence_padding(batch_input_ids_mask)).long()

        if self.mode != 'predict':
            if self.entity2id:
                batch_entity_labels = torch.tensor(sequence_padding(batch_entity_labels, seq_dims=2)).long()
            if self.relation2id:
                batch_relation_labels = torch.tensor(sequence_padding(batch_relation_labels, seq_dims=2)).long()
                batch_head_labels = torch.tensor(sequence_padding(batch_head_labels, seq_dims=2)).long()
                batch_tail_labels = torch.tensor(sequence_padding(batch_tail_labels, seq_dims=2)).long()

        if self.mode != 'predict':
            if self.add_raw_info:
                return {
                    'input_ids': batch_token_ids,
                    'input_ids_mask': batch_input_ids_mask,
                    'labels': [batch_entity_labels, batch_relation_labels, batch_head_labels, batch_tail_labels],
                    'attention_mask': batch_mask_ids,
                    'token_type_ids': batch_token_type_ids,
                    'raw_info': raw_info
                }

            else:
                return {
                    'input_ids': batch_token_ids,
                    'input_ids_mask': batch_input_ids_mask,
                    'labels': [batch_entity_labels, batch_relation_labels, batch_head_labels, batch_tail_labels],
                    'attention_mask': batch_mask_ids,
                    'token_type_ids': batch_token_type_ids,
                }
        else:
            if self.add_raw_info:
                return {
                    'input_ids': batch_token_ids,
                    'input_ids_mask': batch_input_ids_mask,
                    'attention_mask': batch_mask_ids,
                    'token_type_ids': batch_token_type_ids,
                    'raw_info': raw_info
                }
            else:
                return {
                    'input_ids': batch_token_ids,
                    'input_ids_mask': batch_input_ids_mask,
                    'attention_mask': batch_mask_ids,
                    'token_type_ids': batch_token_type_ids,
                }
