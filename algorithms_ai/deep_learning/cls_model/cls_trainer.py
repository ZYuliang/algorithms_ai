import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler
from transformers import Trainer
from transformers.trainer import has_length
from typing import Optional


class CLSTrainerWithWeightedRandomSampler(Trainer):
    def __init__(self, model=None,
                 args=None,
                 data_collator=None,
                 train_dataset=None,
                 eval_dataset=None,
                 tokenizer=None,
                 model_init=None,
                 compute_metrics=None,
                 callbacks=None,
                 optimizers=(None, None),
                 preprocess_logits_for_metrics=None):
        """
        处理数据不平衡的trainer
        """
        super(CLSTrainerWithWeightedRandomSampler, self).__init__(model,
                                                                  args,
                                                                  data_collator,
                                                                  train_dataset,
                                                                  eval_dataset,
                                                                  tokenizer,
                                                                  model_init,
                                                                  compute_metrics,
                                                                  callbacks,
                                                                  optimizers,
                                                                  preprocess_logits_for_metrics, )

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None
        generator = None
        if self.args.world_size <= 1:
            generator = torch.Generator()
            # for backwards compatibility, we generate a seed here (which is sampled from a generator seeded with
            # `args.seed`) if data_seed isn't provided.
            # Further on in this method, we default to `args.seed` instead.
            if self.args.data_seed is None:
                seed = int(torch.empty((), dtype=torch.int64).random_().item())
            else:
                seed = self.args.data_seed
            generator.manual_seed(seed)
        all_labels = torch.as_tensor(self.train_dataset['labels'])
        labels_unique, counts = np.unique(all_labels, return_counts=True)
        class_weights = [sum(counts) / c for c in counts]
        samples_weight = [class_weights[e] for e in all_labels]

        # class_sample_count = torch.tensor([(all_labels == t).sum() for t in torch.unique(all_labels,sorted=True)])
        # samples_weight = 1 / class_sample_count
        # samples_weight = [class_sample_count[i] for i in all_labels]

        return WeightedRandomSampler(samples_weight, num_samples=len(all_labels), generator=generator)
