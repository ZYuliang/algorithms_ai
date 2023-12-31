import numpy as np
import torch
import torch.nn as nn
import transformers
# import nlp
import logging
logging.basicConfig(level=logging.INFO)

from datasets import load_dataset
dataset_dict = {
    "stsb": load_dataset('glue', name="stsb"),
    "rte": load_dataset('glue', name="rte"),
    "commonsense_qa": load_dataset('commonsense_qa'),
}
# print(1)
# for task_name, dataset in dataset_dict.items():
#     print(task_name)
#     print(dataset_dict[task_name]["train"][0])
#     print()

class MultitaskModel(transformers.PreTrainedModel):
    def __init__(self, encoder, taskmodels_dict):
        """
        Setting MultitaskModel up as a PretrainedModel allows us
        to take better advantage of Trainer features
        """
        super().__init__(transformers.PretrainedConfig())

        self.encoder = encoder
        self.taskmodels_dict = nn.ModuleDict(taskmodels_dict)

    @classmethod
    def create(cls, model_name, model_type_dict, model_config_dict):
        """
        This creates a MultitaskModel using the model class and config objects
        from single-task models.

        We do this by creating each single-task model, and having them share
        the same encoder transformer.
        """
        shared_encoder = None
        taskmodels_dict = {}
        for task_name, model_type in model_type_dict.items():
            model = model_type.from_pretrained(
                model_name,
                config=model_config_dict[task_name],
            )
            if shared_encoder is None:
                shared_encoder = getattr(model, cls.get_encoder_attr_name(model))
            else:
                setattr(model, cls.get_encoder_attr_name(model), shared_encoder)
            taskmodels_dict[task_name] = model
        return cls(encoder=shared_encoder, taskmodels_dict=taskmodels_dict)

    @classmethod
    def get_encoder_attr_name(cls, model):
        """
        The encoder transformer is named differently in each model "architecture".
        This method lets us get the name of the encoder attribute
        """
        model_class_name = model.__class__.__name__
        if model_class_name.startswith("Bert"):
            return "bert"
        elif model_class_name.startswith("Roberta"):
            return "roberta"
        elif model_class_name.startswith("Albert"):
            return "albert"
        else:
            raise KeyError(f"Add support for new model {model_class_name}")

    def forward(self, task_name, **kwargs):
        return self.taskmodels_dict[task_name](**kwargs)


model_name = "roberta-base"
multitask_model = MultitaskModel.create(
    model_name=model_name,
    model_type_dict={
        "stsb": transformers.AutoModelForSequenceClassification,
        "rte": transformers.AutoModelForSequenceClassification,
        "commonsense_qa": transformers.AutoModelForMultipleChoice,
    },
    model_config_dict={
        "stsb": transformers.AutoConfig.from_pretrained(model_name, num_labels=1),
        "rte": transformers.AutoConfig.from_pretrained(model_name, num_labels=2),
        "commonsense_qa": transformers.AutoConfig.from_pretrained(model_name),
    },
)
print(1)

tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)


max_length = 128

def convert_to_stsb_features(example_batch):
    inputs = list(zip(example_batch['sentence1'], example_batch['sentence2']))
    features = tokenizer.batch_encode_plus(
        inputs, max_length=max_length, pad_to_max_length=True
    )
    features["labels"] = example_batch["label"]
    return features

def convert_to_rte_features(example_batch):
    inputs = list(zip(example_batch['sentence1'], example_batch['sentence2']))
    features = tokenizer.batch_encode_plus(
        inputs, max_length=max_length, pad_to_max_length=True
    )
    features["labels"] = example_batch["label"]
    return features

def convert_to_commonsense_qa_features(example_batch):
    num_examples = len(example_batch["question"])
    num_choices = len(example_batch["choices"][0]["text"])
    features = {}
    for example_i in range(num_examples):
        choices_inputs = tokenizer.batch_encode_plus(
            list(zip(
                [example_batch["question"][example_i]] * num_choices,
                example_batch["choices"][example_i]["text"],
            )),
            max_length=max_length, pad_to_max_length=True,
        )
        for k, v in choices_inputs.items():
            if k not in features:
                features[k] = []
            features[k].append(v)
    labels2id = {char: i for i, char in enumerate("ABCDE")}
    # Dummy answers for test
    if example_batch["answerKey"][0]:
        features["labels"] = [labels2id[ans] for ans in example_batch["answerKey"]]
    else:
        features["labels"] = [0] * num_examples
    return features

convert_func_dict = {
    "stsb": convert_to_stsb_features,
    "rte": convert_to_rte_features,
    "commonsense_qa": convert_to_commonsense_qa_features,
}

columns_dict = {
    "stsb": ['input_ids', 'attention_mask', 'labels'],
    "rte": ['input_ids', 'attention_mask', 'labels'],
    "commonsense_qa": ['input_ids', 'attention_mask', 'labels'],
}

features_dict = {}
for task_name, dataset in dataset_dict.items():
    features_dict[task_name] = {}
    for phase, phase_dataset in dataset.items():
        features_dict[task_name][phase] = phase_dataset.map(
            convert_func_dict[task_name],
            batched=True,
            load_from_cache_file=False,
        )
        print(task_name, phase, len(phase_dataset), len(features_dict[task_name][phase]))
        features_dict[task_name][phase].set_format(
            type="torch",
            columns=columns_dict[task_name],
        )
        print(task_name, phase, len(phase_dataset), len(features_dict[task_name][phase]))

import dataclasses
from torch.utils.data.dataloader import DataLoader
# from transformers.training_args import is_tpu_available
# from transformers.trainer import get_tpu_sampler
from transformers.data.data_collator import DataCollator, InputDataClass,DefaultDataCollator
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler
from typing import List, Union, Dict


class NLPDataCollator(DataCollator):
    """
    Extending the existing DataCollator to work with NLP dataset batches
    """

    def __call__(self, features: List[Union[InputDataClass, Dict]]) -> Dict[str, torch.Tensor]:
        first = features[0]
        if isinstance(first, dict):
            # NLP data sets current works presents features as lists of dictionary
            # (one per example), so we  will adapt the collate_batch logic for that
            if "labels" in first and first["labels"] is not None:
                if first["labels"].dtype == torch.int64:
                    labels = torch.tensor([f["labels"] for f in features], dtype=torch.long)
                else:
                    labels = torch.tensor([f["labels"] for f in features], dtype=torch.float)
                batch = {"labels": labels}
            for k, v in first.items():
                if k != "labels" and v is not None and not isinstance(v, str):
                    batch[k] = torch.stack([f[k] for f in features])
            return batch
        else:
            # otherwise, revert to using the default collate_batch
            return self(features)


class StrIgnoreDevice(str):
    """
    This is a hack. The Trainer is going call .to(device) on every input
    value, but we need to pass in an additional `task_name` string.
    This prevents it from throwing an error
    """

    def to(self, device):
        return self


class DataLoaderWithTaskname:
    """
    Wrapper around a DataLoader to also yield a task name
    """

    def __init__(self, task_name, data_loader):
        self.task_name = task_name
        self.data_loader = data_loader

        self.batch_size = data_loader.batch_size
        self.dataset = data_loader.dataset

    def __len__(self):
        return len(self.data_loader)

    def __iter__(self):
        for batch in self.data_loader:
            batch["task_name"] = StrIgnoreDevice(self.task_name)
            yield batch


class MultitaskDataloader:
    """
    Data loader that combines and samples from multiple single-task
    data loaders.
    """

    def __init__(self, dataloader_dict):
        self.dataloader_dict = dataloader_dict
        self.num_batches_dict = {
            task_name: len(dataloader)
            for task_name, dataloader in self.dataloader_dict.items()
        }
        self.task_name_list = list(self.dataloader_dict)
        self.dataset = [None] * sum(
            len(dataloader.dataset)
            for dataloader in self.dataloader_dict.values()
        )

    def __len__(self):
        return sum(self.num_batches_dict.values())

    def __iter__(self):
        """
        For each batch, sample a task, and yield a batch from the respective
        task Dataloader.

        We use size-proportional sampling, but you could easily modify this
        to sample from some-other distribution.
        """
        task_choice_list = []
        for i, task_name in enumerate(self.task_name_list):
            task_choice_list += [i] * self.num_batches_dict[task_name]
        task_choice_list = np.array(task_choice_list)
        np.random.shuffle(task_choice_list)
        dataloader_iter_dict = {
            task_name: iter(dataloader)
            for task_name, dataloader in self.dataloader_dict.items()
        }
        for task_choice in task_choice_list:
            task_name = self.task_name_list[task_choice]
            yield next(dataloader_iter_dict[task_name])


class MultitaskTrainer(transformers.Trainer):

    def get_single_train_dataloader(self, task_name, train_dataset):
        """
        Create a single-task data loader that also yields task names
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        # if is_tpu_available():
        #     train_sampler = get_tpu_sampler(train_dataset)
        else:
            train_sampler = (
                RandomSampler(train_dataset)
                if self.args.local_rank == -1
                else DistributedSampler(train_dataset)
            )

        data_loader = DataLoaderWithTaskname(
            task_name=task_name,
            data_loader=DataLoader(
                train_dataset,
                batch_size=self.args.train_batch_size,
                sampler=train_sampler,
                collate_fn=self.data_collator.collate_batch,
            ),
        )

        # if is_tpu_available():
        #     data_loader = pl.ParallelLoader(
        #         data_loader, [self.args.device]
        #     ).per_device_loader(self.args.device)
        return data_loader

    def get_train_dataloader(self):
        """
        Returns a MultitaskDataloader, which is not actually a Dataloader
        but an iterable that returns a generator that samples from each
        task Dataloader
        """
        return MultitaskDataloader({
            task_name: self.get_single_train_dataloader(task_name, task_dataset)
            for task_name, task_dataset in self.train_dataset.items()
        })


train_dataset = {
    task_name: dataset["train"]
    for task_name, dataset in features_dict.items()
}
trainer = MultitaskTrainer(
    model=multitask_model,
    train_dataset=train_dataset,
    args=transformers.TrainingArguments(
        output_dir="./models/multitask_model",
        overwrite_output_dir=True,
        learning_rate=1e-5,
        do_train=True,
        num_train_epochs=3,
        # Adjust batch size if this doesn't fit on the Colab GPU
        per_device_train_batch_size=8,
        save_steps=3000,
    ),
    data_collator=NLPDataCollator,

)
trainer.train()


preds_dict = {}
for task_name in ["rte", "stsb", "commonsense_qa"]:
    eval_dataloader = DataLoaderWithTaskname(
        task_name,
        trainer.get_eval_dataloader(eval_dataset=features_dict[task_name]["validation"])
    )
    print(eval_dataloader.data_loader.collate_fn)
    preds_dict[task_name] = trainer._prediction_loop(
        eval_dataloader,
        description=f"Validation: {task_name}",
    )

preds_dict = {}
for task_name in ["rte", "stsb", "commonsense_qa"]:
    eval_dataloader = DataLoaderWithTaskname(
        task_name,
        trainer.get_eval_dataloader(eval_dataset=features_dict[task_name]["validation"])
    )
    print(eval_dataloader.data_loader.collate_fn)
    preds_dict[task_name] = trainer._prediction_loop(
        eval_dataloader,
        description=f"Validation: {task_name}",
    )
# Evalute RTE
# nlp.load_metric('glue', name="rte").compute(
#     np.argmax(preds_dict["rte"].predictions, axis=1),
#     preds_dict["rte"].label_ids,
# )
#
# # Evalute STS-B
# nlp.load_metric('glue', name="stsb").compute(
#     preds_dict["stsb"].predictions.flatten(),
#     preds_dict["stsb"].label_ids,
# )
#
# # Evalute Commonsense QA
# np.mean(
#     np.argmax(preds_dict["commonsense_qa"].predictions, axis=1)
#     == preds_dict["commonsense_qa"].label_ids
# )


# ##########
# class BertMultiTaskHeads(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.predictions = BertLMPredictionHead(config)
#         # I add additional linear layer to do binary prediciton
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)
#         self.binary_predictions = nn.Linear(config.hidden_size, 2)
#
#     def forward(self, sequence_output):
#         mlm_scores = self.predictions(sequence_output)
#         # additional entity score for binary prediction
#         entity_scores = self.binary_predictions(self.dropout(sequence_output))
#         return mlm_scores, entity_scores
# And I created BertForMultiTask class by following BertForMaskedLM (I have previously created a new BertMultiTaskOutput class for returning also the entity prediction loss) :
#
# class BertForMultiTask(BertPreTrainedModel):
#     _keys_to_ignore_on_load_unexpected = [r"pooler"]
#
#     def __init__(self, config):
#         super().__init__(config)
#         self.bert = BertModel(config, add_pooling_layer=False)
#         self.cls = BertMultiTaskHeads(config)
#         self.init_weights()
#
#     def forward(
#             self,
#             input_ids=None,
#             attention_mask=None,
#             token_type_ids=None,
#             position_ids=None,
#             head_mask=None,
#             inputs_embeds=None,
#             labels=None,
#             tag_labels=None,  # For entity binary prediction, obtained as an additional feature
#             output_attentions=None,
#             output_hidden_states=None,
#             return_dict=None,
#             **kwargs
#     ):
#
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict
#
#         outputs = self.bert(
#             input_ids,
#             attention_mask=attention_mask,
#             token_type_ids=token_type_ids,
#             position_ids=position_ids,
#             head_mask=head_mask,
#             inputs_embeds=inputs_embeds,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#         )
#
#         sequence_output = outputs[0]
#         prediction_scores, label_scores = self.cls(sequence_output)
#
#         total_loss = None
#         if labels is not None and tag_labels is not None:
#             loss_fct = nn.CrossEntropyLoss()
#             masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
#             # additional entity prediction loss
#             entity_prediction_loss = loss_fct(label_scores.view(-1, 2), tag_labels.view(-1))
#             total_loss = masked_lm_loss +  entity_prediction_loss
#
#         if not return_dict:
#             output = (prediction_scores, label_scores) + outputs[2:]
#             return ((total_loss,) + output) if total_loss is not None else output
#
#         return BertMultiTaskOutput(
#             loss=total_loss,
#             prediction_logits=prediction_scores,
#             entity_prediction_logits=label_scores,
#             hidden_states=outputs.hidden_states,
#             attentions=outputs.attentions,
#         )
#
# from datasets import M