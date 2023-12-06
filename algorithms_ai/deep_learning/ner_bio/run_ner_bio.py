import os
import time
import warnings

os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
warnings.filterwarnings("ignore")


def get_tokenizer(checkpoint="model_path",
                  model_max_length=128,
                  special_tokens=None):
    from transformers import AutoTokenizer
    tokenizer_cache = '/large_files/5T/huggingface_cache/tokenizer_cache/'
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=checkpoint,
                                              cache_dir=tokenizer_cache)
    if model_max_length:
        tokenizer.model_max_length = model_max_length
    if special_tokens:
        tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
    return tokenizer


def get_data_collator(tokenizer):
    from transformers import DataCollatorForTokenClassification
    return DataCollatorForTokenClassification(tokenizer=tokenizer,
                                              padding='max_length',
                                              max_length=tokenizer.model_max_length,
                                              pad_to_multiple_of=None,
                                              return_tensors="pt")


class NER_BIO:
    def __init__(self, entity2id, model_path):
        self.entity2id = entity2id
        self.entities = [i[0] for i in sorted(self.entity2id.items(), key=lambda x: x[-1])]
        self.label2id = {"O": 0}
        for i, j in enumerate(self.entities):
            self.label2id[f'B-{j}'] = 2 * (i + 1) - 1
            self.label2id[f'I-{j}'] = 2 * (i + 1)

        self.model_path = model_path

    @staticmethod
    def get_dataset_dict(train_data=None, tokenizer=None, entities=None, label2id=None, val_data=None,
                         dataset_dict_dir='./test_data', use_file_cache=True):

        from zyl_utils.deep_learning.ner_bio.data_process_ner_bio import process_one_sample, tokenize_and_align_labels
        from zyl_utils.dataset_utils.dataset_utils import save_and_load_dataset_dict
        from datasets import DatasetDict, Dataset
        from zyl_utils.utils.tokenizer_utils.regex_tokenizer import RegexTokenizer
        pre_tokenizer = RegexTokenizer().run

        @save_and_load_dataset_dict(dataset_dir=dataset_dict_dir)
        def get_dataset_dict_func():
            train_dataset = [process_one_sample(i, pre_tokenizer, entities, label2id, keep_features=None) for i in
                             train_data]
            train_dataset = [_ for _ in train_dataset if _]
            train_dataset = Dataset.from_list(train_dataset)
            train_dataset = train_dataset.map(tokenize_and_align_labels, batched=True,
                                              fn_kwargs={"tokenizer": tokenizer})

            if val_data:
                val_dataset = [process_one_sample(i, pre_tokenizer, entities, label2id, keep_features=None) for i in
                               val_data]
                val_dataset = [_ for _ in val_dataset if _]
                val_dataset = Dataset.from_list(val_dataset)
                val_dataset = val_dataset.map(tokenize_and_align_labels, batched=True,
                                              fn_kwargs={"tokenizer": tokenizer})
                dt = DatasetDict({'train_for_model': train_dataset,
                                  'val_for_model': val_dataset})
            else:
                dt = DatasetDict({'train_for_model': train_dataset})

            return dt

        return get_dataset_dict_func(use_file_cache=use_file_cache)

    @staticmethod
    def init_model(label2id=None, checkpoint_path='bert_base_uncased', vocab_size=None):
        from transformers import AutoModelForTokenClassification
        model = AutoModelForTokenClassification.from_pretrained(
            checkpoint_path, num_labels=len(label2id), id2label={j: i for i, j in label2id.items()}, label2id=label2id
        )
        if vocab_size:
            if model.bert.embeddings.word_embeddings.weight.size()[1] != vocab_size:
                model.bert.resize_token_embeddings(new_num_tokens=vocab_size)
        return model

    @staticmethod
    def get_train_args(cuda_devices='0', output_dir='./test', metric=None, train_dataset_length=100, num_train_epochs=2,
                       per_device_train_batch_size=4, wandb_project_name=None, wandb_run_name=None, **kwargs):
        if cuda_devices:
            cuda_devices_num = len(cuda_devices.split(','))
            use_cpu = False
        else:
            cuda_devices_num = 1
            use_cpu = True

        gradient_accumulation_steps = 1
        eval_num = 10  # 每轮评估10次
        eval_steps = int(max((train_dataset_length // (
                per_device_train_batch_size * cuda_devices_num) // gradient_accumulation_steps // eval_num), 1))

        default_args = {
            'output_dir': output_dir,
            'overwrite_output_dir': True,

            'num_train_epochs': num_train_epochs,
            'per_device_train_batch_size': per_device_train_batch_size,

            'evaluation_strategy': 'steps',
            'eval_steps': eval_steps,
            'save_steps': eval_steps,
            'logging_steps': 1,

            'dataloader_num_workers': 4,
            'load_best_model_at_end': True,
            'save_total_limit': 3,

            'label_names': ['labels'],
            'logging_dir': os.path.join(output_dir, 'logs'),
            'disable_tqdm': False,

            'use_cpu': use_cpu,
        }
        default_args.update(kwargs)

        if wandb_project_name:
            os.environ["WANDB_DISABLED"] = "false"
            os.environ["WANDB_PROJECT"] = wandb_project_name
            if wandb_run_name:
                os.environ["WANDB_MODE"] = "online"
                c_t = time.localtime(time.time())
                default_args.update(
                    {'report_to': ["wandb"],
                     'run_name': wandb_run_name + f'_{c_t.tm_mon}_{c_t.tm_mday}_{c_t.tm_hour}_{c_t.tm_min}'}
                )
            else:
                os.environ["WANDB_MODE"] = "offline"
        else:
            os.environ["WANDB_DISABLED"] = "true"

        if metric:
            default_args.update(
                {'metric_for_best_model': metric,
                 'greater_is_better': True}
            )

        from transformers import TrainingArguments
        return TrainingArguments(**default_args)

    @staticmethod
    def get_compute_metrics(id2label):
        from functools import partial
        from zyl_utils.deep_learning.ner_bio.metric_ner_bio import compute_metrics_for_ner
        return partial(
            compute_metrics_for_ner,
            id2label=id2label
        )

    def train(self,
              train_data,
              val_data,
              dataset_dict_dir='./test_data',
              use_file_cache=True,
              model_max_length=128,
              special_tokens=None,

              cuda_devices='0,1',
              wandb_project_name=None,
              wandb_run_name=None,
              num_train_epochs=3,
              per_device_train_batch_size=8,
              per_device_eval_batch_size=8,
              output_dir='./models/test',

              metric=None,

              **train_args
              ):
        if cuda_devices:
            os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices

        tokenizer = get_tokenizer(
            checkpoint=self.model_path,
            model_max_length=model_max_length,
            special_tokens=special_tokens
        )

        data_collator = get_data_collator(tokenizer=tokenizer)

        model = self.init_model(label2id=self.label2id,
                                checkpoint_path=self.model_path,
                                vocab_size=len(tokenizer))

        dataset_dict = self.get_dataset_dict(
            train_data=train_data, tokenizer=tokenizer, entities=self.entities, label2id=self.label2id,
            val_data=val_data, dataset_dict_dir=dataset_dict_dir, use_file_cache=use_file_cache
        )
        train_dataset = dataset_dict['train_for_model']
        validation_dataset = dataset_dict['val_for_model']

        train_args = self.get_train_args(
            cuda_devices=cuda_devices, output_dir=output_dir, metric=metric, train_dataset_length=100,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            wandb_project_name=wandb_project_name, wandb_run_name=wandb_run_name,
            # include_inputs_for_metrics=False,
            # remove_unused_columns=False,
            **train_args
        )

        if metric:
            compute_metrics = self.get_compute_metrics(id2label={j: i for i, j in self.label2id.items()})
        else:
            compute_metrics = None
        from transformers import Trainer
        trainer = Trainer(
            model=model,
            args=train_args,
            train_dataset=train_dataset,
            eval_dataset=validation_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )
        trainer.train()

    def eval(self, test_data, cuda_devices,
             metric='eval_sum_micro_score', per_device_eval_batch_size=8, wandb_project_name=None, wandb_run_name=None):
        if cuda_devices:
            os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        tokenizer = get_tokenizer(checkpoint=self.model_path)
        data_collator = get_data_collator(tokenizer=tokenizer)
        model = self.init_model(label2id=self.label2id,
                                checkpoint_path=self.model_path)

        from datasets import Dataset
        from zyl_utils.utils.tokenizer_utils.regex_tokenizer import RegexTokenizer
        from zyl_utils.deep_learning.ner_bio.data_process_ner_bio import process_one_sample, tokenize_and_align_labels

        pre_tokenizer = RegexTokenizer().run

        test_data = [process_one_sample(i, pre_tokenizer, self.entities, self.label2id, keep_features=None)
                     for i in test_data]
        test_data = [_ for _ in test_data if _]
        test_data = Dataset.from_list(test_data)
        test_data = test_data.map(tokenize_and_align_labels, batched=True,
                                  fn_kwargs={"tokenizer": tokenizer})

        train_args = self.get_train_args(
            cuda_devices=cuda_devices, output_dir='./tmp', metric=metric, train_dataset_length=100,
            num_train_epochs=1, per_device_train_batch_size=4, per_device_eval_batch_size=per_device_eval_batch_size,
            wandb_project_name=wandb_project_name, wandb_run_name=wandb_run_name,
        )

        if metric:
            compute_metrics = self.get_compute_metrics(id2label={j: i for i, j in self.label2id.items()})
        else:
            compute_metrics = None

        from transformers import Trainer
        trainer = Trainer(
            model=model,
            args=train_args,
            train_dataset=None,
            eval_dataset=test_data,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )
        trainer.evaluate(test_data)

    def inference(self, to_predicts, cuda_devices, per_device_eval_batch_size):
        if cuda_devices:
            os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        if not hasattr(self, 'inferencer'):
            self.tokenizer = get_tokenizer(checkpoint=self.model_path)
            data_collator = get_data_collator(tokenizer=self.tokenizer)
            self.model = self.init_model(label2id=self.label2id,
                                         checkpoint_path=self.model_path)

            train_args = self.get_train_args(
                cuda_devices=cuda_devices, output_dir='./tmp', metric=None, train_dataset_length=100,
                num_train_epochs=1, per_device_train_batch_size=4,
                per_device_eval_batch_size=per_device_eval_batch_size,
                wandb_project_name=None, wandb_run_name=None, label_names=[],
            )

            from transformers import Trainer
            self.inferencer = Trainer(
                model=self.model,
                args=train_args,
                train_dataset=None,
                eval_dataset=None,
                tokenizer=self.tokenizer,
                data_collator=data_collator,
            )
        import numpy as np
        from datasets import Dataset
        from zyl_utils.deep_learning.ner_bio.ner_bio_utils import get_entity_by_bio

        test_data = self.tokenizer(to_predicts, truncation=True, return_offsets_mapping=True)
        test_data = Dataset.from_dict(test_data)

        all_preds = self.inferencer.predict(test_data)
        all_preds = np.argmax(all_preds.predictions, axis=-1)

        id2label = {j: i for i, j in self.label2id.items()}
        refined_data = []
        for i, d in enumerate(test_data):
            input_text = to_predicts[i]
            pred = all_preds[i]
            pred = [id2label[_] for _ in pred]
            pred_entities = get_entity_by_bio(pred)

            refined_res = {'input_text': input_text}
            refined_res.update({_: [] for _ in self.entities})

            for each_entity in pred_entities:
                if each_entity[0][0] >= len(d['offset_mapping']):
                    continue

                start_offset = d['offset_mapping'][each_entity[0][0]][0]
                end_offset = d['offset_mapping'][min(each_entity[0][-1], len(d['offset_mapping']) - 1)][-1] - 1
                text = input_text[start_offset:end_offset + 1]
                refined_res[each_entity[1]].append(
                    [{
                        'start_offset': start_offset,
                        'end_offset': end_offset,
                        'text': text
                    }]
                )
            refined_data.append(refined_res)

        return refined_data


if __name__ == '__main__':
    pass
