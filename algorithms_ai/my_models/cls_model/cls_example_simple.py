"""
分类的样例
一个核心是数据，先定义数据的输入输出--任务格式
一个核心是模型，选择符合数据输入输出的模型结构（get_model），添加一些config，修改模型架构,定义输入输出
其次是数据的处理，模型的训练参数配置（评估指标等）

由分类头定义任务
二分类，多分类
多标签分类（多个二分类），导入模型时要配置num_labels=2,problem_type="multi_label_classification",
还有指标metric_for_multi_label_classification
回归用 problem_type == "regression"

"""
import os
import time
import torch
from tqdm import tqdm
from transformers import TrainingArguments, DataCollatorWithPadding
from transformers import pipeline

from algorithms_ai.my_models.model_utils import caculate_eval_steps_warmup_steps
from cls_trainer import CLSTrainerWithWeightedRandomSampler
from metric_for_cls import compute_metrics_for_single_label_classification


class SentenceCls:
    def __init__(self,
                 checkpoint=None,
                 cuda_devices='0,1',
                 wandb_project_name='test',
                 ):
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices
        if not wandb_project_name:
            os.environ["WANDB_DISABLED"] = "true"
            self.use_wandb = False
        else:
            os.environ["WANDB_DISABLED"] = "false"
            os.environ["WANDB_PROJECT"] = wandb_project_name
            self.use_wandb = True
        self.cuda_devices = cuda_devices
        self.cache_dir = './model_cache'

        if not checkpoint:
            self.checkpoint = "bert_base_multilingual_cased"
        else:
            self.checkpoint = checkpoint

        self.labels_list = ['positive', 'negetive']
        self.label2id = {j: i for i, j in enumerate(self.labels_list)}
        self.id2label = {i: j for i, j in enumerate(self.labels_list)}

        self.tokenizer = self.get_tokenizer(self.checkpoint)
        self.data_collator = self.get_data_collator(self.tokenizer)

        self.compute_metrics = self.get_compute_metrics(self.id2label)
        self.metric = 'eval_macro avg_f1-score'

    def get_model(self):
        """
        通过一些配置构建模型（包括架构和参数）
        :return: model
        """
        from transformers import BertForSequenceClassification
        model = BertForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=self.checkpoint,
            ignore_mismatched_sizes=False, output_loading_info=False,
            num_labels=len(self.label2id), id2label=self.id2label, label2id=self.label2id,
            cache_dir=self.cache_dir
        )
        # model.resize_token_embeddings(len(self.tokenizer))
        self.frozen_layers(model)
        return model

    # 冻结层
    def frozen_layers(self, model):
        for name, param in model.named_parameters():
            if name.startswith('bert.embeddings'):
                param.requires_grad = False
            if name.startswith('bert.encoder'):
                param.requires_grad = False

    def get_tokenizer(self, checkpoint):
        from transformers import BertTokenizerFast
        tokenizer = BertTokenizerFast.from_pretrained(pretrained_model_name_or_path=checkpoint,
                                                      cache_dir=self.cache_dir)
        tokenizer.model_max_length = 180  # 167
        return tokenizer

    def get_data_collator(self, tokenizer):
        return DataCollatorWithPadding(tokenizer=tokenizer,
                                       padding='max_length',
                                       max_length=tokenizer.model_max_length,
                                       pad_to_multiple_of=None,
                                       return_tensors="pt",
                                       )

    def get_train_args(self, output_dir='./test', train_dataset_length=100, num_train_epochs=2,
                       per_device_train_batch_size=4, wandb_run_name=None, **kwargs):
        cuda_devices_num = len(self.cuda_devices.split(','))
        eval_steps, warmup_steps = caculate_eval_steps_warmup_steps(train_dataset_length,
                                                                    num_train_epochs=num_train_epochs,
                                                                    per_device_train_batch_size=per_device_train_batch_size,
                                                                    warmup_ratio=0.1,
                                                                    cuda_devices_num=cuda_devices_num,
                                                                    gradient_accumulation_steps=1,
                                                                    eval_num=20)

        default_args = {
            'output_dir': output_dir,
            'overwrite_output_dir': True,
            'weight_decay': 0.01,

            'num_train_epochs': num_train_epochs,
            'per_device_train_batch_size': per_device_train_batch_size,

            'evaluation_strategy': 'steps',
            'eval_steps': eval_steps,
            'save_steps': eval_steps,
            'logging_steps': eval_steps,

            'dataloader_num_workers': 4,
            'load_best_model_at_end': True,
            'save_total_limit': 2,
            'metric_for_best_model': self.metric,
            'greater_is_better': True,

            'label_names': ['labels'],
            'logging_dir': os.path.join(output_dir, 'logs'),
            "optim": "adamw_torch",
            'warmup_steps': warmup_steps,
            'disable_tqdm': False
        }

        default_args.update(kwargs)

        if self.use_wandb and wandb_run_name:
            os.environ["WANDB_MODE"] = "online"
            c_t = time.localtime(time.time())
            default_args.update(
                {'report_to': ["wandb"],
                 'run_name': wandb_run_name + f'_{c_t.tm_mon}_{c_t.tm_mday}_{c_t.tm_hour}_{c_t.tm_min}'}
            )
        else:
            os.environ["WANDB_MODE"] = "offline"
        return TrainingArguments(**default_args)

    def get_compute_metrics(self, id2label, show_confusion_matrix=False, show_report=False, return_metric=None):
        from functools import partial
        return partial(
            compute_metrics_for_single_label_classification,
            id2label=id2label,
            show_confusion_matrix=show_confusion_matrix,
            return_metric=return_metric,
            show_report=show_report
        )

    def train(self, train_dataset, validation_dataset, num_train_epochs=3, per_device_train_batch_size=84,
              per_device_eval_batch_size=512,
              output_dir='./models/test', wandb_run_name='',
              ignore_warning=False,
              **train_args):
        if ignore_warning:
            # from transformers import logging
            # logging.set_verbosity_info()  # set logging level
            os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'  # fast tokenizer warning
            import warnings
            warnings.filterwarnings("ignore")  # torch warning
        model = self.get_model()
        train_args = self.get_train_args(train_dataset_length=train_dataset.num_rows,
                                         num_train_epochs=num_train_epochs,
                                         per_device_train_batch_size=per_device_train_batch_size,
                                         per_device_eval_batch_size=per_device_eval_batch_size,
                                         output_dir=output_dir,
                                         wandb_run_name=wandb_run_name,
                                         **train_args
                                         )
        trainer = CLSTrainerWithWeightedRandomSampler(
            model=model,
            args=train_args,
            train_dataset=train_dataset,
            eval_dataset=validation_dataset,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics,
        )
        trainer.train()

    def train_with_hp(self, train_dataset, validation_dataset, output_dir='./models/hp', wandb_run_name='',
                      ignore_warning=True):
        if ignore_warning:
            # from transformers import logging
            # logging.set_verbosity_info()  # set logging level
            os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'  # fast tokenizer warning
            import warnings
            warnings.filterwarnings("ignore")  # torch warning

        train_args = self.get_train_args(train_dataset_length=train_dataset.num_rows,
                                         num_train_epochs=3,
                                         per_device_train_batch_size=84,
                                         per_device_eval_batch_size=16,
                                         output_dir=output_dir,
                                         wandb_run_name=wandb_run_name)

        def optuna_hp_space(trial):
            return {
                "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
                # "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size",
                #                                                          [16, 32, 64, 128]),
            }

        trainer = CLSTrainerWithWeightedRandomSampler(
            model=None,
            model_init=self.get_model,
            args=train_args,
            train_dataset=train_dataset,
            eval_dataset=validation_dataset,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics,
        )

        def my_objective(metrics):
            return metrics[self.metric]

        best_trial = trainer.hyperparameter_search(
            backend="optuna",
            hp_space=optuna_hp_space,
            n_trials=3,
            compute_objective=my_objective,
            direction="maximize",
        )
        print(best_trial)

    def eval(self, validation_dataset, per_device_eval_batch_size=512, wandb_run_name='eval'):
        self.compute_metrics = self.get_compute_metrics(self.id2label, show_confusion_matrix=True,
                                                        show_report=True)
        train_args = self.get_train_args(train_dataset_length=validation_dataset.num_rows,
                                         per_device_eval_batch_size=per_device_eval_batch_size,
                                         wandb_run_name=wandb_run_name)
        model = self.get_model()
        trainer = CLSTrainerWithWeightedRandomSampler(
            model=model,
            args=train_args,
            train_dataset=validation_dataset,
            eval_dataset=validation_dataset,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics,
        )
        trainer.evaluate()

    def predict(self, text_list, predict_batch_size=2000, mode='logits'):
        if not hasattr(self, 'inference_model'):
            self.inference_model = pipeline('text-classification',
                                            tokenizer=self.get_tokenizer(self.checkpoint),
                                            model=self.get_model(),
                                            device_map="cuda:0" if torch.cuda.is_available() else "cpu")

        all_res = []
        if mode == 'logits':
            for i in tqdm(range(0, len(text_list), predict_batch_size), desc='predict for logits'):
                all_res.append(
                    self.inference_model.forward(
                        self.inference_model.preprocess(
                            inputs={'text': text_list[i:i + predict_batch_size], 'truncation': True, 'padding': True}))[
                        'logits'])
            all_res = torch.cat(all_res)
        else:
            for i in tqdm(range(0, len(text_list) - 1, predict_batch_size), desc='predict'):
                all_res.extend(self.inference_model(text_list[i:i + predict_batch_size]))

        return all_res

    def predict_2(self, text_list, predict_batch_size=2000, mode='logits'):
        tokenizer = self.get_tokenizer(self.checkpoint)
        model = self.get_model()
        import os
        import torch.distributed as dist
        def setup(rank, world_size):
            "Sets up the process group and configuration for PyTorch Distributed Data Parallelism"
            os.environ["MASTER_ADDR"] = 'localhost'
            os.environ["MASTER_PORT"] = "12355"

            # Initialize the process group
            dist.init_process_group("gloo", rank=rank, world_size=world_size)

        def cleanup():
            "Cleans up the distributed environment"
            dist.destroy_process_group()

        rank = 1
        world_size = 2
        setup(rank, world_size)
        model = model.to(rank)

        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model, device_ids=[rank])
        model.eval()
        print(1)
        # TODO: add accelerate to inference for ddp
        # for batch in eval_dataloader:
        #     batch = {k: v.to(device) for k, v in batch.items()}
        #     with torch.no_grad():
        #         outputs = model(**batch)
        #
        #     logits = outputs.logits
        #     predictions = torch.argmax(logits, dim=-1)
        #     metric.add_batch(predictions=predictions, references=batch["labels"])

    @staticmethod
    def get_train_date(output_dir):
        import os
        from transformers.trainer_callback import TrainerState
        ckpt_dirs = os.listdir(output_dir)
        ckpt_dirs = sorted(ckpt_dirs, key=lambda x: int(x.split('-')[1]))
        last_ckpt = ckpt_dirs[-1]
        print(f'last_checkpoint:{last_ckpt}')
        state = TrainerState.load_from_json(f"{output_dir}/{last_ckpt}/trainer_state.json")

        print(f'best_checkpoint:{state.best_model_checkpoint}')  # your best ckpoint.


if __name__ == '__main__':
    from data_example import get_train_validation

    pharma_cls = SentenceCls(
        checkpoint="/large_files/5T/huggingface_cache/pretrained_model/bert_base_multilingual_cased/",
        cuda_devices='0,1',
        wandb_project_name='test')
    train, validation = get_train_validation(tokenizer=pharma_cls.tokenizer)

    pharma_cls.train(train.select(range(500)).remove_columns(["sentence", "idx"]),
                     validation.select(range(100)).remove_columns(["sentence", "idx"]),
                     per_device_eval_batch_size=512,
                     output_dir='./test', wandb_run_name='tt', ignore_warning=True)

    # pharma_cls.train_with_hp(train.select(range(500)).remove_columns(["sentence", "idx"]),
    #                          validation.select(range(100)).remove_columns(["sentence", "idx"]),
    #                          output_dir = './models/hp', wandb_run_name = 'test',
    #                           )

    #########
    # pharma_cls = SentenceCls(
    #     checkpoint="/home/zyl/disk/algorithms_ai/algorithms_ai/my_models/cls_model/models/hp/run-2/checkpoint-9",
    #     cuda_devices='0,1',
    #     wandb_project_name='test')
    # train, validation = get_train_validation(tokenizer=pharma_cls.tokenizer)
    # pharma_cls.eval(validation, wandb_run_name='')

    ######

    # t_l = ['positive_precision', 'eval/negetive_precision']
    # s = pharma_cls.predict(t_l)
    # print(s)
