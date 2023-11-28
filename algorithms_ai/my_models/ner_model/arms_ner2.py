import csv
import math
import os
import warnings

import evaluate
import pandas as pd
import wandb
from datasets import Dataset, DatasetDict, load_from_disk
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")
# tf.get_logger().setLevel(logging.ERROR)

os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# logging.getLogger('tensorflow').disabled = True

from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
from transformers import AutoTokenizer, DataCollatorForTokenClassification

import re

max_seq_length = 148


class WordTokenizer:
    def __init__(self, regex=None, keep_words=()):
        # 规则切词，可以保留哪些词不用切分
        # zh_punctuation = '＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､\u3000、〃〈〉《》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏﹑﹔·！？｡。'
        # en_punctuation = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
        # punctuations = ENGLISH_PUNCTUATION + CHINESE_PUNCTUATION

        html_escape = ['&quot;', '&amp;', '&lt;', '&gt;', '&nbsp;']
        # 同一个字符开头满足多个规则，以第一个规则作数，
        if not regex:
            ENGLISH_PUNCTUATION = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
            CHINESE_PUNCTUATION = '＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､\u3000、〃〈〉《》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏﹑﹔·！？｡。'
            PUNCTUATIONS = ENGLISH_PUNCTUATION + CHINESE_PUNCTUATION
            # 数字（整数和小数）| 英文单词 | 空白符 | 中文 | 希腊大写字母 | 希腊小写字母 | 标点 | 其他字符
            regex = f"\d+\.?\d+|[A-Za-z]+|\s+|[\u4e00-\u9fa5]|[\u0391-\u03a9]|[\u03b1-\u03c9]|[{PUNCTUATIONS}]|[^a-zA-Z0-9{PUNCTUATIONS}\s\u4e00-\u9fa5\u03b1-\u03c9\u0391-\u03a9]"
            regex = '|'.join(html_escape) + '|' + regex
        if keep_words:
            regex = '|'.join(keep_words) + '|' + regex
        self.find_re = re.compile(regex)

    def run(self, text):
        # 输入文本，输出切分后的文本以及坐标--start-end--- 最后一个字符的坐标，而不是索引
        res = []
        tmp_index = 0
        for i in self.find_re.finditer(text):
            if tmp_index < i.start():
                if text[tmp_index:i.start()].strip():
                    pass
                    res.append(
                        {
                            'start_offset': tmp_index,
                            'end_offset': i.start() - 1,
                            'text': text[tmp_index:i.start()]
                        }

                    )
            if i.group().strip():
                res.append(
                    {
                        'start_offset': i.start(),
                        'end_offset': i.end() - 1,
                        'text': i.group()
                    }
                )
            tmp_index = i.end()

        if tmp_index < len(text):
            if text[tmp_index:len(text) - 1].strip():
                res.append(
                    {
                        'start_offset': tmp_index,
                        'end_offset': len(text) - 1,
                        'text': text[tmp_index:len(text)]
                    }
                )
        return res


def convert_text_labels_to_ner_labels(text, labels, tokenizer: WordTokenizer):
    labels = sorted(labels, key=lambda x: x['start_offset'])
    labels.append({"start_offset": len(text) + 3, "end_offset": len(text) + 3, "text": "", "label": "other"})
    tokens = tokenizer.run(text)

    ner_tags = []
    label_index = 0
    tmp_label = labels[label_index]

    for token in tokens:
        if tmp_label['start_offset'] > token['end_offset']:
            ner_tags.append('O')
        else:  # tmp_label['start_offset']<=token['end_offset']
            if not ner_tags or ner_tags[-1] == 'O':
                ner_tags.append('B-' + tmp_label['label'])
            else:
                ner_tags.append('I-' + tmp_label['label'])

            if tmp_label['end_offset'] <= token['end_offset']:
                label_index += 1
                tmp_label = labels[label_index]
    return [i['text'] for i in tokens], ner_tags


import numpy as np


def get_dataset_dict(tokenizer, label2id, use_cache=True):
    word_tokenizer = WordTokenizer()

    def yield_data(file):
        data = pd.read_csv(file, sep='\t', quoting=csv.QUOTE_NONE).to_dict(
            orient='records')
        progress_bar = tqdm(range(len(data)))
        for i in data:
            i['doc_id'] = '-1' if not isinstance(i['doc_id'], str) else i['doc_id']
            i['category'] = 'UNASSIGNED' if not isinstance(i['category'], str) else i['category']
            i['sample_id'] = '-1' if not isinstance(i['sample_id'], str) else i['sample_id']
            i['raw_input'] = '-1' if not isinstance(i['raw_input'], str) else i['raw_input']
            i['text_section'] = '-1' if not isinstance(i['text_section'], str) else i['text_section']
            i['sent_start_offset'] = -1 if np.isnan(i['sent_start_offset']) else i['sent_start_offset']
            i['checked'] = False if not isinstance(i['checked'], bool) else i['checked']
            i['arms_ner'] = eval(i['arms_ner'])
            i['tokens'], i['ner_tags'] = convert_text_labels_to_ner_labels(i['input_text'],
                                                                           i['arms_ner'],
                                                                           word_tokenizer)
            i['ner_tags'] = [label2id[i] for i in i['ner_tags']]
            yield i
            progress_bar.update(1)

    disk_file = "/home/zyl/disk/PharmAI/Pipelines/components/deepmed_ctrain/data/V_3_3_2/arms_ner"
    if use_cache:
        dataset_dict = load_from_disk(disk_file,
                                      keep_in_memory=True,  # 放到内存中，会加速，否则直接磁盘映射来使用
                                      # storage_options: Optional[dict] = None
                                      )
        return dataset_dict

    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

        labels = []
        for i, label in enumerate(examples[f"ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:  # Set the special tokens to -100.
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            labels.append(label_ids)
        tokenized_inputs["labels"] = labels

        return tokenized_inputs

    train_file = '/home/zyl/disk/PharmAI/Pipelines/components/deepmed_ctrain/data/V_3_3/train.arms_ner.tsv'
    data_cache = "/large_files/5T/huggingface_cache/data"
    train_dataset = Dataset.from_generator(generator=yield_data,
                                           # features= features,  # 特征Features 自定义
                                           cache_dir=data_cache,  # cache路径
                                           keep_in_memory=False,  # 是否存进内存
                                           gen_kwargs={'file': train_file},  # 生成器的参数
                                           num_proc=1,  # 存储和读取时的多进程数
                                           )
    train_dataset = train_dataset.shuffle(seed=123)  # 重排
    train_dataset = train_dataset.map(tokenize_and_align_labels, batched=True)

    eval_file = '/home/zyl/disk/PharmAI/Pipelines/components/deepmed_ctrain/data/V_3_3/dev.arms_ner_checked.tsv'
    eval_dataset = Dataset.from_generator(generator=yield_data,
                                          # features= features,  # 特征Features 自定义
                                          cache_dir=data_cache,  # cache路径
                                          keep_in_memory=False,  # 是否存进内存
                                          gen_kwargs={'file': eval_file},  # 生成器的参数
                                          num_proc=1,  # 存储和读取时的多进程数
                                          )
    eval_dataset = eval_dataset.map(tokenize_and_align_labels, batched=True)
    eval_dataset = eval_dataset.shuffle(seed=123)  # 重排

    dataset_dict = DatasetDict({'train': train_dataset, 'eval': eval_dataset})
    dataset_dict['train'].info.update(dataset_dict['train'].info.from_dict({"description": '训练集'}))
    dataset_dict['eval'].info.update(dataset_dict['eval'].info.from_dict({"description": '评估集'}))

    dataset_dict.save_to_disk(disk_file, num_proc=3)
    return dataset_dict


seqeval = evaluate.load("seqeval")

from functools import partial


def compute_metrics_for_ner(eval_pred, id2label):
    """
    eval_pred 第一个值是模型预测的logits，第二个是输入的label张量
    然后变成
    [['O', 'O', 'O', 'B-arm_option',  'I-shared_arm_option', 'I-shared_arm_option', 'I-shared_arm_option', 'O'],
    ['O', 'O', 'O', 'O', 'B-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option', 'I-arm_option',]]

    然后计算
    """
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

    true_predictions2 = [
        [id2label[p] for (p, l) in zip(prediction, label)]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels2 = [
        [id2label[l] for (p, l) in zip(prediction, label)]
        for prediction, label in zip(predictions, labels)
    ]

    print('#'*40)
    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    print(results)
    print(seqeval.compute(predictions=true_predictions2, references=true_labels2))
    return results


def get_args(
        output_dir,
        metric_for_best_model,
        dataset_length=0,
        logging_dir='./logs',
        gpus='0',
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        label_name='labels',
        wandb_run_name='test',
        **kwargs
):
    default_args = {
        # data
        'dataloader_drop_last': False,  # 是否删除最后一个不完整的batch
        'dataloader_num_workers': 0,  # 多进程处理数据
        'remove_unused_columns': True,  # 自动取出多余的列
        # 'label_names':  # label相关的列的名称，List[str]，默认["label"],如果是问答则是["start_positions", "end_positions"]
        # 'disable_tqdm': # 是否关闭进度和metric
        # 'past_index'
        # 'ignore_data_skip':False
        'group_by_length': False,  # 动态padding时使用，
        'dataloader_pin_memory': True,  # 固定数据的内存

        # train
        'num_train_epochs': 3,  # 训练epoch
        'max_steps': -1,  # 总的迭代次数，如果设置会覆盖epoch
        'per_device_train_batch_size': 8,  # 每个设备（可能是GPU）的训练batch大小
        'gradient_accumulation_steps': 1,  # 梯度累计step数
        'weight_decay': 0,  # 权重衰减，L2正则化
        'max_grad_norm': 1,  # 梯度截断，控制梯度膨胀
        'no_cuda': False,  # 是否使用GPU
        'seed': 42,  # 训练时的种子
        # 'data_seed': # 数据种子，如果不设置就默认为训练的种子
        # 'deepspeed',fsdp,sharded_ddp,fp16,resume_from_checkpoint,gradient_checkpointing，auto_find_batch_size，
        # full_determinism，torchdynamo

        # 优化器和学习器
        'optim': 'adamw_hf',
        # 'optim_args':  # 参数
        'learning_rate': 5e-5,  # lr,默认使用AdamW优化器
        'lr_scheduler_type': 'linear',  # 学习率优化
        'adam_beta1': 0.9,
        'adam_beta2': 0.999,
        'adam_epsilon': 1e-8,
        'warmup_ratio': 0.0,  # 预热的整个step的比例
        'warmup_steps': 0,  # 预热的步数，如果设置，会覆盖比例

        # evaluate
        'evaluation_strategy': 'steps',  # 默认若干步评估一次，同时log输出一次
        # 'eval_steps':  # 评估的间隔step数，如果不设置，就会和logging_steps一样
        'per_device_eval_batch_size': 8,  # 每个设备（可能是GPU）的评估batch大小
        # 'eval_accumulation_steps': , # 在把结果放到cpu上前，评估累计的次数，不设置就评估整个评估集
        # 'eval_delay': # 评估的延迟步数，一般不用
        'load_best_model_at_end': False,  # 最后载入最好的指标的那次模型，
        # 如果设置为True，save_strategy要和evaluation_strategy一样，`save_steps` 必须是 `eval_steps`的整数倍
        # 'metric_for_best_model':, # 最好模型的指标，主要要加上eval_ 前缀
        # 'greater_is_better':True # 模型指标的选择
        'include_inputs_for_metrics': False,  # 是否把输入传给metric计算

        # log
        'logging_dir': './output_dir/runs',  # log日志保存在哪
        'logging_strategy': 'steps',  # 日志的保存策略，默认steps
        'logging_steps': 500,  # log策略是steps时的步骤数
        'log_level': 'passive',  # 主进程的log的等级
        'log_level_replica': 'warning',  # 重复log的等级
        'log_on_each_node': True,  # 分布式训练是否log每个节点
        'logging_first_step': False,  # 是否评估和log开始时的结果
        'logging_nan_inf_filter': True,  # 是否过滤空值和无穷小的loss
        # 'report_to':  上传到哪
        # 'run_name':  名称
        'skip_memory_metrics': True,  # 是否跳过内存检查
        'push_to_hub': False,

        # save-model
        'overwrite_output_dir': False,  # 是否覆写输出的路径
        'save_strategy': "steps",  # 保存策略，默认steps
        'save_steps': 500,  # steps时保存的步骤
        'save_total_limit': 2,  # 最大存储的模型数
        'save_on_each_node': False,  # 多节点训练时，是否在每个节点上存储模型，还是只在主节点上存储

        # predict
        "prediction_loss_only": False,  # 是否预测时只输出损失
        'jit_mode_eval': False,  # 是否使用jit进行inference
    }
    # 设置经过若干steps进行评估，log和save

    if dataset_length <= 0:
        eval_steps = 20
        warmup_steps = 20
    else:
        gpu_num = len(gpus.strip(','))
        # 近似最大更新步数
        sim_max_update_steps = math.ceil(
            num_train_epochs * max((dataset_length // (per_device_train_batch_size * gpu_num) // 1), 1))

        eval_steps = int(max(sim_max_update_steps // 20, 1))  # 每次训练都评估20次
        warmup_steps = int(max(sim_max_update_steps * 0.1, 1))

    default_args.update(
        {
            'output_dir': output_dir,
            'learning_rate': learning_rate,
            'num_train_epochs': num_train_epochs,
            'per_device_train_batch_size': per_device_train_batch_size,
            'per_device_eval_batch_size': per_device_eval_batch_size,
            'weight_decay': 0.01,
            'max_grad_norm': 1,
            'eval_steps': eval_steps,
            'save_steps': eval_steps,
            'logging_steps': eval_steps,

            'dataloader_num_workers': 4,
            'load_best_model_at_end': True,
            'save_total_limit': 2,
            'metric_for_best_model': metric_for_best_model,
            'greater_is_better': True,
            'group_by_length': True,
            'report_to': ["wandb"],
            'run_name': wandb_run_name,
            'label_names': [label_name],
            'logging_dir': logging_dir,
            'warmup_steps': warmup_steps
        }
    )
    default_args.update(kwargs)
    return TrainingArguments(**default_args)


class ArmsNer:
    def __init__(self, gpus='6,7,8'):
        # checkpoint = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
        # checkpoint ="/large_files/5T/huggingface_cache/model/models--microsoft--BiomedNLP-PubMedBERT-base-uncased-abstract/"
        self.checkpoint = "/large_files/5T/huggingface_cache/pretrained_model/microsoft--BiomedNLP-PubMedBERT-base-uncased-abstract/"
        # checkpoint = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
        self.model_cache_dir = "/large_files/5T/huggingface_cache/model"

        self.label2id = {
            "O": 0,
            "B-arm_count": 1,
            "I-arm_count": 2,
            "B-arm_option": 3,
            "I-arm_option": 4,
            "B-shared_arm_option": 5,
            "I-shared_arm_option": 6,
        }
        self.id2label = {0: "O",
                         1: "B-arm_count",
                         2: "I-arm_count",
                         3: "B-arm_option",
                         4: "I-arm_option",
                         5: "B-shared_arm_option",
                         6: "I-shared_arm_option",
                         }

        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=self.checkpoint,
                                                       cache_dir=self.model_cache_dir)
        self.tokenizer.model_max_length = 148

        self.dataset_dict = get_dataset_dict(self.tokenizer, self.label2id, use_cache=True)
        self.data_collator = DataCollatorForTokenClassification(tokenizer=self.tokenizer,
                                                                padding='max_length',
                                                                max_length=self.tokenizer.model_max_length,
                                                                pad_to_multiple_of=None,
                                                                return_tensors="pt",
                                                                )

        self.compute_metrics = partial(compute_metrics_for_ner, id2label=self.id2label)
        self.metric = 'eval_overall_f1'
        self.get_train_args = partial(get_args,
                                      output_dir="/home/zyl/disk/PharmAI/Pipelines/components/deepmed_ctrain/output/arms_ner/models",
                                      metric_for_best_model=self.metric,
                                      logging_dir="/home/zyl/disk/PharmAI/Pipelines/components/deepmed_ctrain/output/arms_ner/logs",
                                      gpus=gpus,
                                      learning_rate=2e-5,
                                      label_name='labels',
                                      wandb_run_name='arms_ner',
                                      )
        print(1)

    def get_model(self):
        return AutoModelForTokenClassification.from_pretrained(
            self.checkpoint, num_labels=len(self.label2id), id2label=self.id2label, label2id=self.label2id,
            cache_dir=self.model_cache_dir
        )

    def test_train(self):
        wandb.init(project='deepmed')
        train_dataset = self.dataset_dict['train']
        model = self.get_model()
        trainer = Trainer(
            model=model,
            args=self.get_train_args(dataset_length=train_dataset.num_rows,
                                    num_train_epochs=10,
                                    per_device_train_batch_size=168,
                                    per_device_eval_batch_size=8),
            train_dataset=train_dataset,
            eval_dataset=self.dataset_dict['eval'].select(range(8)),
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics,
        )
        trainer.train()

    def train_with_hp(self, n_trials=5):
        train_dataset = self.dataset_dict['train']
        self.train_args = self.get_train_args(dataset_length=train_dataset.num_rows,
                                              num_train_epochs=20,
                                              wandb_run_name='arms_ner',
                                              per_device_train_batch_size=168,
                                              per_device_eval_batch_size=32,
                                              )
        trainer = Trainer(
            model=None,
            model_init=self.get_model,
            args=self.train_args,
            train_dataset=self.dataset_dict['train'],
            eval_dataset=self.dataset_dict['eval'],
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics,
        )

        def wandb_hp_space(trial):
            return {
                "method": "bayes",
                "metric": {"name": self.metric, "goal": "maximize"},
                "parameters": {
                    "learning_rate": {"distribution": "uniform", "min": 1e-7, "max": 1e-4},
                    # "weight_decay": {"value": 20},
                    # "max_grad_norm":{"value": 20},
                    # "warmup_ratio":{'max': 0.1, 'min': 0.0001},
                    "num_train_epochs": {"values": [10, 20, 30, 50]}
                },
                "name": "arms_ner",
                "project": 'deepmed'
            }

        def my_objective(metrics):
            return metrics[self.metric]

        best_trial = trainer.hyperparameter_search(
            backend="wandb",
            hp_space=wandb_hp_space,
            n_trials=n_trials,
            compute_objective=my_objective,
            metric=self.metric,
            direction="maximize",
        )
        print(best_trial)


if __name__ == '__main__':
    os.environ["WANDB_MODE"] = "offline"
    gpus = "8"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    ArmsNer(gpus=gpus).test_train()

    # ArmsNer(gpus=gpus).train_with_hp(n_trials=10)
