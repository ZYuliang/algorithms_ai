"""
语言模型
预训练模型
1. mll ,mask词填空
2. nsp  ，下一句预测：句子之间的关系和上下文的连贯性
3. 生成，语言模型预测：在这个任务中，模型需要根据给定的上下文预测下一个单词或字符。
模型会根据前面的文本内容预测最可能出现的下一个单词。这个任务可以帮助模型学习到语言的概率分布和上下文信息
4. 相似文本匹配，KL散度相似
5. 词性标注
6. 短语识别，实体链接预测--可以使用字典，注意只能做预训练任务，实际放在下游任务中

"""
from transformers import DataCollatorForLanguageModeling, TrainingArguments, Trainer

from transformers import BertForMaskedLM
import os

class PretrainedModelForNews:
    def __init__(self, gpus, load_dataset=True):
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus
        self.gpus = gpus
        self.checkpoint = "/large_files/5T/huggingface_cache/pretrained_model/bert_base_multilingual_cased/"
        self.tokenizer = self.get_tokenizer(self.checkpoint)

        self.data_collator = self.get_data_collator(self.tokenizer)
        if load_dataset:
            train_dataset, validation_dataset = get_train_eval_dataset(create_raw_dataset=False,
                                                                                 remove_cache=False)
            self.train_dataset=self.process_for_model(tokenizer=self.tokenizer,sub_dataset=train_dataset)
            self.validation_dataset = self.process_for_model(tokenizer=self.tokenizer,sub_dataset=validation_dataset)

    @staticmethod
    def get_tokenizer(checkpoint):
        from transformers import BertTokenizerFast
        tokenizer = BertTokenizerFast.from_pretrained(pretrained_model_name_or_path=checkpoint,
                                                      cache_dir="/large_files/5T/huggingface_cache/model")

        special_tokens_dict = {'additional_special_tokens': ['[NUM]']}
        tokenizer.add_special_tokens(special_tokens_dict)
        tokenizer.model_max_length = 180  # 167

        return tokenizer

    def get_data_collator(self, tokenizer):
        return DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

    def get_model(self):
        from transformers import BertForMaskedLM
        model = BertForMaskedLM.from_pretrained(
            self.checkpoint,
            cache_dir="/large_files/5T/huggingface_cache/model"
        )
        model.resize_token_embeddings(len(self.tokenizer))
        return model

    def get_train_args(self, train_dataset_length=100, num_train_epochs=2, per_device_train_batch_size=4, gpu_num=1,
                       wandb_run_name='test', output_dir='./test', **kwargs):
        eval_steps, warmup_steps = caculate_eval_steps_warmup_steps(train_dataset_length,
                                                                    num_train_epochs=num_train_epochs,
                                                                    per_device_train_batch_size=per_device_train_batch_size,
                                                                    warmup_ratio=0.1,
                                                                    gpu_num=gpu_num,
                                                                    gradient_accumulation_steps=1,
                                                                    eval_num=20)
        default_args = {
            'output_dir': output_dir,
            'num_train_epochs': num_train_epochs,
            'per_device_train_batch_size': per_device_train_batch_size,

            'fp16':True,
            'evaluation_strategy': 'epoch',
            'weight_decay': 0.01,
            'save_total_limit': 3,

            'report_to': ["wandb"],
            'run_name': wandb_run_name,
            'logging_dir': "./logs",
            'warmup_steps': warmup_steps
        }

        default_args.update(kwargs)
        return TrainingArguments(**default_args)

    def process_for_model(self,sub_dataset,tokenizer):
        def tokenize_function(examples):
            result = tokenizer(examples["news_title"], truncation=True)
            if tokenizer.is_fast:
                result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
            return result

        tokenized_datasets = sub_dataset.map(
            tokenize_function, batched=True, remove_columns=["news_title"]
        )
        return tokenized_datasets


    def test_train(self, num_train_epochs=3, per_device_train_batch_size=4, per_device_eval_batch_size=10):
        # wandb.init(project='test',group='DDP')
        # if int(os.environ.get("LOCAL_RANK", -1)) == 0:

        model = self.get_model()
        test_train_dataset = self.train_dataset.select(range(500))
        test_validation_dataset = self.validation_dataset.select(range(200))
        current_time = time.localtime(time.time())

        output_dir = "./models/output/test"
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        experiential_learning_rate = 0.003239 - 0.0001395 * math.log(model.num_parameters())
        print(f'learning_rate：{experiential_learning_rate}')
        experiential_batch_size = int(2 * (10 ** 8) / 0.001)
        train_args = self.get_train_args(train_dataset_length=test_train_dataset.num_rows,
                                         num_train_epochs=num_train_epochs,
                                         per_device_train_batch_size=per_device_train_batch_size,
                                         per_device_eval_batch_size=per_device_eval_batch_size,
                                         gpu_num=len(self.gpus.split(',')),
                                         wandb_run_name=f'test_train_{current_time.tm_mon}_{current_time.tm_mday}_{current_time.tm_hour}_{current_time.tm_min}',
                                         output_dir=output_dir
                                         )

        trainer = Trainer(
            model=model,
            args=train_args,
            train_dataset=test_train_dataset,
            eval_dataset=test_validation_dataset,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator
        )
        trainer.train()





