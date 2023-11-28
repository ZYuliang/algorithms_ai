import os
import time
import warnings

os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'  # fast tokenizer warning
warnings.filterwarnings("ignore")  # torch warning


def get_tokenizer(encoder_type='bert',
                  encoder_checkpoint="/home/zyl/disk/algorithms_ai/algorithms_ai/my_models/relation_extraction/GPLinker_torch-main/model/",
                  model_max_length=128):
    tokenizer_cache = '/large_files/5T/huggingface_cache/tokenizer_cache/'
    if encoder_type == 'bert':
        from transformers import BertTokenizerFast
        tokenizer = BertTokenizerFast.from_pretrained(pretrained_model_name_or_path=encoder_checkpoint,
                                                      cache_dir=tokenizer_cache)
    else:
        from transformers import BertTokenizerFast
        tokenizer = BertTokenizerFast.from_pretrained(pretrained_model_name_or_path=encoder_checkpoint,
                                                      cache_dir=tokenizer_cache)

    tokenizer.model_max_length = model_max_length
    return tokenizer


def get_data_collator(entity2id=None, relation2id=None, mode='train', add_raw_info=None):
    from information_extraction.data_collator_ie import CollatorForIE
    return CollatorForIE(entity2id=entity2id, relation2id=relation2id, mode=mode, add_raw_info=add_raw_info)


class InformationExtraction:
    def __init__(self,
                 entity2id=None,
                 relation2id=None,
                 head_size=64,
                 encoder_type='bert',
                 encoder_path='bert-base-uncased',
                 model_head='EfficientGlobalPointerHead',
                 cache_dir='./cache',
                 model_path=None,
                 ):
        self.entity2id = entity2id
        self.relation2id = relation2id
        self.head_size = head_size
        self.model_head = model_head
        self.encoder_type = encoder_type
        self.encoder_path = encoder_path
        self.cache_dir = cache_dir
        self.model_path = model_path

    @staticmethod
    def get_dataset_dict(train_data=None, tokenizer=None, entity2id=None, val_data=None,
                         dataset_dict_dir='./test_data', use_file_cache=True):
        from information_extraction.data_process_ie import  process_one_sample
        from information_extraction.utils_ie import save_and_load_dataset
        from datasets import DatasetDict, Dataset
        @save_and_load_dataset(dataset_dir=dataset_dict_dir)
        def get_dataset_dict():
            train_dataset = [process_one_sample(i, tokenizer, entity2id) for i in train_data]
            if val_data:
                val_dataset = [process_one_sample(i, tokenizer, entity2id) for i in val_data]
                dt = DatasetDict({'train': Dataset.from_list(train_dataset),
                                  'val': Dataset.from_list(val_dataset)})
            else:
                dt = DatasetDict({'train': Dataset.from_list(train_dataset)})

            return dt

        return get_dataset_dict(use_file_cache=use_file_cache)

    @staticmethod
    def init_model(entity2id=None, relation2id=None, head_size=64, encoder_type='bert',
                   encoder_path='bert_base_uncased', model_head='', frozen_encoder=False, model_max_length=128):
        from information_extraction.modeling_ie import IEModelConfig, IEModel

        model_config = IEModelConfig(
            head_size=head_size,
            entity2id=entity2id,
            relation2id=relation2id,
            model_head=model_head,
            encoder_type=encoder_type,
            encoder_path=encoder_path,
            model_max_length=model_max_length
        )

        model = IEModel(model_config)

        def frozen_layers(model):
            for name, param in model.named_parameters():
                if name.startswith('bert.embeddings'):
                    param.requires_grad = False
                if name.startswith('bert.encoder'):
                    param.requires_grad = False

        if frozen_encoder:
            frozen_layers(model)
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
    def get_compute_metrics(entity2id, relation2id, threshold=0):
        from functools import partial
        from information_extraction.metric_ie import metric_for_ie

        return partial(
            metric_for_ie,
            entity2id=entity2id,
            relation2id=relation2id,
            threshold=threshold
        )

    def train(self,
              train_data,
              val_data,
              dataset_dict_dir='./test_data',
              use_file_cache=True,
              model_max_length=128,

              cuda_devices='0,1',
              wandb_project_name=None,
              wandb_run_name=None,
              num_train_epochs=3,
              per_device_train_batch_size=8,
              per_device_eval_batch_size=8,
              output_dir='./models/test',

              metric=None,
              frozen_encoder=False,
              **train_args
              ):
        if cuda_devices:
            os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices

        tokenizer = get_tokenizer(
            encoder_type=self.encoder_type,
            encoder_checkpoint=self.encoder_path,
            model_max_length=model_max_length)

        data_collator = get_data_collator(
            entity2id=self.entity2id,
            relation2id=self.relation2id)

        model = self.init_model(
            entity2id=self.entity2id, relation2id=self.relation2id,
            head_size=self.head_size, encoder_type=self.encoder_type, encoder_path=self.encoder_path,
            model_head=self.model_head, frozen_encoder=frozen_encoder, model_max_length=model_max_length)

        dataset_dict = self.get_dataset_dict(
            train_data=train_data, tokenizer=tokenizer, entity2id=self.entity2id,
            val_data=val_data, dataset_dict_dir=dataset_dict_dir, use_file_cache=use_file_cache,
        )
        train_dataset = dataset_dict['train']
        validation_dataset = dataset_dict['val']

        train_args = self.get_train_args(
            cuda_devices=cuda_devices, output_dir=output_dir, metric=metric, train_dataset_length=100,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            wandb_project_name=wandb_project_name, wandb_run_name=wandb_run_name,
            include_inputs_for_metrics=False,
            remove_unused_columns=False,
            **train_args
        )

        if metric:
            compute_metrics = self.get_compute_metrics(self.entity2id, self.relation2id)
        else:
            compute_metrics = None

        from information_extraction.trainer_ie import IETrainer
        trainer = IETrainer(
            model=model,
            args=train_args,
            train_dataset=train_dataset,
            eval_dataset=validation_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )
        trainer.train()

    @staticmethod
    def get_model(model_path):
        from information_extraction.modeling_ie import IEModel
        model = IEModel.from_pretrained(pretrained_model_name_or_path=model_path)

        tokenizer = get_tokenizer(
            encoder_type=model.config.encoder_type,
            encoder_checkpoint=model.config.encoder_path,
            model_max_length=model.config.model_max_length)

        return model, tokenizer

    def eval(self, test_data, cuda_devices,
             metric='eval_sum_micro_score', per_device_eval_batch_size=8, wandb_project_name=None, wandb_run_name=None):
        if cuda_devices:
            os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        model, tokenizer = self.get_model(self.model_path)

        data_collator = get_data_collator(
            entity2id=model.config.entity2id,
            relation2id=model.config.relation2id,
            mode='eval'
        )

        from information_extraction.data_process_ie import process_one_sample
        from datasets import Dataset

        test_data = [process_one_sample(i, tokenizer, model.config.entity2id) for i in test_data]
        test_data = Dataset.from_list(test_data)

        train_args = self.get_train_args(
            cuda_devices=cuda_devices, output_dir='./tmp', metric=metric, train_dataset_length=100,
            num_train_epochs=1, per_device_train_batch_size=4, per_device_eval_batch_size=per_device_eval_batch_size,
            wandb_project_name=wandb_project_name, wandb_run_name=wandb_run_name,
            include_inputs_for_metrics=False,
            remove_unused_columns=False,
        )

        if metric:
            compute_metrics = self.get_compute_metrics(model.config.entity2id, model.config.relation2id)
        else:
            compute_metrics = None

        from information_extraction.trainer_ie import IETrainer
        trainer = IETrainer(
            model=model,
            args=train_args,
            train_dataset=None,
            eval_dataset=test_data,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )
        trainer.evaluate(test_data)

    def inference(self, to_predicts, cuda_devices, per_device_eval_batch_size,threshold=0):
        if cuda_devices:
            os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        from information_extraction.data_process_ie import process_one_sample, sequence_padding
        from datasets import Dataset

        model, tokenizer = self.get_model(self.model_path)

        data_collator = get_data_collator(
            entity2id=model.config.entity2id,
            relation2id=model.config.relation2id,
            mode='predict',
            add_raw_info=['input_text', 'offset_mapping']
        )

        test_data = [process_one_sample({'input_text': i}, tokenizer) for i in to_predicts]
        test_data = Dataset.from_list(test_data)

        train_args = self.get_train_args(
            cuda_devices=cuda_devices, output_dir='./tmp', metric=None, train_dataset_length=100,
            num_train_epochs=1, per_device_train_batch_size=4, per_device_eval_batch_size=per_device_eval_batch_size,
            wandb_project_name=None, wandb_run_name=None, label_names=[]
        )

        from information_extraction.trainer_ie import IETrainer
        trainer = IETrainer(
            model=model,
            args=train_args,
            train_dataset=None,
            eval_dataset=None,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )
        res = trainer.predict(test_data)

        from information_extraction.logits_process_ie import process_model_output
        from information_extraction.utils_ie import postprocess_entities
        all_preds = process_model_output(res.predictions, threshold=threshold)

        all_offset_mapping = [i['offset_mapping'] for i in test_data]
        all_texts = [i['input_text'] for i in test_data]

        refined_res = []
        assert len(all_preds) == len(all_offset_mapping)
        id2entity = {j: i for i, j in model.config.entity2id.items()}

        for i in range(len(all_preds)):
            sub_res = dict()
            offset_mapping = all_offset_mapping[i]
            input_text = all_texts[i]
            sub_res['input_text'] = input_text

            for entity_id, entity_res in all_preds[i].items():
                refined_entity_res = [[{'start_offset': offset_mapping[each_entity_part[0]][0],
                                        'end_offset': offset_mapping[each_entity_part[-1]][1] - 1,
                                        'text': input_text[offset_mapping[each_entity_part[0]][0]:
                                                           offset_mapping[each_entity_part[-1]][1]]}
                                       for each_entity_part in each_entity]
                                      for each_entity in entity_res]
                refined_entity_res = postprocess_entities(refined_entity_res, input_text)

                sub_res[id2entity[entity_id]] = refined_entity_res
            refined_res.append(sub_res)
        return refined_res


if __name__ == '__main__':
    pass
    # m = InformationExtraction(model_path='/home/zyl/disk/PharmAI/Pipelines/components/discontinous_ner/models/test/checkpoint-150')
    # m.eval(test_data,cuda_devices,
    #          metric = 'eval_sum_micro_score',per_device_eval_batch_size=8,wandb_project_name=None,wandb_run_name=None)
    # entity2id = {'ADR': 0}
    #
    # trainer = InformationExtractionTrainer(
    #     entity2id=entity2id,
    #     relation2id=None,
    #     head_size=64,
    #     encoder_type='bert',
    #     encoder_path="/large_files/pretrained_pytorch/pubmed_bert_base_cased/",
    #     model_head='GlobalPointerHead',
    #     cache_dir='./cache',
    #     metric='eval_entity_f1',
    # )
    # from data_process import get_dataset_dict
    #
    # dataset_dic = get_dataset_dict(
    #     tokenizer=trainer.get_tokenizer(model_max_length=108),
    #     entity2id=entity2id,
    #     use_file_cache=True
    # )
    # train_data = dataset_dic['train']
    # val_data = dataset_dic['val']
    #
    # trainer.train(train_data,
    #               val_data,
    #               cuda_devices='1,6,7,8',
    #               wandb_project_name='ie',
    #               wandb_run_name='test',
    #               num_train_epochs=30,
    #               per_device_train_batch_size=232,
    #               per_device_eval_batch_size=128,
    #               output_dir='./models/test',
    #               learning_rate=8e-4,
    #               ignore_warning=True,
    #               frozen_encoder=False,
    #               )
