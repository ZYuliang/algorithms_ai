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
    from algorithms_ai.deep_learning.information_extraction.data_collator_ie import CollatorForIE
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
        from algorithms_ai.deep_learning.information_extraction.data_process_ie import process_one_sample

        from algorithms_ai.dataset_utils.dataset_utils import save_and_load_dataset_dict
        from datasets import DatasetDict, Dataset
        @save_and_load_dataset_dict(dataset_dir=dataset_dict_dir)
        def get_dataset_dict_func():
            train_dataset = [process_one_sample(i, tokenizer, entity2id) for i in train_data]
            if val_data:
                val_dataset = [process_one_sample(i, tokenizer, entity2id) for i in val_data]
                dt = DatasetDict({'train_for_model': Dataset.from_list(train_dataset),
                                  'val_for_model': Dataset.from_list(val_dataset)})
            else:
                dt = DatasetDict({'train_for_model': Dataset.from_list(train_dataset)})

            return dt

        return get_dataset_dict_func(use_file_cache=use_file_cache)

    @staticmethod
    def init_model(entity2id=None, relation2id=None, head_size=64, encoder_type='bert',
                   encoder_path='bert_base_uncased', model_head='', frozen_encoder=False, model_max_length=128):
        from algorithms_ai.deep_learning.information_extraction.modeling_ie import IEModelConfig, IEModel

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
        from algorithms_ai.deep_learning.information_extraction.metric_ie import metric_for_ie

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
        train_dataset = dataset_dict['train_for_model']
        validation_dataset = dataset_dict['val_for_model']

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

        from algorithms_ai.deep_learning.information_extraction.trainer_ie import IETrainer
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
        from algorithms_ai.deep_learning.information_extraction.modeling_ie import IEModel
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

        from algorithms_ai.deep_learning.information_extraction.data_process_ie import process_one_sample
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

        from algorithms_ai.deep_learning.information_extraction.trainer_ie import IETrainer
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

    def inference(self, to_predicts, cuda_devices, per_device_eval_batch_size, threshold=0):
        if cuda_devices:
            os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        if not hasattr(self, 'inferencer'):
            self.model, self.tokenizer = self.get_model(self.model_path)
            data_collator = get_data_collator(
                entity2id=self.model.config.entity2id,
                relation2id=self.model.config.relation2id,
                mode='predict',
                add_raw_info=['input_text', 'offset_mapping']
            )
            train_args = self.get_train_args(
                cuda_devices=cuda_devices, output_dir='./tmp', metric=None, train_dataset_length=100,
                num_train_epochs=1, per_device_train_batch_size=4,
                per_device_eval_batch_size=per_device_eval_batch_size,
                wandb_project_name=None, wandb_run_name=None, label_names=[],
                remove_unused_columns=False,
            )

            from algorithms_ai.deep_learning.information_extraction.trainer_ie import IETrainer
            self.inferencer = IETrainer(
                model=self.model,
                args=train_args,
                train_dataset=None,
                eval_dataset=None,
                tokenizer=self.tokenizer,
                data_collator=data_collator,
            )

        from algorithms_ai.deep_learning.information_extraction.data_process_ie import process_one_sample
        from datasets import Dataset
        test_data = [process_one_sample({'input_text': i}, self.tokenizer) for i in to_predicts]
        test_data = Dataset.from_list(test_data)

        res = self.inferencer.predict(test_data)

        from algorithms_ai.deep_learning.information_extraction.logits_process_ie import process_model_output
        all_preds = process_model_output(res.predictions, threshold=threshold)

        all_offset_mapping = [i['offset_mapping'] for i in test_data]
        all_texts = [i['input_text'] for i in test_data]

        refined_res = []
        assert len(all_preds) == len(all_offset_mapping)
        id2entity = {j: i for i, j in self.model.config.entity2id.items()}

        from algorithms_ai.dataset_utils.ner_process_utils import LabelNormalizer
        l_n = LabelNormalizer().run
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
                sub_res[id2entity[entity_id]] = refined_entity_res
            sub_res = l_n(input_text=input_text, ner_results=sub_res,
                          ner_keys=list(id2entity.values()), entity_repair_mode='complete')
            refined_res.append(sub_res)
        return refined_res

    def test(self, test_train=False, test_eval=False, test_predict=False):
        data = [
                   {'input_text': 'After 1 more month had episodes of extreme fatigue, increased belching, '
                                  'stomach discomfort and leg heaviness, low bp and pulse.',
                    'NER_ADR': [
                        [{'end_offset': 49, 'start_offset': 43, 'text': 'fatigue'}],
                        [{'end_offset': 69, 'start_offset': 52, 'text': 'increased belching'}],
                        [{'end_offset': 89, 'start_offset': 72, 'text': 'stomach discomfort'}],
                        [{'end_offset': 107, 'start_offset': 95, 'text': 'leg heaviness'}],
                        [{'end_offset': 115, 'start_offset': 110, 'text': 'low bp'}],
                        [{'end_offset': 112, 'start_offset': 110, 'text': 'low'},
                         {'end_offset': 125, 'start_offset': 121, 'text': 'pulse'}]]
                    }
               ] * 100

        if test_train:
            self.entity2id = {'NER_ADR': 0}
            self.encoder_path = "/large_files/pretrained_pytorch/pubmed_bert_base_cased/"

            self.train(train_data=data,
                       val_data=data,
                       dataset_dict_dir='./tmp/dataset',
                       use_file_cache=False,
                       model_max_length=108,

                       cuda_devices='2,3',
                       wandb_project_name=None,
                       wandb_run_name=None,
                       num_train_epochs=3,
                       per_device_train_batch_size=32,
                       per_device_eval_batch_size=16,
                       output_dir='./tmp/model',
                       learning_rate=1e-4,

                       metric='eval_sum_micro_score',
                       frozen_encoder=False,
                       warmup_steps=20
                       )

        if test_eval:
            from algorithms_ai.deep_learning.deep_learning_utils import get_train_date
            checkpoint = get_train_date('./tmp/model', 'best')
            self.model_path = checkpoint
            self.eval(data, cuda_devices='2,3',
                      metric='eval_sum_micro_score',
                      per_device_eval_batch_size=8,
                      wandb_project_name=None,
                      wandb_run_name=None)

        if test_predict:
            from algorithms_ai.deep_learning.deep_learning_utils import get_train_date
            checkpoint = get_train_date('./tmp/model', 'best')
            self.model_path = checkpoint
            from algorithms_ai.deep_learning.deep_learning_utils import get_train_date
            to_predicts = [i['input_text'] for i in data]
            res = self.inference(to_predicts, cuda_devices='2,3', per_device_eval_batch_size=64, threshold=2)
            res2 = self.inference(to_predicts[0:50], cuda_devices='2,3', per_device_eval_batch_size=64, threshold=2)
            for i in range(len(res)):
                print('*' * 30)
                print(res[i])
                print(data[i])
            from algorithms_ai.deep_learning.metric.ner_metric import entity_recognition_metric_for_data

            entity_recognition_metric_for_data(data, res, all_entity_type=['NER_ADR'], show_each_entity_score=False,
                                               return_score_key=None)


if __name__ == '__main__':
    InformationExtraction().test(test_train=False, test_eval=False, test_predict=True)
