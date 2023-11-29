"""
首先获取预处理（去重，删错）后的数据字典，还包括原始训练和评估的数据，raw_train,raw_validation--是原始格式
然后将训练和评估进行进一步处理成模型输入的形式--train,validation
"""

from datasets import load_dataset, load_from_disk, DatasetDict, Dataset,ClassLabel
from algorithms_ai.my_models.dataset_utils import load_datasets_dict,load_train_validation
data_cache = './cache'
datasets_dict_dir = './single_cls_data'

@load_datasets_dict(datasets_dict_dir=datasets_dict_dir, use_cache=True)
def get_single_cls_datasets_dict():
    d = load_dataset('glue', 'sst2', split='train', cache_dir=data_cache, streaming=True)
    all_data = []
    for i in d:
        all_data.append(i)
        if len(all_data) > 3000:
            break
    d = Dataset.from_list(all_data)
    train_dataset, validation_dataset = d.train_test_split(test_size=0.1).values()
    dsd = DatasetDict({'raw_train': train_dataset, 'raw_validation': validation_dataset})
    return dsd


@load_train_validation(datasets_dict_dir=datasets_dict_dir, use_cache=True)
def get_train_validation(tokenizer):
    def process_sub_dataset_for_model(sub_dataset, tokenizer):
        def preprocess_for_model(examples):
            return tokenizer(examples["sentence"], truncation=True)
        sub_dataset = sub_dataset.map(preprocess_for_model, batched=True, keep_in_memory=True, batch_size=2000)
        sub_dataset = sub_dataset.rename_columns({"label": "labels"})  # 重命名
        return sub_dataset

    dsd = get_single_cls_datasets_dict()

    train_dataset = process_sub_dataset_for_model(dsd['raw_train'], tokenizer)
    validation_dataset = process_sub_dataset_for_model(dsd['raw_validation'], tokenizer)
    return train_dataset,validation_dataset
