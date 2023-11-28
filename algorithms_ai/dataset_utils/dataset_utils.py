import json
import os
import random
import shutil
from loguru import logger
from datasets import load_from_disk

def split_train_dev_test(samples: list, splits=(0.8, 0.1, 0.1), shuffle=True, random_seed=None):
    # 切分数据集,一堆样本的列表,
    # 注意：切分的使用不应该是输入模型的样本，而是实际使用时候是需要怎样切分的样本，样本的细粒度最好是最大
    # 比如原始是一篇一篇文章的结果，就以一篇篇文章的结果进行切分，此时评估的时候也以这样细粒度的样本进行评估
    if shuffle:
        if random_seed:
            random.seed(random_seed)
        random.shuffle(samples)
    if sum(splits) == 1:
        train_length = int(len(samples) * splits[0])
        dev_length = int(len(samples) * (splits[0] + splits[1]))
        return samples[0:train_length], samples[train_length:dev_length], samples[dev_length:]
    else:
        if splits[2] > 1:
            test_data = samples[0:splits[2]]
            samples = samples[splits[2]:]
        else:
            test_data = None

        if splits[1] > 1:
            dev_data = samples[0:splits[1]]
            samples = samples[splits[1]:]
        else:
            dev_data = None

        if splits[0] > 1:
            train_data = samples[0:splits[0]]
            samples = samples[splits[0]:]
        else:
            train_data = None
        if test_data and not dev_data and not train_data:
            if splits[0] + splits[1] == 1:
                train_length = int(len(samples) * splits[0])
                train_data, dev_data = samples[0:train_length], samples[train_length:]

        return train_data, dev_data, test_data


def add_dataset_to_dataset_dict(dataset, datasets_dict_dir, split_name, desc='', use_cache=False):
    # 把数据存入数据字典中,默认覆写模式

    if use_cache and os.path.exists(os.path.join(datasets_dict_dir, split_name)):
        return

    datasets_dict_json = os.path.join(datasets_dict_dir, 'dataset_dict.json')
    if not os.path.exists(datasets_dict_json):
        os.mkdir(datasets_dict_json)
        with open(datasets_dict_json, 'w', encoding='utf-8') as file:
            json.dump({'splits': [split_name]}, file, ensure_ascii=False, indent=4)

    if os.path.exists(os.path.join(datasets_dict_dir, split_name)):
        shutil.rmtree(os.path.join(datasets_dict_dir, split_name))
    dataset.info.update(dataset.info.from_dict({"description": desc}))
    dataset.save_to_disk(os.path.join(datasets_dict_dir, split_name))

    with open(datasets_dict_json, 'r', encoding='utf-8') as file:
        dataset_dict_config = json.load(file)
    if split_name not in dataset_dict_config.get('splits', []):
        if 'splits' not in dataset_dict_config:
            dataset_dict_config['splits'] = []
        dataset_dict_config['splits'].append(split_name)
    with open(datasets_dict_json, 'w', encoding='utf-8') as file:
        json.dump(dataset_dict_config, file, ensure_ascii=False, indent=4)



def save_and_load_dataset(dataset_dir):
    from datasets import load_from_disk
    def main_func(func):
        def inner_func(*args, **kwargs):
            use_file_cache = kwargs.get('use_file_cache', True)
            if 'use_file_cache' in kwargs:
                kwargs.pop('use_file_cache')
            if os.path.exists(dataset_dir) and use_file_cache:
                logger.info(f'load dataset dict dir : {dataset_dir}')
                dt = load_from_disk(dataset_dir, keep_in_memory=True)
            else:
                dt = func(*args, **kwargs)
                dt.save_to_disk(dataset_dir, num_proc=4)
                logger.info(f'save dataset dict dir : {dataset_dir}')
            return dt

        return inner_func


def load_train_validation(datasets_dict_dir, use_cache=False):
    """
    读取数据集字典中train和validation的装饰器
    """

    def main_func(func):
        def inner_func(*args, **kwargs):
            if os.path.exists(datasets_dict_dir):
                dsd = load_from_disk(datasets_dict_dir, keep_in_memory=True)
            else:
                dsd = dict()
            if use_cache and ('train' in dsd) and ('validation' in dsd):
                train = dsd['train']
                validation = dsd['validation']
            else:
                train, validation = func(*args, **kwargs)
                add_dataset_to_dataset_dict(train, datasets_dict_dir, 'train', desc='train', use_cache=use_cache)
                add_dataset_to_dataset_dict(validation, datasets_dict_dir, 'validation', desc='validation',
                                            use_cache=use_cache)
            return train, validation

        return inner_func

    return main_func


def get_data_structure(data):
    if isinstance(data,str):
        return 'str\n'
    elif isinstance(data,list):
        if len(data)>=1:
            return f'list({len(data)}:\n'+get_data_structure(data[0])+')\n'
        else:
            return f'list{0}:\n'
    elif isinstance(data,dict):
        if not data:
            return 'dict\n'

        r = 'dict(\n'
        for i,j in data.items():
            r += f'{i}:'
            r += get_data_structure(j)
        r+=')\n'
        return r
    elif isinstance(data,bool):
        return 'bool\n'
    else:
        if not data:
            return 'None\n'
        return 'other\n'

def show_data_structure(data):
    data_structure = get_data_structure(data)
    data_structure = data_structure.split('\n')
    index=0
    step = 4
    for i in data_structure:
        if '(' in i:
            print(index * ' ' + i)
            index += step
        elif ')' in i:
            index = index - step
            print(index * ' ' + i)
        else:
            print(index * ' ' + i)



if __name__ == '__main__':
    a = list(range(30))
    print(split_train_dev_test(a))
