"""
只支持pickle(快速读取数据结构)，json（一般用于可视化一个数据或一组字符串列表数据），jsonl（存储records，一组列表数据）
保存数据到文件：save_data
读取文件到数据：load_data
删除一个文件夹 remove_dir

获取当前文件的绝对路径 os.path.realpath(__file__)
获取当前文件所在的目录的路径os.path.dirname(os.path.realpath(__file__))
获取当前文件所在目录的上一级目录的路径，os.path.dirname()可以多次使用，直到到达要读写的文件所在的上一级目录为止

TODO: 添加压缩加压处理
TODO: 流式处理大文件
"""

import _pickle as cPickle
import json
import os
import shutil

from loguru import logger

file_type_map_suffix = {
    'pickle': ['.pkl', '.pickle'],
    'json': ['.json', ],
    'jsonl': ['.jsonl', '.json'],
    'tsv': ['.tsv', ]
}
suffix_map_file_type = {
    '.pkl': 'pickle',
    '.pickle': 'pickle',
    '.json': 'json',
    '.jsonl': 'jsonl',
    '.tsv': '.tsv'
}


def remove_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
        logger.info(f'path: {path} removed!')


def save_data(data, filepath, file_type=None, mode='w'):
    """
    存储数据
    :param data: 一组列表数据
    :param filepath: 文件路径，文件路径要与文件类型匹配
    :param file_type: 文件类型
    :param mode: 添加模式还是覆盖模型，jsonl的写入格式 ‘w’ 或 ‘a’
    :return:
    """
    file_suffix = '.' + filepath.split('.')[-1]
    if not file_type:
        file_type = suffix_map_file_type.get(file_suffix)
    if file_suffix not in file_type_map_suffix.get(file_type, []):
        logger.info(f'{filepath} ’s suffix is not right!')

    # 没有路径创建路径
    filepath = os.path.abspath(filepath)
    if not os.path.isdir(os.path.dirname(filepath)):
        logger.info(f'{os.path.dirname(filepath)} not exist! Make it!')
        os.makedirs(os.path.dirname(filepath))

    if file_type == 'json':
        with open(filepath, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=4)
    elif file_type == 'jsonl':
        with open(filepath, mode, encoding='utf-8') as file:
            for i in data:
                json.dump(i, file, ensure_ascii=False)
                file.write('\n')
    elif file_type == 'tsv':
        with open(filepath, mode='w') as f_write:
            for t in data:
                f_write.write(t)
    else:  # pickle
        with open(filepath, 'wb') as wf:
            cPickle.dump(data, wf)
    logger.info(f"save data to file:{os.path.abspath(filepath)}")


def load_data(filepath, file_type=None):
    """
    从文件中导入数据,只支持jsonl的batch读取
    :param filepath: 文件路径，要与文件类型对应
    :param file_type: 文件类型
    :return:
    """
    file_suffix = '.' + filepath.split('.')[-1]
    if not file_type:
        file_type = suffix_map_file_type.get(file_suffix)

    if file_suffix not in file_type_map_suffix.get(file_type, []):
        logger.error(f"path: {os.path.abspath(filepath)} suffix is not right")

    if file_type == 'json':  # 2.90613 s
        with open(filepath, 'r', encoding='utf-8') as file:
            data = json.load(file)
    elif file_type == 'jsonl':
        with open(filepath, "r", encoding='utf-8') as file:
            data = [json.loads(line) for line in file]
    elif file_type == 'tsv':
        with open(filepath, mode='r') as f_read:
            data = f_read.readlines()
    else:
        with open(filepath, 'rb') as rf:
            data = cPickle.load(rf)
    logger.info(f"load data from file:{os.path.abspath(filepath)}")
    return data


def load_jsonl_batch(filepath, batch_size=3):
    with open(filepath, "r", encoding='utf-8') as file:
        data = []
        json_loads = json.loads
        for line in file:
            data.append(json_loads(line))
            if len(data) >= batch_size:
                yield data
                data = []
    if data:
        return data


#
# def save_and_load(file_name, file_type=None):
#     """
#     导入或读取所有的数据，装饰器，如果使用use_file_cache,则直接载入文件
#     :param file_name:
#     :param file_type:
#     :return:
#     """
#
#     def main_func(func):
#         def inner_func(*args, **kwargs):
#             use_file_cache = kwargs.get('use_file_cache', True)
#             if 'use_file_cache' in kwargs:
#                 kwargs.pop('use_file_cache')
#             if os.path.isfile(file_name) and use_file_cache:
#                 dt = load_data(file_name, file_type)
#             else:
#                 dt = func(*args, **kwargs)
#                 save_data(dt, file_name, file_type)
#             return dt
#
#         return inner_func
#
#     return main_func


def save_and_load():
    def main_func(func):
        def inner_func(*args, **kwargs):
            use_file_cache = kwargs.get('use_file_cache', True)
            results_file_name = kwargs.get('results_file_name', './tmp.json')
            results_file_type = kwargs.get('results_file_type', None)
            if 'use_file_cache' in kwargs:
                kwargs.pop('use_file_cache')
            if 'results_file_name' in kwargs:
                kwargs.pop('results_file_name')
            if 'results_file_type' in kwargs:
                kwargs.pop('results_file_type')
            if os.path.isfile(results_file_name) and use_file_cache:
                dt = load_data(results_file_name, results_file_type)
            else:
                dt = func(*args, **kwargs)
                save_data(dt, results_file_name, results_file_type)
            return dt

        return inner_func

    return main_func


@save_and_load()
def test_data():
    d = ['a', 'b', 'c']
    return d


if __name__ == '__main__':
    d = test_data()
    print(2)
