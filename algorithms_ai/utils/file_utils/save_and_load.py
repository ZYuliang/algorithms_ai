import _pickle as cPickle

import json
import jsonlines
import os

import pandas as pd
from loguru import logger

file_type_map_suffix = {
    'pickle': ['.pkl', '.pickle'],
    'json': ['.json'],
    'jsonl': ['.jsonl', '.json'],
}


def file_decrypt(file):
    # 文件解密
    if 'json' in file:
        dt = pd.read_json(file)
        dt.to_json(file)
    else:
        dt = pd.read_excel(file)

        dt.to_excel(file)


def is_valid_file(filepath: str) -> bool:
    if not os.path.exists(filepath):
        logger.error(f"path: {os.path.abspath(filepath)} is not exist")
        return False
    return True


def is_valid_suffix(filepath, file_type):
    if '.' + filepath.split('.')[-1] in file_type_map_suffix[file_type]:
        return True
    else:
        logger.error(f"path: {os.path.abspath(filepath)} suffix is not right")
        return False

# data.to_json(filepath, orient='records', force_ascii=False, indent=4)

def save_data(data, filepath, file_type='pickle', mode='w'):
    # mode: jsonl的写入格式 ‘w’ 或 ‘a’
    if not is_valid_suffix(filepath, file_type):
        return

    filepath = os.path.abspath(filepath)
    if not os.path.isdir(os.path.dirname(filepath)):
        logger.info(f'{os.path.dirname(filepath)} not exist! Make it!')
        os.makedirs(os.path.dirname(filepath))

    if file_type == 'json':
        with open(filepath, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=4)
    elif file_type == 'jsonl':
        with jsonlines.open(filepath, mode) as writer:
            writer.write_all(data)
    else: # pickle
        with open(filepath, 'wb') as wf:
            cPickle.dump(data, wf)
    logger.info(f"save data to file:{os.path.abspath(filepath)}")


def load_data(filepath, file_type='pickle'):
    if not is_valid_file(filepath) or not is_valid_suffix(filepath, file_type):
        return

    if file_type == 'json':  # 2.90613 s
        with open(filepath, 'r', encoding='utf-8') as file:
            data = json.load(file)
    elif file_type == 'jsonl':
        with open(filepath, "r", encoding='utf-8') as file:
            data = [json.loads(line) for line in file]
    else:
        with open(filepath, 'rb') as rf:
            data = cPickle.load(rf)
    logger.info(f"load data from file:{os.path.abspath(filepath)}")
    return data


def save_and_load(file_name, file_type='pickle', use_cache=True):
    def main_func(func):
        def inner_func(*args, **kwargs):
            if use_cache and os.path.isfile(file_name):
                dt = load_data(file_name, file_type)
            else:
                dt = func(*args, **kwargs)
                save_data(dt, file_name, file_type)
            return dt

        return inner_func

    return main_func


if __name__ == '__main__':
    a = [1, 23, 4, 5]
    save_data()

    # @save_and_load('a.pickle', file_type='pickle', use_cache=True)
    # def get_data():
    #     data = []
    #     return data
    #
    # data = get_data(use_cache=True)
