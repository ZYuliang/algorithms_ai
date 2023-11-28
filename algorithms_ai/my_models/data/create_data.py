"""
https://blog.csdn.net/yaohaishen/article/details/123702163

glue/super glue
二分类任务使用：SST-2
多分类任务使用：MNLI
实体识别：conll2003

"""
from datasets import load_dataset,load_from_disk,DatasetDict,Dataset
import os
data_cache = './cache'
def get_single_cls_dataset():
    cls_data_dir = './single_cls_data'
    if os.path.exists(cls_data_dir):
        dsd = load_from_disk(cls_data_dir,keep_in_memory=True)
        train_dataset = dsd['train']
        validation_dataset = dsd['validation']
    else:
        d = load_dataset('glue', 'sst2', split='train',cache_dir=data_cache,streaming=True)
        all_data = []
        for i in d:
            all_data.append(i)
            if len(all_data)>3000:
                break
        d = Dataset.from_list(all_data)
        train_dataset, validation_dataset = d.train_test_split(test_size=0.1).values()
        d = DatasetDict({'train': train_dataset, 'validation': validation_dataset})
        d.save_to_disk(cls_data_dir)

    return train_dataset,validation_dataset


def get_multi_cls_dataset():
    cls_data_dir = './multi_cls_data'
    if os.path.exists(cls_data_dir):
        dsd = load_from_disk(cls_data_dir,keep_in_memory=True)
        train_dataset = dsd['train']
        validation_dataset = dsd['validation']
    else:
        d = load_dataset('glue', 'mnli', split='train', cache_dir=data_cache, streaming=True)
        all_data = []
        for i in d:
            all_data.append(i)
            if len(all_data) > 3000:
                break
        d = Dataset.from_list(all_data)
        train_dataset, validation_dataset = d.train_test_split(test_size=0.1).values()
        d = DatasetDict({'train': train_dataset, 'validation': validation_dataset})
        d.save_to_disk(cls_data_dir)

    return train_dataset,validation_dataset


if __name__ == '__main__':
    print(1)
    # t, v = get_single_cls_dataset()
    t, v = get_multi_cls_dataset()
    print(1)



