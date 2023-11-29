import math
def caculate_eval_steps_warmup_steps(train_dataset_length, num_train_epochs=2, per_device_train_batch_size=4,
                                     warmup_ratio=0.1, cuda_devices_num=1, gradient_accumulation_steps=1, eval_num=20):
    # 计算一共需要eval_num次计算指标时，每次的eval_steps数,以及warmup-steps数
    # 近似最大更新步数
    sim_max_update_steps = math.ceil(num_train_epochs * max(
        (train_dataset_length // (per_device_train_batch_size * cuda_devices_num) // gradient_accumulation_steps), 1))

    eval_steps = int(max(sim_max_update_steps // eval_num, 1))  # 每次训练都评估eval_num次
    warmup_steps = int(max(sim_max_update_steps * warmup_ratio, 1))
    return eval_steps, warmup_steps


# # 设置transformers模块的日志等级，减少不必要的警告，对训练过程无影响，请忽略
# from transformers import TrainingArguments, Trainer, logging
# logging.set_verbosity_error()




# GPU 利用率
def print_gpu_utilization():
    """
    nvidia-ml-py3库允许我们从 Python 中监控模型的内存使用情况
    pip install transformers datasets accelerate nvidia-ml-py3
    """
    from pynvml import *
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")



def tokenize_and_align_labels(text, labels, tokenizer, label2id, sub_word_prefix='##'):
    """
    注意输入的要是原始文本，为了后面能够映射回原始文本的坐标
    label是原始文本熵的标签，注意：这个标签最好处理成对的标签
    tokenizer必须是模型的tokenizer，能直接调用，且能返回坐标映射
    label2id是实体的映射
    sub_word_prefix是使用word-piece时的前缀，不同模型可能不同
    :param text:
    :param labels:
    :param tokenizer:
    :param label2id:
    :param sub_word_prefix:
    :return: 默认truncation到tokenizer的model_max_length,返回token好的结果，input_ids,attention,type,labels,
    """
    labels = sorted(labels, key=lambda x: x['start_offset'])
    labels.append({"start_offset": len(text) + 3, "end_offset": len(text) + 3, "text": "", "label": "other"}) # 为了走完token
    token_results = tokenizer(text, truncation=True, return_offsets_mapping=True)
    tokens = tokenizer.convert_ids_to_tokens(token_results['input_ids'])
    special_tokens = tokenizer.special_tokens_map.values()

    ner_tags = []
    label_index = 0
    tmp_label = labels[label_index]

    for token_span,token in zip(token_results['offset_mapping'],tokens):
        if token_span[0]==token_span[1]:
            ner_tags.append('O')
            continue

        if tmp_label['start_offset'] > (token_span[1]-1):
            ner_tags.append('O')
        else:  # tmp_label['start_offset']<=token['end_offset']
            if not ner_tags or ner_tags[-1] == 'O':
                ner_tags.append('B-' + tmp_label['label'])
            else:
                if ner_tags[-1][2:] == tmp_label['label']:
                    ner_tags.append('I-' + tmp_label['label'])
                else:
                    ner_tags.append('B-' + tmp_label['label'])

            if tmp_label['end_offset'] <= (token_span[1]-1):
                label_index += 1
                tmp_label = labels[label_index]
    assert len(ner_tags)==len(tokens)
    labels_id = []
    for i,j in zip(tokens,ner_tags):
        if i in special_tokens:
            labels_id.append(-100)
        elif i.startswith(sub_word_prefix):
            labels_id.append(-100)
        else:
            labels_id.append(label2id[j])

    token_results['labels'] = labels_id
    return token_results

def process_func(example, tokenizer, label2id):
    token_results = tokenize_and_align_labels(example['input_text'], example['arms_ner'], tokenizer, label2id, sub_word_prefix='##')
    return token_results
#
#
# inspect.getmodule(model) # 获得模型信息
#
#
# import torch
# import torch.nn.functional as F
# import torch.nn as nn
# x = torch.randn(5, 5)
# target = torch.tensor([0, 2, 3, 1, 4]) # 标签 这里还有一个torch.tensor与torch.Tensor的知识点https://blog.csdn.net/weixin_40607008/article/details/107348254
# one_hot = F.one_hot(target).float() # 对标签进行one_hot编码
# softmax = torch.exp(x)/torch.sum(torch.exp(x), dim = 1).reshape(-1, 1)
# logsoftmax = torch.log(softmax)
# nllloss = -torch.sum(one_hot*logsoftmax)/target.shape[0]
# print(nllloss)
# ogsoftmax = F.log_softmax(x, dim = 1)
# nllloss2 = F.nll_loss(logsoftmax, target) # 无需对标签做one_hot编码
# print(nllloss2)
#
# ###最后我们直接用torch.nn.CrossEntropyLoss验证一下以上两种方法的正确性
# cross_entropy = F.cross_entropy(x, target)
# print(cross_entropy)
# # CrossEntropyLoss要除以batch，表示每个样本的平均loss
# torch.cuda.empty_cache()  # 释放一些存储
#
# torch.device("cuda：1，2") if torch.cuda.is_available() else torch.device("cpu")