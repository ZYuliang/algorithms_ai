import numpy as np
import os
from datasets import load_from_disk
from loguru import logger


def sequence_padding(inputs, length=None, value=0, seq_dims=1, mode='post'):
    """
    Numpy函数，将序列padding到同一长度
    """
    if length is None:
        length = np.max([np.shape(x)[:seq_dims] for x in inputs], axis=0)  # 最大实体数
    elif not hasattr(length, '__getitem__'):
        length = [length]

    slices = [np.s_[:length[i]] for i in range(seq_dims)]  # 索引
    slices = tuple(slices) if len(slices) > 1 else slices[0]
    pad_width = [(0, 0) for _ in np.shape(inputs[0])]  # padding的宽度，每条数据的类型数

    outputs = []
    for x in inputs:
        x = x[slices]
        for i in range(seq_dims):
            if mode == 'post':
                pad_width[i] = (0, length[i] - np.shape(x)[i])
            elif mode == 'pre':
                pad_width[i] = (length[i] - np.shape(x)[i], 0)
            else:
                raise ValueError('"mode" argument must be "post" or "pre".')
        x = np.pad(x, pad_width, 'constant', constant_values=value)
        outputs.append(x)
    return np.array(outputs)


def save_and_load_dataset(dataset_dir):
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

    return main_func


CHINESE_PUNCTUATION = '＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､\u3000、〃〈〉《》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏﹑﹔·！？｡。'
ENGLISH_PUNCTUATION = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'


def get_string_type():
    def get_string_type(c):
        if '\u0030' <= c <= '\u0039':
            return 'num'
        elif c in CHINESE_PUNCTUATION:
            return 'punctuation_zh'
        elif c in ENGLISH_PUNCTUATION:
            return 'punctuation_en'
        elif '\u0041' <= c <= '\u005a':
            return 'char_english_uppercase'
        elif '\u0061' <= c <= '\u007a':
            return 'char_english_lowercase'
        elif '\u0391' <= c <= '\u03a9':
            return 'greek_uppercase'
        elif '\u03b1' <= c <= '\u03c9':
            return 'greek_lowercase'
        elif '\u4e00' <= c <= '\u9fff':
            return 'chinese'
        return 'other'


def judge_segmentation(str1: str, str2: str, str1_left: str = None, str2_right: str = None):
    # 判断两个相邻的字符串，str1-str2的中间是否应该断开
    if str1 in ' ，；﹔！？｡。\t!,:;?' or str2 in ' \t':
        return True

    if '\u0030' <= str1 <= '\u0039' and '\u0030' <= str2 <= '\u0039':
        # 左右都是数字
        return False
    if ('\u0041' <= str1 <= '\u005a' or '\u0061' <= str1 <= '\u007a') and (
            '\u0041' <= str2 <= '\u005a' or '\u0061' <= str2 <= '\u007a'):
        # 左右都是英文字符
        return False

    if '\u4e00' <= str1 <= '\u9fff' and '\u4e00' <= str2 <= '\u9fff':
        # 左右都是中文
        return True

    if str1 in '.·' and '\u0030' <= str2 <= '\u0039':
        # 1  .2
        if str1_left and '\u0030' <= str1_left <= '\u0039':
            return False
        else:
            return True

    if '\u0030' <= str1 <= '\u0039' and str2 in '.·':
        # 1.  2
        if str2_right and '\u0030' <= str2_right <= '\u0039':
            return False
        else:
            return True

    return True


def completion_entity(input_text, start_offset, end_offset):
    """补全片段的实体
    e = [{'start_offset': 2, 'end_offset': 2, 'text': 'e'}, {'start_offset': 4, 'end_offset': 9, 'text': 'statin'}]

    input_text = 'the statins'

    print(completion_entity(input_text,e[0]['start_offset'],e[0]['end_offset']))
    """

    while start_offset > 1:
        if not judge_segmentation(str1=input_text[start_offset - 1],
                                  str2=input_text[start_offset],
                                  str1_left=input_text[start_offset - 2]):
            start_offset = start_offset - 1
        else:
            break
    while start_offset > 0:
        if not judge_segmentation(str1=input_text[start_offset - 1],
                                  str2=input_text[start_offset]):
            start_offset = start_offset - 1
        else:
            break

    while end_offset < len(input_text) - 2:
        if not judge_segmentation(str1=input_text[end_offset],
                                  str2=input_text[end_offset + 1],
                                  str2_right=input_text[end_offset + 2]):
            end_offset = end_offset + 1
        else:
            break

    while end_offset < len(input_text) - 1:
        if not judge_segmentation(str1=input_text[end_offset],
                                  str2=input_text[end_offset + 1]):
            end_offset = end_offset + 1
        else:
            break

    return start_offset, end_offset


def combine_entity_part(entity, input_text):
    """不同实体片段的合并，处理嵌套片段
    e = [{'start_offset': 2, 'end_offset': 2, 'text': 'e'}, {'start_offset': 4, 'end_offset': 9, 'text': 'statin'}]

    input_text = 'the statins'

    print(combine_entities(e, input_text))

    """
    if not entity:
        return []

    if len(entity) == 1:
        return entity

    entities = sorted(entity, key=lambda x: (x['start_offset'], x['end_offset']))
    refined_entity = []
    last_entity_part = entities[0]

    for current_entity_part in entities[1:]:
        if current_entity_part['start_offset'] <= last_entity_part['end_offset'] or \
                input_text[last_entity_part['end_offset'] + 1:current_entity_part['start_offset']] in [' ', '-', '-',
                                                                                                       '_']:
            last_entity_part = {
                'start_offset': last_entity_part['start_offset'],
                'end_offset': (tmp_max := max(current_entity_part['end_offset'], last_entity_part['end_offset'])),
                'text': input_text[last_entity_part['start_offset']:tmp_max + 1],
            }
        elif current_entity_part['start_offset'] > last_entity_part['end_offset']:
            refined_entity.append(last_entity_part)
            last_entity_part = current_entity_part
    refined_entity.append(last_entity_part)
    return refined_entity


def postprocess_entity(entity, input_text):
    # 处理单个实体：补全实体片段，合并实体片段
    refined_entity = []
    for entity_part_index, entity_part in enumerate(entity):
        start_offset = entity_part['start_offset']
        end_offset = entity_part['end_offset']
        start_offset, end_offset = completion_entity(input_text, start_offset, end_offset)
        if start_offset <= end_offset:
            if input_text[start_offset:end_offset + 1] in ['and', '', ' ', 'or'] and \
                    (entity_part_index == len(entity) - 1 or entity_part_index == 0):
                pass
            else:
                refined_entity.append({
                    'start_offset': start_offset,
                    'end_offset': end_offset,
                    'text': input_text[start_offset:end_offset + 1]
                })
    refined_entity = combine_entity_part(refined_entity, input_text)
    return refined_entity


def judge_entity_included(entity_1, entity_2):
    """判断两个实体是否互相包含：互不包含，1包含2，2包含1，互相包含
    e = [{'start_offset': 70, 'end_offset': 77, 'text': 'Terrible'},
         {'start_offset': 96, 'end_offset': 99, 'text': 'pain'}]
    b = [{'start_offset': 70, 'end_offset': 84, 'text': 'Terrible muscle'},
         {'start_offset': 90, 'end_offset': 99, 'text': 'joint pain'}]
    e2 = [{'start_offset': 70, 'end_offset': 77, 'text': 'Terrible'},
          {'start_offset': 96, 'end_offset': 99, 'text': 'pain'}]
    print(judge_entity_included(b, e2))
    """
    entity_1_include_2_matrix = np.zeros((len(entity_1), len(entity_2)))  # 1包含2的计数
    entity_2_include_1_matrix = np.zeros((len(entity_2), len(entity_1)))  # 2包含1的计数
    for entity_1_part_index, entity_1_part in enumerate(entity_1):
        for entity_2_part_index, entity_2_part in enumerate(entity_2):
            if entity_2_part['start_offset'] <= entity_1_part['start_offset'] <= entity_1_part['end_offset'] <= \
                    entity_2_part['end_offset']:
                entity_2_include_1_matrix[entity_2_part_index, entity_1_part_index] = 1
            if entity_1_part['start_offset'] <= entity_2_part['start_offset'] <= entity_2_part['end_offset'] <= \
                    entity_1_part['end_offset']:
                entity_1_include_2_matrix[entity_1_part_index, entity_2_part_index] = 1

    if sum(np.max(entity_2_include_1_matrix, axis=0)) == len(entity_1):
        # 2包含1
        entity_2_include_1 = True
    else:
        entity_2_include_1 = False
    if sum(np.max(entity_1_include_2_matrix, axis=0)) == len(entity_2):
        # 1包含2
        entity_1_include_2 = True
    else:
        entity_1_include_2 = False
    if entity_1_include_2 and entity_2_include_1:
        return 'include_each_other'
    elif entity_1_include_2 and not entity_2_include_1:
        return 'entity_1_include_2'
    elif entity_2_include_1 and not entity_1_include_2:
        return 'entity_2_include_1'
    else:
        return 'not_include'


def postprocess_entities(entities, input_text):
    """
    # 处理得到的实体 : 首先补足（由decoder决定，补足更优）每个实体片段，然后对每个实体进行处理，去除不需要的前后缀（and之类的），
                    有的合并就合并不同的part，然后对不同实体进行覆盖嵌套的去除
    # 在输出模型结果后，若干个实体的合并，把相邻的（a，b合并成ab），重复交叉的（a,b|b,c合并成abc），覆盖的（abc|ab变成abc）
    e = [[{'start_offset': 70, 'end_offset': 84, 'text': 'Terrible muscle'}],
         [{'start_offset': 103, 'end_offset': 138, 'text': 'Burning sensations in neck shoulders'}],
         [{'start_offset': 74, 'end_offset': 77, 'text': 'ible'},
          {'start_offset': 86, 'end_offset': 99, 'text': 'and joint pain'}],
         [{'start_offset': 74, 'end_offset': 84, 'text': 'ible muscle'},
          {'start_offset': 90, 'end_offset': 99, 'text': 'joint pain'}],
         [{'start_offset': 103, 'end_offset': 123, 'text': 'Burning sensations in'},
          {'start_offset': 144, 'end_offset': 154, 'text': 'upper chest'}],
         [{'start_offset': 140, 'end_offset': 142, 'text': 'and'},
          {'start_offset': 150, 'end_offset': 154, 'text': 'chest'}],
         [{'start_offset': 70, 'end_offset': 77, 'text': 'Terrible'},
          {'start_offset': 86, 'end_offset': 88, 'text': 'and'}],
         ]
    input_text = 'I was in deniel that the statins would cause ME side effects - I have Terrible muscle and joint pain , Burning sensations in neck shoulders and upper chest .'

    print(postprocess_entities(e, input_text))

    """
    entities = [postprocess_entity(each_entity, input_text) for each_entity in entities]
    entity_delete = [0 for _ in range(len(entities))]
    for each_entity_index in range(len(entities) - 1):
        for each_entity_index2 in range(each_entity_index + 1, len(entities)):
            include_info = judge_entity_included(entities[each_entity_index], entities[each_entity_index2])
            if include_info == 'entity_1_include_2':
                entity_delete[each_entity_index2] = 1
            elif include_info == 'entity_2_include_1':
                entity_delete[each_entity_index] = 1
            elif include_info == 'include_each_other':
                entity_delete[each_entity_index2] = 1
    entities = [i for i, j in zip(entities, entity_delete) if j == 0]
    return entities


if __name__ == '__main__':
    e = [[{'start_offset': 70, 'end_offset': 84, 'text': 'Terrible muscle'}],
         [{'start_offset': 103, 'end_offset': 138, 'text': 'Burning sensations in neck shoulders'}],
         [{'start_offset': 74, 'end_offset': 77, 'text': 'ible'},
          {'start_offset': 86, 'end_offset': 99, 'text': 'and joint pain'}],
         [{'start_offset': 74, 'end_offset': 84, 'text': 'ible muscle'},
          {'start_offset': 90, 'end_offset': 99, 'text': 'joint pain'}],
         [{'start_offset': 103, 'end_offset': 123, 'text': 'Burning sensations in'},
          {'start_offset': 144, 'end_offset': 154, 'text': 'upper chest'}],
         [{'start_offset': 140, 'end_offset': 142, 'text': 'and'},
          {'start_offset': 150, 'end_offset': 154, 'text': 'chest'}],
         [{'start_offset': 70, 'end_offset': 77, 'text': 'Terrible'},
          {'start_offset': 86, 'end_offset': 88, 'text': 'and'}],
         ]
    input_text = 'I was in deniel that the statins would cause ME side effects - I have Terrible muscle and joint pain , Burning sensations in neck shoulders and upper chest .'

    print(postprocess_entities(e, input_text))
    e = [{'start_offset': 2, 'end_offset': 2, 'text': 'e'}, {'start_offset': 4, 'end_offset': 9, 'text': 'statin'}]

    input_text = 'the statins'

    print(combine_entity_part(e, input_text))

    # e = [{'start_offset': 70, 'end_offset': 77, 'text': 'Terrible'},
    #      {'start_offset': 96, 'end_offset': 99, 'text': 'pain'}]
    # b = [{'start_offset': 70, 'end_offset': 84, 'text': 'Terrible muscle'},
    #      {'start_offset': 90, 'end_offset': 99, 'text': 'joint pain'}]
    # e2 = [{'start_offset': 70, 'end_offset': 77, 'text': 'Terrible'},
    #       {'start_offset': 96, 'end_offset': 99, 'text': 'pain'}]
    # print(judge_entity_included(b, e2))
