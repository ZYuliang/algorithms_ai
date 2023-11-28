# 实体识别数据集处理

def split_sample_and_label(sample, sentence_tokenizer=None, ner_keys=None):
    # 对一组结果和对应的标签进行分句
    input_text = sample['input_text']
    refined_samples = []
    if sentence_tokenizer:
        sentence_token_result = sentence_tokenizer(input_text)
        offsets_mapping = sentence_token_result['offsets_mapping']
        texts = sentence_token_result['texts']

        if ner_keys:
            ner_keys_res_dict = {i: {'input_text': texts[i]} for i in range(len(texts))}
            for i, j in ner_keys_res_dict.items(): j.update({j: [] for j in ner_keys})

            for ner_key in ner_keys:
                ner_key_res = sample.get(ner_key, [])
                for each_entity in ner_key_res:
                    seq_indexs = [find_seq_index((_['start_offset'], _['end_offset']), offsets_mapping)
                                  for _ in each_entity]
                    if -1 in seq_indexs or len(set(seq_indexs)) > 1:  # 确保一个实体只在一个输入（一句话）中出现
                        return []
                    seq_index = seq_indexs[0]
                    refined_entity = [{'text': _['text'],
                                       'start_offset': _['start_offset'] - offsets_mapping[seq_index][0],
                                       'end_offset': _['end_offset'] - offsets_mapping[seq_index][0]}
                                      for _ in each_entity]
                    ner_keys_res_dict[seq_index][ner_key].append(refined_entity)
            refined_samples.extend(ner_keys_res_dict.values())
        return refined_samples
    else:
        return [sample]


def find_seq_index(seq, seq_list):
    # 从一组offsets_mapping中找到某个seq的所属，否则返回 -1
    for sub_seq_index, sub_seq in enumerate(seq_list):
        if sub_seq[0] <= seq[0] <= seq[1] <= sub_seq[1]:
            return sub_seq_index
    return -1


if __name__ == '__main__':
    sample = {
        'input_text': 'Terrible first of all I had diarrhoea my doseage was increased to 75 and a few days ago my feet and ankles started swelling up.\nI went to my GP and the results of my blood test showed there was something wrong with my liver and I am a non-drinker.\nMy blood pressure had gone sky high so my that is why the swelling.\nI was told to stop taking Arthrotec and am now using a natural product that seems to work it is called Pernamax.\n',
        'NER_ADR': [[{'text': 'diarrhoea', 'start_offset': 28, 'end_offset': 36}],
                    [{'text': 'swelling', 'start_offset': 115, 'end_offset': 122},
                     {'text': 'feet', 'start_offset': 91, 'end_offset': 94}],
                    [{'text': 'wrong with my liver', 'start_offset': 204, 'end_offset': 222}],
                    [{'text': 'blood pressure had gone sky high',
                      'start_offset': 251,
                      'end_offset': 282}],
                    [{'text': 'ankles', 'start_offset': 100, 'end_offset': 105},
                     {'text': 'swelling', 'start_offset': 115, 'end_offset': 122}],
                    [{'text': 'swelling', 'start_offset': 306, 'end_offset': 313}]],
        'NER_Drug': [[{'text': 'Arthrotec', 'start_offset': 342, 'end_offset': 350}],
                     [{'text': 'Pernamax', 'start_offset': 419, 'end_offset': 426}]]
    }

    from algorithms_ai.utils.tokenizer_utils.regex_tokenizer import RegexTokenizer

    r_t = RegexTokenizer(regex='\n').run
    print(split_sample_and_label(sample=sample, sentence_tokenizer=None,
                                 ner_keys=['NER_ADR', 'NER_Drug']))
