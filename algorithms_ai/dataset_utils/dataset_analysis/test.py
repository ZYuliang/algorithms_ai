from algorithms_ai.data.processor.tokenizer import WordTokenizer
from algorithms_ai.data.utils import get_char_categories
from loguru import logger
from nltk import pos_tag
from utils import analyze_unordered_numerical_array


# 实体标签的分析：
# 输入每个样本的实体标签
# 主要分析是否有错误数据，是否可以进行规则判别
# 分析的是处理过后的实体，


class Text:
    def __init__(self, text, word_tokenizer: WordTokenizer = None):
        self.text = text
        self.char_length = len(text)  # 文本字符长度
        self.tokens = word_tokenizer.run(text)  # 文本切分为token
        self.tokens_length = len(self.tokens)  # 文本的token长度
        self.pos_tag = '-'.join([k[-1] for k in pos_tag([i.text for i in self.tokens])])  # 文本的词性标注
        self.char_category = get_char_categories(text)  # 文本的字符种类


from collections import Counter


class LabelsAnalysis:
    def __init__(self, labels: list, word_tokenizer: WordTokenizer):
        self.labels = [Text(i, word_tokenizer) for i in labels]

    def run(self):
        self.analyze_label_text()
        self.analyze_label_length()
        self.analyze_tokens()
        self.analyze_pos_tag()
        self.analyze_category()

    def analyze_label_length(self):
        logger.info('每个实体包含字符数的分布:')
        all_char_length = [i.char_length for i in self.labels]
        analyze_unordered_numerical_array(all_char_length)

        logger.info('每个实体包含token数的分布:')
        all_token_length = [i.tokens_length for i in self.labels]
        analyze_unordered_numerical_array(all_token_length)

    def analyze_label_text(self, top_k=10):
        all_text = [i.text for i in self.labels]
        all_text = dict(Counter(all_text))
        all_text = sorted(all_text.items(), key=lambda x: x[-1], reverse=True)
        logger.info(f'最经常出现的实体：{all_text[0:top_k]}')

    def analyze_tokens(self, top_k=10):
        start_tokens = [i.tokens[0].text for i in self.labels]
        start_tokens = dict(Counter(start_tokens))
        start_tokens = sorted(start_tokens.items(), key=lambda x: x[-1], reverse=True)
        logger.info(f'最常出现在实体开头的token有：{start_tokens[0:top_k]}')

        end_tokens = [i.tokens[-1].text for i in self.labels]
        end_tokens = dict(Counter(end_tokens))
        end_tokens = sorted(end_tokens.items(), key=lambda x: x[-1], reverse=True)
        logger.info(f'最常出现在实体结尾的token有：{end_tokens[0:top_k]}')

        all_tokens = [j.text for i in self.labels for j in i.tokens]
        all_tokens = dict(Counter(all_tokens))
        all_tokens = sorted(all_tokens.items(), key=lambda x: x[-1], reverse=True)
        logger.info(f'最常出现在实体中（包括开头结尾）的token有：{all_tokens[0:top_k]}')

    def analyze_pos_tag(self):
        all_pos_tag = [i.pos_tag for i in self.labels]
        all_pos_tag = dict(Counter(all_pos_tag))
        all_pos_tag = sorted(all_pos_tag.items(), key=lambda x: x[-1], reverse=True)
        logger.info(f'所有实体可能的词性标注模板：{all_pos_tag}')

    def analyze_category(self):
        all_char_categories = [i.char_category for i in self.labels]
        all_char_categories = dict(Counter(all_char_categories))
        all_char_categories = sorted(all_char_categories.items(), key=lambda x: x[-1], reverse=True)
        logger.info(f'所有实体字符类型可能的组合：{all_char_categories}')


class NerSamples:
    def __init__(self, ner_samples, word_tokenizer):
        """
        一般一个样本是一个dict，必须包含‘input_text’ (一般是句的文本)，‘labels’：0或多个实体
        :param ner_samples: 多个样本的列表
        :param word_tokenizer:
        """
        pass


if __name__ == '__main__':
    f = '/home/zyl/disk/PharmAI/Pipelines/components/data_mining/data_mining/data/raw/technology.json'
    from algorithms_ai.data.utils import load_data

    dt = load_data(f, 'json')
    data = []
    for i in dt:
        data.append(i['name_en'])
    all_labels = [j for i in data for j in i]
    LabelsAnalysis(labels=all_labels, word_tokenizer=WordTokenizer()).run()
    # NerAnalysis(data).run()
    #
    # print(1)
    # f2 = '/home/zyl/disk/PharmAI/Pipelines/components/data_mining/data_mining/data/other/test.xlsx'
    # import pandas as pd
    # dt = pd.read_excel(f2)
    # a = dt['title_title_num'].tolist()
    # a = [i for i in a if 396>i>12]
    # analyze_unordered_numerical_array(a,bins=100)

    pass
