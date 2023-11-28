f  ="/home/zyl/disk/PharmAI/Pipelines/components/deepmed_ctrain/data/V_2_8/dev.arms_ner.tsv"
from loguru import logger





class Analysis:
    def __init__(self, samples, model_tokenizer=None, word_tokenizer=None):
        self.samples = samples
        self.model_tokenizer = self.init_model_tokenizer(model_tokenizer)
        self.word_tokenizer = self.init_word_tokenizer(word_tokenizer)
        # feature: string(pattern,length),category（多类别）,number（线性）,binary（二分类）,
        # set(多标签)，序列（一组数字），ner-label

    def init_model_tokenizer(self, model_tokenizer=None):
        if not model_tokenizer:
            from transformers import BertTokenizer
            model_tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased').tokenize
        return model_tokenizer

    def init_word_tokenizer(self, word_tokenizer=None):
        if not word_tokenizer:
            from NERProcessor import WordTokenizer
            word_tokenizer = WordTokenizer().run
        return word_tokenizer

    def run(self, input_key, output_key, features_key):
        logger.info(f'sample length: {len(self.samples)}')

        input_x = []
        output_y = []
        features = dict()
        for feature in features_key:
            features[feature] = []

        for s in self.samples:
            if input_key in s and output_key in s:

                input_x.append(s[input_key])
                output_y.append(s[output_key])
            else:
                logger.error(f'error data: {s}')
            for feature in features_key:
                features[feature].append(s.get(feature, 'None'))

        print(1)

    @staticmethod
    def get_text_string_length(text: str):
        return len(text)

    @staticmethod
    def get_text_token_length(text: str, tokenizer=None):
        if not tokenizer:
            from transformers import BertTokenizer
            tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        return len(tokenizer.tokenize(text))

    def get_string_info(self, string):
        char_length = len(string)
        model_token_length = self.model_tokenizer(string)
        word_token_length = self.word_tokenizer(string)

    def get_sentence_length_distribution(self, input_texts):
        # 句子长度分布 string,model_token，设定max-sen-length,去除极异常值
        res = []

        for i in tqdm(input_texts,desc='deal with data'):
            res.append({
                'token_len': len(self.model_tokenizer.tokenize(i)),
                'len':len(i),
                # 'word_token_len':self.word_tokenizer(i)
            })
        df = pd.DataFrame(res)
        desc = df.describe(percentiles=[0.10, 0.25, 0.5, 0.75, 0.85, 0.95, 0.99])
        print(desc)
        info = df.info()
        print(info)


    def get_tags_distribution(self):
        # B，I，O的分布，word-token级别， 查看是否标注的数据有问题
        pass

    def get_tag_sequences_distribution(self):
        # 实体长度的平均，最大，最小,估计实体大概多长，难度多大，数据有多少
        pass

    def get_token_distribution(self):
        # token级别的分布，查看哪些token最多，以及是否要使用stop-words
        pass

    def get_vocabulary_distribution(self):
        # 查看字符的大小以及怎样处理
        pass

    def check_incorrect_predictions(self):
        # 查看出错的数据的分布
        pass
