"""
有很多和原始token不同的常用词汇，也就是使用的语料库和原始预训练的语料库差距很大，这时候需要tokenizer，会生成完全不同的vocab
如果使用的语料库和原始语料库相近，或者使用的语料库不够大，那这个时候就用原始的tokenizer好了，因为训练得到的tokenizer不会比原始的好很很多，甚至会更差
所以如果使用专业的领域预料，最好有更多相关的预料，包括一些通用的预料，和一些相关的专业预料
词表大小,词表大小应该是和语料大小去匹配的，具体设置我们可以参考下ChatGLM、和一些Chinese-LLaMA模型，像ChatGLM词表大小有13万，其他Chinese-LLaMA模型基本上在5万-8万左右。词表大小设置是否合理直接影响了模型参数以及训练速度

tokenizer是单独的，是由语料库决定的，如果要在hugging-face模型中使用就要用PreTrainedTokenizer
"""


class MyTokenizer:
    def __init__(self, save_dir):
        self.save_dir = save_dir
        pass

    def train_from_model(self, model_path, train_corpus, vocab_size=52000, do_lower_case=False,
                         add_alphabet=True, add_old_tokens=False):
        from transformers import AutoTokenizer
        old_tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_path,
                                                      do_lower_case=do_lower_case)

        if add_alphabet:
            new_tokenizer = old_tokenizer.train_new_from_iterator(train_corpus, vocab_size,
                                                                  initial_alphabet=list(
                                                                      old_tokenizer.get_vocab().keys()))
        else:
            new_tokenizer = old_tokenizer.train_new_from_iterator(train_corpus, vocab_size)

        # tokenizer.backend_tokenizer.model

        if add_old_tokens:
            new_tokenizer.add_tokens(list(old_tokenizer.get_vocab().keys()))

        new_tokenizer.save_pretrained(self.save_dir)
        return new_tokenizer  # PreTrainedTokenizer

    def get_tokenizer(self):
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(pretrained_model_name_or_path=self.save_dir)

    def train_from_vocab(self, vocab, tokenizer_type='WordPiece'):
        # 返回的是Tokenizer，如果要用于模型，则自定义继承PreTrainedTokenizer类
        from tokenizers import Tokenizer
        from tokenizers.models import WordPiece, BPE, Unigram, WordLevel
        from tokenizers import trainers
        from tokenizers import decoders

        if tokenizer_type == 'WordPiece':
            my_tokenizer = Tokenizer(WordPiece(vocab=vocab, unk_token="[UNK]", max_input_chars_per_word=20))
            my_trainer = trainers.WordPieceTrainer(vocab_size=90, min_frequency=3,
                                                   special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])

        elif tokenizer_type == 'WordLevel':
            my_tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
            my_trainer = trainers.WordLevelTrainer()
            my_tokenizer.decoder = decoders.WordPiece()
        elif tokenizer_type == 'BPE':
            my_tokenizer = Tokenizer(BPE(vocab=vocab, merges=[('l', 'l'), ('ll', 'o')], unk_token="[UNK]"))
            my_trainer = trainers.BpeTrainer()
            my_tokenizer.decoder = decoders.BPEDecoder()
        elif tokenizer_type == 'Unigram':
            my_tokenizer = Tokenizer(Unigram(vocab=vocab, unk_id=None, byte_fallback=None))
            my_trainer = trainers.UnigramTrainer(vocab_size=20000, special_tokens=["<PAD>", "<BOS>", "<EOS>"])
            my_tokenizer.decoder = decoders.ByteLevel()
        else:
            my_tokenizer = Tokenizer(WordPiece(vocab=vocab, unk_token="[UNK]", max_input_chars_per_word=20))
            my_trainer = trainers.WordPieceTrainer(vocab_size=90, min_frequency=3,
                                                   special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
            my_tokenizer.decoder = decoders.WordPiece()

        from tokenizers import normalizers
        my_tokenizer.normalizer = normalizers.Sequence([normalizers.NFD(),
                                                        normalizers.Lowercase(),
                                                        normalizers.StripAccents()])

        from tokenizers import pre_tokenizers
        my_tokenizer.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.Punctuation(),
                                                              pre_tokenizers.WhitespaceSplit()])

        from tokenizers import processors
        my_tokenizer.post_processor = processors.TemplateProcessing(
            single="[CLS] $A [SEP]",
            pair="[CLS] $A [SEP] $B:1 [SEP]:1",
            special_tokens=[
                ("[CLS]", 1),
                ("[SEP]", 2),
            ],
        )

        my_tokenizer.train_from_iterator(data, trainer=my_trainer)

        # my_tokenizer.encode
        # my_tokenizer.save("test_bert_vocab.json")
        # Tokenizer.from_file("test_bert_vocab.json")
        return my_tokenizer


if __name__ == '__main__':
    data = [
        "Beautiful is better than ugly.",
        "Explicit is better than implicit.",
        "Simple is better than complex.",
        "Complex is better than complicated.",
        "Flat is better than nested.",
        "Sparse is better than dense.",
        "Readability counts.",
    ]
    m_p = "/large_files/pretrained_pytorch/biobert-base-cased-v1.2"
    m_t = MyTokenizer('./save_tokenier')
    m_t.train_from_vocab(vocab=None, tokenizer_type='WordPiece')
    # m_t.train_from_model(m_p,data)
    print(2)
