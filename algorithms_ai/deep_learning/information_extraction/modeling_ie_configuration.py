from transformers import PretrainedConfig


class IEModelConfig(PretrainedConfig):
    model_type = "information_extraction"

    def __init__(
            self,
            head_size=64,
            relation2id=None,
            entity2id=None,
            model_head='EfficientGlobalPointerHead',
            encoder_type='bert',
            encoder_path='bert_base_uncased',
            encoder_config=None,
            model_max_length=128,
            vocab_size = 20000,
            **kwargs
    ):
        super(IEModelConfig, self).__init__(**kwargs)
        self.entity2id = entity2id
        self.relation2id = relation2id
        self.head_size = head_size
        self.model_head = model_head
        self.encoder_type = encoder_type
        self.encoder_path = encoder_path
        self.encoder_config = encoder_config
        self.model_max_length = model_max_length
        self.vocab_size = vocab_size
