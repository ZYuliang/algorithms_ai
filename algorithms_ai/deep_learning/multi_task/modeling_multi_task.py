
from transformers import PreTrainedModel,BertModel,AutoModel
from configuration_multi_task import MultiTaskConfig
from torch import nn

class MultiTaskModel(PreTrainedModel):
    config_class = MultiTaskConfig

    def __init__(self, config:MultiTaskConfig):
        super().__init__(config)

        # self.encoders = dict()
        # for i in config.encoders:
        #     self.encoders[i] = AutoModel.from_pretrained(i)
        # self.encoders = nn.ModuleDict(self.encoders)

        self.multi_task_encoder = AutoModel.from_pretrained(config.encoder)

        self.tasks = dict()
        for i in config.tasks:
            self.tasks[i.task_name]=i


        # block_layer = BLOCK_MAPPING[config.block_type]
        # self.embedding = BertModel(
        #     block_layer,
        #     config.layers,
        #     num_classes=config.num_classes,
        #     in_chans=config.input_channels,
        #     cardinality=config.cardinality,
        #     base_width=config.base_width,
        #     stem_width=config.stem_width,
        #     stem_type=config.stem_type,
        #     avg_down=config.avg_down,
        # )

    def __init_task_module(self):
        pass


    def forward(self, tensor, task,):
        # self.encoders[self.tasks[task].encoder].forward(tensor)
        self.multi_task_encoder[]

        return self.model.forward_features(tensor)

if __name__ == '__main__':
    from datasets import load_dataset
    imdb = load_dataset("imdb")
    print(1)

    # tokenized_imdb = imdb.map(preprocess_function, batched=True)
    # tokenized_imdb_train = tokenized_imdb["train"].select(range(1000))
    # tokenized_imdb_eval = tokenized_imdb["test"].select(range(1000))


    from configuration_multi_task import Task
    task1 = Task(task_name='cls1',loss_weight=12,input_column='1',label_column='2')
    task2 = Task(task_name='cls2',loss_weight=12,input_column='1',label_column='2')
    c = MultiTaskConfig(encoder='bert-base-uncased',
                        tasks=(task1,task2))
    m = MultiTaskModel(config=c)
    print(m)

