# 数据


# 模型
架构ECD：encoders-combiner-decoders
## 多个encoder
{
  "encoder":{
  "bert1":{
    "name_or_path":"",
    "finetune_layers":null,
    "para":...
},
  "bert2":{
    "name_or_path":"",
    "finetune_layers":null,
    "para":...
}
"combiner":{
  "starge":
  "para":
}
"decoder":{
  "cls":{
    "para":
  },
  "ner":{
    "para":
}
}
  "tasks":{
  "task1":{
  "encoder":"bert1",
  "decoder":"cls",
  "loss_weight":
  "input_column":"",
  "label_column":"",
  
}
}

}

}




{
  "dec": "假设输入的是一个batch的样本，一般最好训练的时候一个batch一个任务的交替训练，每个batch有个task标签，一个epoch训练完所有样本，，这个样本有很多key，不同的key给不同的encoder使用"，
  "des2": "最后的损失是所有损失相加，任务难度不一样导致有些任务过度学习，可以缩小学习率，或使用权重加权损失, ",
  "des3": "或者直接定义model。定义任务，",

  "encoders": {
    "e1": {
      "tokenizer": "",
      "path": "2",
      "para": {}
    },
    "e2": {
      "path": "3",
      "para": {}
    }
  },
  "decoders": {
    "e3": {
      "path": "4",
      "para": {}
    }
  },
  "tasks":{
    "task_1":{
      "features": [
        { "name":"a",
          "encoder":"e1",
        "type": "num"},
        { "name":"b",
          "encoder":"e2"}
      ],
      "decoder": "e3"
    },
    "task_2":{
      "features": [
        { "name":"a",
          "encoder":"e1",
        "type": "num"},
        { "name":"b",
          "encoder":"e2"}
      ],
      "decoder": "e4"
    }

  },
  "conbiner": {
  },
  "postprocess": "combine/split"
}