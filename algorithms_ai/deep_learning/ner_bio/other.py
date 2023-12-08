default_args = {
    # data
    'dataloader_drop_last': False,  # 是否删除最后一个不完整的batch
    'dataloader_num_workers': 0,  # 多进程处理数据
    'remove_unused_columns': True,  # 自动取出多余的列
    # 'label_names':  # label相关的列的名称，List[str]，默认["label"],如果是问答则是["start_positions", "end_positions"]
    # 'disable_tqdm': # 是否关闭进度和metric
    # 'past_index'
    # 'ignore_data_skip':False
    'group_by_length': False,  # 动态padding时使用，
    'dataloader_pin_memory': True,  # 固定数据的内存

    # train
    'num_train_epochs': 3,  # 训练epoch
    'max_steps': -1,  # 总的迭代次数，如果设置会覆盖epoch
    'per_device_train_batch_size': 8,  # 每个设备（可能是GPU）的训练batch大小
    'gradient_accumulation_steps': 1,  # 梯度累计step数
    'weight_decay': 0,  # 权重衰减，L2正则化
    'max_grad_norm': 1,  # 梯度截断，控制梯度膨胀
    'no_cuda': False,  # 是否使用GPU
    'seed': 42,  # 训练时的种子
    # 'data_seed': # 数据种子，如果不设置就默认为训练的种子
    # 'deepspeed',fsdp,sharded_ddp,fp16,resume_from_checkpoint,gradient_checkpointing，auto_find_batch_size，
    # full_determinism，torchdynamo

    # 优化器和学习器
    'optim': 'adamw_hf',
    # 'optim_args':  # 参数
    'learning_rate': 5e-5,  # lr,默认使用AdamW优化器
    'lr_scheduler_type': 'linear',  # 学习率优化
    'adam_beta1': 0.9,
    'adam_beta2': 0.999,
    'adam_epsilon': 1e-8,
    'warmup_ratio': 0.0,  # 预热的整个step的比例
    'warmup_steps': 0,  # 预热的步数，如果设置，会覆盖比例

    # evaluate
    'evaluation_strategy': 'steps',  # 默认若干步评估一次，同时log输出一次
    # 'eval_steps':  # 评估的间隔step数，如果不设置，就会和logging_steps一样
    'per_device_eval_batch_size': 8,  # 每个设备（可能是GPU）的评估batch大小
    # 'eval_accumulation_steps': , # 在把结果放到cpu上前，评估累计的次数，不设置就评估整个评估集
    # 'eval_delay': # 评估的延迟步数，一般不用
    'load_best_model_at_end': False,  # 最后载入最好的指标的那次模型，
    # 如果设置为True，save_strategy要和evaluation_strategy一样，`save_steps` 必须是 `eval_steps`的整数倍
    # 'metric_for_best_model':, # 最好模型的指标，主要要加上eval_ 前缀
    # 'greater_is_better':True # 模型指标的选择
    'include_inputs_for_metrics': False,  # 是否把输入传给metric计算

    # log
    'logging_dir': './output_dir/runs',  # log日志保存在哪
    'logging_strategy': 'steps',  # 日志的保存策略，默认steps
    'logging_steps': 500,  # log策略是steps时的步骤数
    'log_level': 'passive',  # 主进程的log的等级
    'log_level_replica': 'warning',  # 重复log的等级
    'log_on_each_node': True,  # 分布式训练是否log每个节点
    'logging_first_step': False,  # 是否评估和log开始时的结果
    'logging_nan_inf_filter': True,  # 是否过滤空值和无穷小的loss
    # 'report_to':  上传到哪
    # 'run_name':  名称
    'skip_memory_metrics': True,  # 是否跳过内存检查
    'push_to_hub': False,

    # save-model
    'overwrite_output_dir': False,  # 是否覆写输出的路径
    'save_strategy': "steps",  # 保存策略，默认steps
    'save_steps': 500,  # steps时保存的步骤
    'save_total_limit': 2,  # 最大存储的模型数
    'save_on_each_node': False,  # 多节点训练时，是否在每个节点上存储模型，还是只在主节点上存储

    # predict
    "prediction_loss_only": False,  # 是否预测时只输出损失
    'jit_mode_eval': False,  # 是否使用jit进行inference
}
# 设置经过若干steps进行评估，log和save


def test_train(self):
    wandb.init(project='deepmed')
    train_dataset = self.dataset_dict['train']
    model = self.get_model()
    trainer = Trainer(
        model=model,
        args=self.get_train_args(dataset_length=train_dataset.num_rows,
                                 num_train_epochs=10,
                                 per_device_train_batch_size=168,
                                 per_device_eval_batch_size=8),
        train_dataset=train_dataset,
        eval_dataset=self.dataset_dict['eval'].select(range(8)),
        tokenizer=self.tokenizer,
        data_collator=self.data_collator,
        compute_metrics=self.compute_metrics,
    )
    trainer.train()


def train_with_hp(self, n_trials=5):
    train_dataset = self.dataset_dict['train']
    self.train_args = self.get_train_args(dataset_length=train_dataset.num_rows,
                                          num_train_epochs=20,
                                          wandb_run_name='arms_ner',
                                          per_device_train_batch_size=168,
                                          per_device_eval_batch_size=32,
                                          )
    trainer = Trainer(
        model=None,
        model_init=self.get_model,
        args=self.train_args,
        train_dataset=self.dataset_dict['train'],
        eval_dataset=self.dataset_dict['eval'],
        tokenizer=self.tokenizer,
        data_collator=self.data_collator,
        compute_metrics=self.compute_metrics,
    )

    def wandb_hp_space(trial):
        return {
            "method": "bayes",
            "metric": {"name": self.metric, "goal": "maximize"},
            "parameters": {
                "learning_rate": {"distribution": "uniform", "min": 1e-7, "max": 1e-4},
                # "weight_decay": {"value": 20},
                # "max_grad_norm":{"value": 20},
                # "warmup_ratio":{'max': 0.1, 'min': 0.0001},
                "num_train_epochs": {"values": [10, 20, 30, 50]}
            },
            "name": "arms_ner",
            "project": 'deepmed'
        }

    def my_objective(metrics):
        return metrics[self.metric]

    best_trial = trainer.hyperparameter_search(
        backend="wandb",
        hp_space=wandb_hp_space,
        n_trials=n_trials,
        compute_objective=my_objective,
        metric=self.metric,
        direction="maximize",
    )
    print(best_trial)