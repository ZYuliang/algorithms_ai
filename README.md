# algorithms_ai
algorithms_ai



得到标准格式的数据后
- 数据集的处理
1. 数据集的切分：训练（train），验证（dev），测试（test）
先切分出一个个样本 --sentences-tokenizer
然后normalize 把输入和输出都做规范化-----之后可加数据增强和特征
然后pretokenizer 预分词 ---去重，训练集采样
然后tokenizer
然后input_ids


2. 数据集的处理:(1)切句子（需要标签映射），（2）数据增强（召回正确，需要标签同时映射，增加表示数据类型的特征）和特征增强（3）构建输入输出，（4）根据输入进行去重（输出一致删一个，输出不一致全删掉）（5）删除可能错误的数据（需要具体处理），过采样，上采样，欠采样，下采样（6）把数据集变成模型需要的格式（input_ids）等，
3. 模型训练：（1）数据集的dataloader，sample等（2）模型的架构（3）模型的forward得到logits，（3）处理logits得到loss或者metric或结果
4.不同任务 -- 不同模型 --- 各种训练方式


4. 模型的部署：


zyl_utils.file_utils --- algorithms_ai.utils.file_utils
deep_learning.information_extraction --- algorithms_ai.deep_learning.information_extraction
zyl_utils.dataset_utils.dataset_utils --- algorithms_ai.dataset_utils.dataset_utils

