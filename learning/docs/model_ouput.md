单标签多分类
logits: 是（B * C） ，B表示样本数，C表示类别数

array([[ 2.94216871e-01,  1.38212293e-01, -4.08781081e-01],
       [ 6.50090352e-02,  9.60926190e-02, -9.94987115e-02],
       [ 2.96252578e-01, -1.25854552e-01, -4.38775182e-01],
       [ 5.36635220e-01,  1.58213630e-01, -1.42326221e-01],
       [-1.57728076e-01, -1.10996835e-01, -5.18414199e-01]], dtype=float32)

labels: 每个index表示这个样本应该属于哪个类别
array([0, 2, 0, 0, 1]) 

#
多标签二分类
logits: 是（B * L） ，B表示样本数，L表示标签数，每个标签的元素表示属于这个标签的logit

array([[-0.1562816 ,  0.2501327 ],
       [-0.449539  ,  0.45732367],
       [-0.65591633,  0.44608763],
       [-0.49023682,  0.32599747],
       [-0.3568234 ,  0.6620542 ]], dtype=float32)

labels: 是（B * L）,
array([[0., 1.],
       [0., 1.],
       [1., 1.],
       [0., 1.],
       [1., 0.]], dtype=float32)

# 
ner
logits :  (B * max_length * L) , 每个样本有token个数，每个token有对应的每个token标签的个数
(3, 4, 5)
array([[[ 2.2506793 , -0.8132773 , -0.01186355,  0.14490508,
          0.93542314],
        [ 2.443797  ,  0.03088955, -0.16494827, -0.1682785 ,
          0.23977919],
        [ 2.040088  , -0.6232247 , -0.61508423, -0.15205576,
          0.7342508 ],
        [ 1.8656479 , -0.7564634 , -0.66188306,  0.13958456,
          1.0774814 ]],
       [[ 2.084756  , -0.58094364, -0.27040902,  0.22114624,
          0.81494707],
        [ 2.497244  ,  0.01945196, -0.37512547, -0.15108521,
          0.18048623],
        [ 2.0875173 , -0.4600353 , -0.7967091 ,  0.14546968,
          0.3282098 ],
        [ 1.9215117 , -0.46921113, -0.57524234,  0.03791665,
          0.22812463]],
       [[ 2.2448199 , -0.57947654, -0.11964509,  0.21332876,
          0.4830669 ],
        [ 2.5065737 ,  0.14929567, -0.4059034 , -0.15295218,
          0.01763484],
        [ 2.35984   , -0.5419374 , -0.8941026 , -0.06301189,
          0.33747166],
        [ 2.2343924 , -0.851829  , -0.39760923, -0.02063408,
          0.34981337]]], dtype=float32)


labels:  B*max_length 每个样本的每个token的label值
array([[-100,    0,    0,    0],
       [-100,    0,    1,    0],
       [-100,    0,    2,    1]])

##
使用bert的输出，一个是last_hidden_state 模型encoder的最后一层输出，这个提供给ner，另一个是pooler_output该句子语义的特征向量表示,提供给cls，是encoder经过pooler线性层得到的，first_token_tensor = hidden_states[:, 0]使用计算的是每句话的CLS 的token
last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
pooler_output (`torch.FloatTensor` of shape `(batch_size, hidden_size)`):
            Last layer hidden-state of the first token of the sequence (classification token)               after further processing
            through the layers used for the auxiliary pretraining task. E.g. for BERT-family of             models, this returns
            the classification token after processing through a linear layer and a tanh                     activation function. The linear
            layer weights are trained from the next sentence prediction (classification)                        objective during pretraining.

