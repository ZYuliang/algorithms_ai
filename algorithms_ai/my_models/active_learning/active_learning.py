"""
least：以模型为标注进行区分
margin：严格区分相近的两个标签
ration：区分相近的两个标签，比margin的严格强度弱一点
entropy：关注的更多是不同标签的区分，是least的加强版
# Monte Carlo Dropout：预测期间的随机辍学会为域内数据产生高斯变化，如果您的模型没有为预测提供可靠的置信度估计，这将很有用
"""

import numpy as np
class UncertaintySampling:
    @staticmethod
    def least_confidence(logits, labels=None):
        """
        比较多个样本最大概率，选择其中最小的作为置信度最低的作为输出
        计算每个样本的最大类别的概率,进行从小到达排序并抽取样本index
        如果针对二分类问题，least confident 和 margin sampling 其实是等价的
        这个最大概率越小越应该被选择出来
        """
        logits = np.exp(logits)
        softmax_logits = logits / np.sum(logits, axis=1).reshape(logits.shape[0], 1)

        if isinstance(labels, np.ndarray) or isinstance(labels, list):
            max_pro = [i[j] for i, j in zip(softmax_logits, labels)]
        else:
            max_pro = np.max(softmax_logits, axis=1)
        return np.argsort(max_pro)

    @staticmethod
    def margin_confidence(logits, labels=None):
        """ 适用于容易混淆的数据，比如一个输入既是又是，成对出现的样本
        对于每个样本,选择其中最大概率和第二大概率的差异值,这个差异值越大，表示效果越好
        这个差异值越小越应该被选择出来
        """
        logits = np.exp(logits)
        softmax_logits = logits / np.sum(logits, axis=1).reshape(logits.shape[0], 1)

        if isinstance(labels, np.ndarray) or isinstance(labels, list):
            difference = []
            for i, j in zip(softmax_logits, labels):
                sorted_res = np.argsort(-i)
                difference.append(i[j] - sorted_res[0])
        else:
            difference = []
            for i in softmax_logits:
                sorted_res = np.argsort(-i)
                difference.append(sorted_res[0] - sorted_res[1])
        return np.argsort(difference)


    @staticmethod
    def ratio_confidence(logits, labels=None):
        """适用于容易混淆的数据，比如一个输入既是又是，成对出现的样本
        对于每个样本,选择其中最大概率和第二大概率的比值作为差异值，从小到大选择
        这个比值越小越应该被选择出来
        """
        logits = np.exp(logits)
        softmax_logits = logits / np.sum(logits, axis=1).reshape(logits.shape[0], 1)

        if isinstance(labels, np.ndarray) or isinstance(labels, list):
            difference = []
            for i, j in zip(softmax_logits, labels):
                sorted_res = np.argsort(-i)
                difference.append(i[j]/max(sorted_res[0],1e-5))
        else:
            difference = []
            for i in softmax_logits:
                sorted_res = np.argsort(-i)
                difference.append(sorted_res[0]/max(sorted_res[1],1e-5))

        return np.argsort(difference)

    @staticmethod
    def entropy_based(logits, labels=None):
        """ 或从整体熵区别不同标签的差别，而不是相近标签或标记
        基于信息熵的方法
        熵越大越不好，越有标的价值
        """
        logits = np.exp(logits)
        softmax_logits = logits / np.sum(logits, axis=1).reshape(logits.shape[0], 1)

        if isinstance(labels, np.ndarray) or isinstance(labels, list):
            entropy = [np.log2(i[j]) for i, j in zip(softmax_logits, labels)]
        else:
            entropy = softmax_logits * np.log2(softmax_logits)
            entropy = np.sum(entropy, axis=1)
        return np.argsort(entropy)



if __name__ == '__main__':
    logits = np.array([[2.94216871e-01, 1.38212293e-01, -4.08781081e-01],
                       [6.50090352e-02, 9.60926190e-02, -9.94987115e-02],
                       [2.96252578e-01, -1.25854552e-01, -4.38775182e-01],
                       [5.36635220e-01, 1.58213630e-01, -1.42326221e-01],
                       [-1.57728076e-01, -1.10996835e-01, -5.18414199e-01]])
    labels = np.array([0, 2, 0, 0, 1])
    print(UncertaintySampling().margin_confidence(logits, labels=labels))
    print(UncertaintySampling().margin_confidence(logits, labels=None))
    print(UncertaintySampling().ratio_confidence(logits, labels=labels))
    print(UncertaintySampling().ratio_confidence(logits, labels=None))
    print(UncertaintySampling().entropy_based(logits, labels=labels))
    print(UncertaintySampling().entropy_based(logits, labels=None))
    # logits = np.exp(logits)
    # print(logits)
    # print(logits /np.sum(logits,axis=1).reshape(logits.shape[0],1))

    # logits = np.argmax(logits, axis=1)
    # UncertaintySampling().least_confidence(logits,labels)
