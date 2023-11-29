import numpy as np
import torch


def sparse_multilabel_categorical_crossentropy(y_true, y_pred, mask_zero=False):
    """稀疏版多标签分类的交叉熵
    y_true:[batch-size,heads,entity_num]
    y_pred:[batch-size,heads,seq*seq]
    如果logit大于0，exp(logit)>1,log(exp(logit))>0
    说明：
        1. y_true.shape=[..., num_positive]，
           y_pred.shape=[..., num_classes]； 如果无意义，默认值为-1e_12,它的exp为0，它的log_exp为无穷小
        2. 请保证y_pred的值域是全体实数，换言之一般情况下
           y_pred不用加激活函数，尤其是不能加sigmoid或者
           softmax；
        3. 预测阶段则输出y_pred大于0的类；
        4. 详情请看：https://kexue.fm/archives/7359 。
        5. mask_zero表面意思
    """
    zeros = torch.zeros_like(y_pred[..., :1])  # [8, 11, 1] 取最后一个维度第一个元素的值，作为0
    y_pred = torch.cat([y_pred, zeros], dim=-1)  # [8, 11, 129*129+1] # 铺平的图后面加上一个元素维度的0
    if mask_zero:
        # 对于平铺后的预测，第一个值设为无穷大，
        infs = zeros + 1e12  # [8, 11, 1] 无穷大
        y_pred = torch.cat([infs, y_pred[..., 1:]],
                           dim=-1)  # [8, 11, 129*129+1] # 拼接，最后一个维度第一个元素无穷大，最后一个元素为0,这个元素0实际是阈值，exp(0)==1,log(1)==0,而无穷小对应的exp为0

    y_pos_2 = torch.gather(y_pred, index=y_true, dim=-1)  # [8, 11, 6] 从预测结果中抽取真实结果是真的index,和y_true的shape一样
    y_pos_1 = torch.cat([y_pos_2, zeros], dim=-1)  # [8, 11, 7] 上面的最后一个维度再加上0
    if mask_zero:
        # infs = zeros + 1e12
        # 对于平铺后的预测，第一个值设为无穷小
        y_pred = torch.cat([-infs, y_pred[..., 1:]], dim=-1)  # [8, 11, 129*129+1] # 拼接，最后一个维度第一个元素无穷小，最后一个元素为0
        y_pos_2 = torch.gather(y_pred, index=y_true, dim=-1)  # [8, 11, 6] 就是上面的y-pos-2

    pos_loss = torch.logsumexp(-y_pos_1,
                               dim=-1)  # torch.Size([8, 11]) 真实结果是真，预测结果对应的部分进行logsumexp，实际log里面的值是在1左右进行加减，如果是无穷，则为0，如果是正值，则最终概率值大于0
    all_loss = torch.logsumexp(y_pred, dim=-1)  # torch.Size([8,11]) 每张图进行log-exp得到损失,mask掉的地方为无穷小，对应概率为0,
    # 在无意义的地方，logit为-1e12，无穷小，然后它的exp为0，如果有意义，则它的exp绝对大于0，然后就有sum_exp大于0，然后就有了log-sum-exp
    aux_loss = torch.logsumexp(y_pos_2, dim=-1) - all_loss  # torch.Size([8, 11])  # pos2的损失减去所有的损失
    aux_loss = torch.clip(1 - torch.exp(aux_loss), 1e-7, 1)
    neg_loss = all_loss + torch.log(aux_loss)  # torch.Size([8, 11])
    # print(torch.sum(pos_loss), torch.sum(neg_loss))
    return pos_loss, neg_loss


def loss_for_ie(y_true, y_pred, mask_zero):
    '''
        稀疏多标签交叉熵损失的torch实现
        y_true:  torch.Size([8, 11, 6, 2]) batch,heads,max_entities_num,2
        y_pred:  torch.Size([8, 11, 129, 129]) batch,heads,seq,seq
    '''
    shape = y_pred.shape  # [b, heads, seq, seq]
    # 根据实际的坐标（），把这些坐标在平铺后的一维向量上的索引表示出来，比如坐标（1，2）在实际图平铺后的索引为 1*seq+2
    # 也就是把预测的图进行了平铺，然后平铺后标记为1的索引也得到了
    y_true = y_true[..., 0] * shape[2] + y_true[..., 1]  # [b, heads,entities_num] 第一个维度（坐标）乘以seq 加上第二个维度（坐标），再相加
    y_pred = y_pred.reshape(shape[0], -1, np.prod(shape[2:]))  # torch.Size([b, heads, seq*seq]) 把最后两个维度铺平，把图铺平
    loss = sparse_multilabel_categorical_crossentropy(y_true, y_pred, mask_zero=mask_zero)
    return torch.mean(torch.sum(loss[0] + loss[1], dim=1))  # .type(torch.float16)
