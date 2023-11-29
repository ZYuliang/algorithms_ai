import torch
import torch.nn as nn

from algorithms_ai.deep_learning.information_extraction.modeling_ie_embedding import SinusoidalPositionEmbedding, apply_rotary_position_embeddings


class GlobalPointerHead(nn.Module):
    def __init__(self, hidden_size, heads, head_size, RoPE=True, tril_mask=True):
        """
        lecun_normal

        三个不一样
        hidden_size:前一层输入的每个token的维度
        heads：指针图的数量
        head-size：指针图的size，（head-size，head-size）
        RoPE:是否使用旋转位置编码
        tril-mask：下三角掩码
        """
        super(GlobalPointerHead, self).__init__()
        self.hidden_size = hidden_size
        self.heads = heads  # 2 最后的图的维度，有几张图
        self.head_size = head_size  # q，k的最后一个维度，旋转编码的维度，需要偶数

        # self.dense_o1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.dense = nn.Linear(self.hidden_size, self.head_size * self.heads * 2)
        self.sinusoidal_position_embeddings = SinusoidalPositionEmbedding(self.head_size, merge_mode='zero')

        self.RoPE = RoPE
        self.tril_mask = tril_mask

    def forward(self, hidden_state, attention_mask=None):
        batch_size, seq_len = hidden_state.shape[0], hidden_state.shape[1]
        self.device = hidden_state.device
        # hidden_state = self.dense_o1(hidden_state)
        hidden_state = self.dense(hidden_state)  # [b,s,hidden-size]-->[b,s,2*heads*head_size]

        hidden_state = torch.split(hidden_state, self.head_size * 2, dim=-1)  # heads * [b,s,2*head_size]
        hidden_state = torch.stack(hidden_state, dim=-2)  # [b,s,heads,2*head_size]

        qw, kw = hidden_state[..., :self.head_size], hidden_state[...,
                                                     self.head_size:]  # [b,s,heads, head_size],[b,s,heads,head_size]
        if self.RoPE:
            pos_emb = self.sinusoidal_position_embeddings(hidden_state)  # [b,s,head_size]
            qw, kw = apply_rotary_position_embeddings(pos_emb, qw, kw)  # [b,s,heads,head_size],[b,s,heads,head_size]

        logits = torch.einsum('bmhd,bnhd->bhmn', qw, kw) / self.head_size ** 0.5  # [b,heads,s,s]

        # 排除padding,把所有padding的位置变成无穷小：-1e12，mask后只有图的attention-mask为1的地方有非无穷小的值
        pad_mask = attention_mask.unsqueeze(1).unsqueeze(1).expand(batch_size, self.heads, seq_len,
                                                                   seq_len)  # [b,heads,s,s]
        logits = logits * pad_mask - (1 - pad_mask) * 1e12  # [b,heads,s,s]

        # 在mask-padding的基础上进一步mask掉下三角部分，只保留上三角+对角线元素，其余地方设置为无穷小： -1e12
        if self.tril_mask:
            mask = torch.tril(torch.ones_like(logits), -1)
            logits = logits - mask * 1e12
        return logits  # [b,heads,s,s]


class EfficientGlobalPointerHead(nn.Module):
    def __init__(self, hidden_size, heads, head_size, RoPE=True, tril_mask=True):
        """
        heads
        """
        super(EfficientGlobalPointerHead, self).__init__()
        self.heads = heads  # 2 最后的图的维度，有几张图
        self.head_size = head_size  # q，k的最后一个维度，旋转编码的维度，需要偶数
        self.hidden_size = hidden_size

        self.p_dense = nn.Linear(self.hidden_size, self.head_size * 2)
        self.q_dense = nn.Linear(self.hidden_size, self.heads * 2)
        # self.q_dense = nn.Linear(self.head_size * 2, self.heads * 2)

        self.sinusoidal_position_embeddings = SinusoidalPositionEmbedding(self.head_size)

        self.RoPE = RoPE
        self.tril_mask = tril_mask

    def forward(self, hidden_state, attention_mask=None):
        self.device = hidden_state.device
        outputs = self.p_dense(hidden_state)  # [b,s,hidden-size]-->[b,s,2*head_size]
        qw, kw = outputs[..., ::2], outputs[..., 1::2]  # 从0,1开始间隔为2,[b,s,head_size],[b,s,head_size]
        if self.RoPE:
            pos_emb = self.sinusoidal_position_embeddings(outputs)  # [b,s,head_size]
            qw, kw = apply_rotary_position_embeddings(pos_emb, qw, kw)  # [b,s,head_size],[b,s,head_size]

        # 计算内积
        logits = torch.einsum('bmd,bnd->bmn', qw, kw) / self.head_size ** 0.5  # [b,s,s]
        bias = torch.einsum('bnh->bhn', self.q_dense(hidden_state)) / 2  # [b,s,2*head_size]
        # bias = torch.einsum('bnh->bhn', self.q_dense(outputs)) / 2

        # [b,1,s,s]+[b,heads,1,s]+[b,heads,s,1] --> [b,heads,s,s]
        logits = logits[:, None] + bias[:, ::2, None] + bias[:, 1::2, :, None]  # 一张图复制然后加上许多偏置

        # 排除padding,把所有padding的位置变成无穷小：-1e12，mask后只有图的attention-mask为1的地方有非无穷小的值
        logits = self.sequence_masking(logits, attention_mask, '-inf', logits.ndim - 2)  # 倒数第二维度mask
        logits = self.sequence_masking(logits, attention_mask, '-inf', logits.ndim - 1)  # 倒数第一维度mask

        # 在mask-padding的基础上进一步mask掉下三角部分，只保留上三角+对角线元素，其余地方设置为无穷小： -1e12
        if self.tril_mask:
            mask = torch.triu(torch.ones_like(logits), diagonal=0)
            logits = logits - (1 - mask) * 1e12

        return logits  # [b,heads,s,s]

    @staticmethod
    def sequence_masking(x, mask, value='-inf', axis=None):
        """为序列条件mask的函数
        mask: 形如(batch_size, seq_len)的0-1矩阵；
        value: mask部分要被替换成的值，可以是'-inf'或'inf'；
        axis: 序列所在轴，默认为1；
        除了mask的地方替换，其他地方值不变
        x = torch.randint(1,10,(2,4,3,3))
        mask = torch.Tensor([[1,1,0],[1,0,0]])
        mask2 = torch.unsqueeze(mask, 1)
        mask2 = torch.unsqueeze(mask2, 1)
        """
        if mask is None:
            return x
        else:
            if value == '-inf':
                value = -1e12
            elif value == 'inf':
                value = 1e12
            assert axis > 0, 'greater than 0'
            # attention_mask.unsqueeze(1).unsqueeze(1).expand(batch_size, self.ent_type_size, seq_len, seq_len)

            # 中间维度补
            for _ in range(axis - 1):
                mask = torch.unsqueeze(mask, 1)  # [b,1。。。1，s]
            # 其余维度补
            for _ in range(x.ndim - mask.ndim):
                mask = torch.unsqueeze(mask, mask.ndim)  # [b,1,s,1]
            return x * mask + value * (1 - mask)
