import torch
import torch.nn as nn


def apply_rotary_position_embeddings(position_embeddings, *tensors):
    """应用RoPE到tensors中
        其中，sinusoidal.shape=[1, n, d]，tensors为tensor的列表，而
        tensor.shape=[b, n, ..., d]。
    """
    if tensors[0].ndim == 4:
        cos_pos = position_embeddings[..., None, 1::2].repeat_interleave(2, dim=-1)
        sin_pos = position_embeddings[..., None, ::2].repeat_interleave(2, dim=-1)
    else:
        cos_pos = position_embeddings[..., 1::2].repeat_interleave(2, dim=-1)
        sin_pos = position_embeddings[..., ::2].repeat_interleave(2, dim=-1)

    outputs = []
    for tensor in tensors:
        tensor2 = torch.stack([-tensor[..., 1::2], tensor[..., ::2]], -1)
        tensor2 = tensor2.reshape(tensor.shape)
        outputs.append(tensor * cos_pos + tensor2 * sin_pos)
    return outputs[0] if len(outputs) == 1 else outputs


class SinusoidalPositionEmbedding(nn.Module):
    def __init__(self, output_dim, merge_mode='zero', custom_position_ids=False, base=10000):
        super(SinusoidalPositionEmbedding, self).__init__()
        self.output_dim = output_dim
        self.merge_mode = merge_mode
        self.custom_position_ids = custom_position_ids
        self.base = base

    def forward(self, inputs):
        if self.custom_position_ids:
            inputs, position_ids = inputs
            batch_size = inputs.shape[0]
            position_ids = position_ids.type(torch.float)
        else:
            input_shape = inputs.shape
            batch_size, seq_len = input_shape[0], input_shape[1]
            position_ids = torch.arange(0, seq_len, dtype=torch.float)[None]

        embeddings = self.sinusoidal_embeddings(position_ids=position_ids,
                                                dim=self.output_dim,
                                                base=self.base)
        embeddings = embeddings.repeat(batch_size, 1, 1)

        if self.merge_mode == 'add':
            return inputs + embeddings.to(inputs.device)
        elif self.merge_mode == 'mul':
            return inputs * (embeddings + 1.0).to(inputs.device)
        elif self.merge_mode == 'zero':
            return embeddings.to(inputs.device)

    @staticmethod
    def sinusoidal_embeddings(position_ids, dim, base=10000):
        """计算pos位置的dim维sinusoidal编码
        tensor([[0.],
        [1.],
        [2.]])

        [5,4] seq,dim
        """
        if dim % 2 == 1:
            indices = torch.arange(0, (dim + 1) // 2, dtype=torch.float)
        else:
            indices = torch.arange(0, dim // 2, dtype=torch.float)
        indices = torch.pow(base, -2 * indices / dim)

        embeddings = torch.einsum('...,d->...d', position_ids, indices)
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        embeddings = torch.flatten(embeddings, start_dim=-2)
        if dim % 2 == 1:
            embeddings = embeddings[..., : -1]

        return embeddings
