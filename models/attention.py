import math
import torch
from torch import nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, attention_dim=512, head_num=8, dropout_rate=0.1):
        super(MultiHeadAttention, self).__init__()
        self.q_linear = nn.Linear(attention_dim, attention_dim)
        self.v_linear = nn.Linear(attention_dim, attention_dim)
        self.k_linear = nn.Linear(attention_dim, attention_dim)
        self.dropout = nn.Dropout(dropout_rate)

        self.out_linear = nn.Linear(attention_dim, attention_dim)
        self.head_num = head_num
        self.attention_dim = attention_dim
        self.head_dim = attention_dim // head_num

    def forward(self, q, k, v, mask):
        q = self._split_head(self.q_linear(q))
        k = self._split_head(self.k_linear(k))
        v = self._split_head(self.v_linear(v))

        q *= self.head_dim ** -0.5

        weights = torch.matmul(q, torch.transpose(k, 2, 3)) / math.sqrt(self.head_dim)

        mask = mask.unsqueeze(1).unsqueeze(1)
        weights = weights.masked_fill(mask == 0, -1e9)
        weights = self.dropout(weights)

        normalized_weights = F.softmax(weights, dim=-1)

        output = self._combine_head(torch.matmul(normalized_weights, v))
        return self.out_linear(output)

    def _split_head(self, inputs):
        batch_size, length, _ = inputs.shape
        split_inputs = torch.reshape(inputs, (batch_size, length, self.head_num, self.head_dim))
        return torch.transpose(split_inputs, 1, 2)

    def _combine_head(self, inputs):
        batch_size, _, length, _ = inputs.shape
        combine_inputs = torch.reshape(torch.transpose(inputs, 1, 2), (batch_size, length, self.attention_dim))
        return combine_inputs


if __name__ == '__main__':
    model = MultiHeadAttention()
    tensor = torch.rand(2, 10, 512)
    mask = torch.ones(2, 10)
    out = model(tensor, tensor, tensor, mask)
