import math
import torch
from torch import nn
import torch.nn.functional as F


class Embedder(nn.Module):
    def __init__(self, text_embedding_vectors):
        super(Embedder, self).__init__()
        self.embeddings = nn.Embedding.from_pretrained(
            embeddings=text_embedding_vectors, freeze=True
        )

    def forward(self, words):
        word_vector = self.embeddings(words)
        return word_vector


class PositionalEncoder(nn.Module):
    """Embedding position vectors in word vectors

    Attributes:
        features_dim (int): Number of features in word vector
        pe (torch.Tensor(1, max_len, features_dim)): position vectors
    """

    def __init__(self, features_dim=512, text_max_len=256, device=torch.device('cpu')):
        """
        Args:
            features_dim (int): Number of features in word vector
            text_max_len (int): The maximum length of text
            device (torch.device): using device(cuda or cpu)
        """
        super(PositionalEncoder, self).__init__()
        self.features_dim = features_dim

        pe = torch.zeros(text_max_len, features_dim)
        pe = pe.to(device)

        # Position vector generation
        for pos in range(text_max_len):
            for i in range(0, features_dim, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/features_dim)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1))/features_dim)))

        self.pe = pe.unsqueeze(0)
        self.pe.requires_grad = False

    def forward(self, inputs):
        ret = math.sqrt(self.features_dim)*inputs + self.pe
        return ret


class FeedForward(nn.Module):
    """Simple full connected module

    Attributes:
        linear1 (torch.nn.Linear): The first layer
        dropout (torch.nn.Dropout): Dropout layer
        linear2 (torch.nn.Linear): The second layer
    """

    def __init__(self, features_dim=512, hidden_dim=1024, dropout_rate=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(features_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.linear2 = nn.Linear(hidden_dim, features_dim)

    def forward(self, inputs):
        out = self.linear1(inputs)
        out = self.dropout(F.relu(out))
        out = self.linear2(out)
        return out


class MultiHeadAttention(nn.Module):
    """multi-head attention module

    Performs multi-headed attention processing for q, k, and v specified in the argument.

    Attributes:
        q_linear (torch.nn.Linear): Full connected layer for argument q
        v_linear (torch.nn.Linear): Full connected layer for argument v
        k_linear (torch.nn.Linear): Full connected layer for argument k
        dropout (torch.nn.Dropout): Dropout layer for weights of attention processing
        out_linear (torch.nn.Linear): Full connected layer for output
        head_num (int): Number of multi-head
        features_dim (int): Number of dimension of　q, v, and k features
        head_dim (int): Number of dimension of a single head

    """
    def __init__(self, features_dim=512, head_num=8, dropout_rate=0.1):
        super(MultiHeadAttention, self).__init__()
        self.q_linear = nn.Linear(features_dim, features_dim)
        self.v_linear = nn.Linear(features_dim, features_dim)
        self.k_linear = nn.Linear(features_dim, features_dim)
        self.dropout = nn.Dropout(dropout_rate)

        self.out_linear = nn.Linear(features_dim, features_dim)
        self.head_num = head_num
        self.attention_dim = features_dim
        self.head_dim = features_dim // head_num

    def forward(self, q, k, v, mask=None):
        """multi-head attention processing

        Calculate the similarity of the arguments q and k, and multiply them by argument v.
        In normal-attention, the values of k and v will be the same, and in self-attention, q, k, and v will all be the same.
        Args:
            q (torch.Tensor(batch_num, data_len, feature_num)): Query to be compared for similarity.
            k (torch.Tensor(batch_num, data_len, feature_num)): Key for calculating the similarity.
            v (torch.Tensor(batch_num, data_len, feature_num)): Value to be multiplied by similarity.
            mask (torch.Tensor(batch_num, data_len)): Mask for features not considered. Classify by 0 and 1.

        Returns:
            torch.Tensor(batch_num, data_len, feature_num): Value through the attention process.
        """
        q = self._split_head(self.q_linear(q))
        k = self._split_head(self.k_linear(k))
        v = self._split_head(self.v_linear(v))

        # Scaling with multi-heads in mind
        q *= self.head_dim ** -0.5

        weights = torch.matmul(q, torch.transpose(k, 2, 3)) / math.sqrt(self.head_dim)

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(3)
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
