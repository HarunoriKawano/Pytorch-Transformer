import torch
from torch import nn

from module import MultiHeadAttention, FeedForward, PositionalEncoder, Embedder


class TransformerEncoder(nn.Module):
    """Encoder module of Transformer

    embedding -> positional_encoder -> transformer encoder block × 6

    Attributes:
        embedder (Embedder): Word embedding module
        positional_encoder (PositionalEncoder): Module for receiving position vectors for words
        transformer1 ~ transformer6 (TransformerEncoderBlock): self-attention and feedforward module

    """
    def __init__(self, text_embedding_vectors, features_dim=512, head_num=8, text_max_len=256, device=torch.device('cpu')):
        super(TransformerEncoder, self).__init__()
        self.embedder = Embedder(text_embedding_vectors)
        self.positional_encoder = PositionalEncoder(features_dim=features_dim, text_max_len=text_max_len, device=device)
        self.transformer1 = TransformerEncoderBlock(features_dim=features_dim, head_num=head_num)
        self.transformer2 = TransformerEncoderBlock(features_dim=features_dim, head_num=head_num)
        self.transformer3 = TransformerEncoderBlock(features_dim=features_dim, head_num=head_num)
        self.transformer4 = TransformerEncoderBlock(features_dim=features_dim, head_num=head_num)
        self.transformer5 = TransformerEncoderBlock(features_dim=features_dim, head_num=head_num)
        self.transformer6 = TransformerEncoderBlock(features_dim=features_dim, head_num=head_num)

    def forward(self, words, mask):
        word_vector = self.embedder(words)

        positional_word_vector = self.positional_encoder(word_vector)

        features = self.transformer1(positional_word_vector, mask)
        features = self.transformer2(features, mask)
        features = self.transformer3(features, mask)
        features = self.transformer4(features, mask)
        features = self.transformer5(features, mask)
        features = self.transformer6(features, mask)

        return features


class TransformerEncoderBlock(nn.Module):
    """Encoder module of Transformer

    Perform self-attention and feedforward processing on the input features.

    Attributes:
        norm1 (torch.nn.LayerNorm): Layer normalization before self-attention
        norm2 (torch.nn.LayerNorm): Layer normalization before feedforward
        attention (MultiHeadAttention): self-attention module
        feedforward (FeedForward): feedforward module
        dropout1 (torch.nn.Dropout): Dropout after self-attention
        dropout2 (torch.nn.Dropout): Dropout after feedforward

    """

    def __init__(self, features_dim=512, head_num=8, dropout_rate=0.1):
        super(TransformerEncoderBlock, self).__init__()
        self.norm1 = nn.LayerNorm(features_dim)
        self.norm2 = nn.LayerNorm(features_dim)

        self.attention = MultiHeadAttention(features_dim=features_dim, head_num=head_num)
        self.feedforward = FeedForward(features_dim=features_dim)

        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, inputs, mask=None):
        """self-attention -> layer normalization -> dropout -> feedforward -> layer normalization -> dropout

        Args:
            inputs (torch.Tensor(batch_num, data_len, feature_num)): features of encoder
            mask (torch.Tensor(batch_num, data_len)): Mask for features not considered.

        Returns:
            torch.Tensor(batch_num, data_len, feature_num): features of encoder
        """
        q = k = v = inputs
        attention = self.attention(q, k, v, mask)

        out = inputs + self.dropout1(self.norm1(attention))

        feedforward = self.feedforward(out)
        out = out + self.dropout2(self.norm2(feedforward))

        return out
