import torch
from torch import nn
import torch.nn.functional as F

from utils import MultiHeadAttention, FeedForward, PositionalEncoder, Embedder


class TransformerDecoder(nn.Module):
    """Decoder module of Transformer

    embedding -> positional_encoder -> transformer decoder block × 6 -> linear

    Attributes:
        embedder (Embedder): Word embedding module
        positional_encoder (PositionalEncoder): Module for receiving position vectors for words
        transformer1 ~ transformer6 (TransformerDecoderBlock): self-attention, encoder attention and feedforward module

    """

    def __init__(self, vocab_size=30000, features_dim=512, head_num=8, text_max_len=256, device=torch.device('cpu')):
        super(TransformerDecoder, self).__init__()
        self.embedder = Embedder(vocab_size=vocab_size, features_dim=features_dim)
        self.positional_encoder = PositionalEncoder(features_dim=features_dim, text_max_len=text_max_len, device=device)
        self.transformer1 = TransformerDecoderBlock(features_dim=features_dim, head_num=head_num)
        self.transformer2 = TransformerDecoderBlock(features_dim=features_dim, head_num=head_num)
        self.transformer3 = TransformerDecoderBlock(features_dim=features_dim, head_num=head_num)
        self.transformer4 = TransformerDecoderBlock(features_dim=features_dim, head_num=head_num)
        self.transformer5 = TransformerDecoderBlock(features_dim=features_dim, head_num=head_num)
        self.transformer6 = TransformerDecoderBlock(features_dim=features_dim, head_num=head_num)
        self.linear = nn.Linear(features_dim, vocab_size)

    def forward(self, words, encoder_out, decoder_mask, encoder_mask):
        """
        Args:
            words (torch.LongTensor(batch_num, data_len)): Decoder word sequence of a sentence. Output value of the previous deocder.
            encoder_out (torch.FloatTensor(batch_num, data_len, feature_num)): features of encoder
            encoder_mask (torch.LongTensor(batch_num, data_len)(0 or 1)): mask for encoder
            decoder_mask (torch.LongTensor(batch_num, data_len)(0 or 1)): mask for decoder

        Returns:
            torch.LongTensor(batch_num, data_len): Predicted value for the next value.
        """
        word_vector = self.embedder(words)

        positional_word_vector = self.positional_encoder(word_vector)

        features = self.transformer1(
            inputs=positional_word_vector, encoder_out=encoder_out,
            decoder_mask=decoder_mask, encoder_mask=encoder_mask
        )
        features = self.transformer2(
            inputs=features, encoder_out=encoder_out,
            decoder_mask=decoder_mask, encoder_mask=encoder_mask
        )
        features = self.transformer3(
            inputs=features, encoder_out=encoder_out,
            decoder_mask=decoder_mask, encoder_mask=encoder_mask
        )
        features = self.transformer4(
            inputs=features, encoder_out=encoder_out,
            decoder_mask=decoder_mask, encoder_mask=encoder_mask
        )
        features = self.transformer5(
            inputs=features, encoder_out=encoder_out,
            decoder_mask=decoder_mask, encoder_mask=encoder_mask
        )
        features = self.transformer6(
            inputs=features, encoder_out=encoder_out,
            decoder_mask=decoder_mask, encoder_mask=encoder_mask
        )
        out = F.softmax(self.linear(features), dim=2)

        return torch.argmax(out, dim=2)


class TransformerDecoderBlock(nn.Module):
    """Decoder module of Transformer

    Perform self-attention, encoder attention and feedforward processing on the input features.

    Attributes:
        masked_attention (MultiHeadAttention): self-attention module with mask
        attention (MultiHeadAttention): encoder attention module
        feedforward (FeedForward): feedforward module
        norm1 (torch.nn.LayerNorm): Layer normalization after masked_attention
        norm2 (torch.nn.LayerNorm): Layer normalization after encoder attention
        norm3 (torch.nn.LayerNorm): Layer normalization after feedforward
        dropout1 (torch.nn.Dropout): Dropout after masked_attention
        dropout2 (torch.nn.Dropout): Dropout after encoder attention
        dropout3 (torch.nn.Dropout): Dropout after feedforward

    """
    def __init__(self, features_dim=512, head_num=8, dropout_rate=0.1):
        super(TransformerDecoderBlock, self).__init__()
        self.masked_attention = MultiHeadAttention(features_dim=features_dim, head_num=head_num)
        self.attention = MultiHeadAttention(features_dim=features_dim, head_num=head_num)
        self.feedforward = FeedForward(features_dim=features_dim)

        self.norm1 = nn.LayerNorm(features_dim)
        self.norm2 = nn.LayerNorm(features_dim)
        self.norm3 = nn.LayerNorm(features_dim)

        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.dropout3 = nn.Dropout(dropout_rate)

    def forward(self, inputs, encoder_out, encoder_mask=None, decoder_mask=None):
        """self-attention -> layer normalization -> dropout -> feedforward -> layer normalization -> dropout

                Args:
                    inputs (torch.FloatTensor(batch_num, data_len, feature_num)): features of decoder
                    encoder_out (torch.FloatTensor(batch_num, data_len, feature_num)): features of encoder
                    encoder_mask (torch.LongTensor(batch_num, data_len)(0 or 1)): Mask for features not considered.
                    decoder_mask (torch.LongTensor(batch_num, data_len)(0 or 1)): Mask for features not considered.

                Returns:
                    torch.FloatTensor(batch_num, data_len, feature_num): features of decoder
                """
        q = k = v = inputs
        masked_attention = self.masked_attention(q, k, v, decoder_mask)

        out = inputs + self.dropout1(self.norm1(masked_attention))

        k = v = encoder_out
        attention = self.attention(out, k, v, encoder_mask)

        out = inputs + self.dropout2(self.norm2(attention))

        feedforward = self.feedforward(out)
        out = out + self.dropout3(self.norm3(feedforward))

        return out
