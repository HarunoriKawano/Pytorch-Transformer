import torch
from torch import nn

from models.encoder import TransformerEncoder
from models.decoder import TransformerDecoder


class Transformer(nn.Module):
    """Transformer model for translation

    encoder <- original_text_ids(x0...xi), decoder <- translate_text_ids(0x..xj)  => output -> translate_text_ids(0x...x(j+1))

    Attributes:
        encoder (TransformerEncoder): encoder module of Transformer
        decoder (TransformerDecoder): decoder module of Transformer
    """
    def __init__(self, encoder_vocab_size=30000, decoder_vocab_size=30000, features_dim=512, head_num=8, text_max_len=256, device=torch.device('cpu')):
        super(Transformer, self).__init__()
        print(f"using device: {device}")
        self.text_max_len = text_max_len
        self.device = device
        self.encoder = TransformerEncoder(
            vocab_size=encoder_vocab_size, features_dim=features_dim, head_num=head_num, text_max_len=text_max_len, device=device
        )
        self.decoder = TransformerDecoder(
            vocab_size=decoder_vocab_size, features_dim=features_dim, head_num=head_num, text_max_len=text_max_len, device=device
        )

    def forward(self, encoder_inputs, decoder_inputs, encoder_mask=None, decoder_mask=None):
        encoder_inputs, decoder_inputs = encoder_inputs.to(self.device), decoder_inputs.to(self.device)
        if encoder_mask is not None:
            encoder_mask = self._encoder_mask_reshape(encoder_mask.to(self.device))
        if decoder_mask is not None:
            decoder_mask = self._decoder_mask_reshape(decoder_mask.to(self.device))
        encoder_features = self.encoder(encoder_inputs, encoder_mask)
        predictive_words = self.decoder(decoder_inputs, encoder_features, decoder_mask, encoder_mask)
        return predictive_words

    def _encoder_mask_reshape(self, encoder_mask):
        reshaped_encoder_mask = encoder_mask.view(-1, 1, 1, self.text_max_len)
        return reshaped_encoder_mask

    def _decoder_mask_reshape(self, decoder_mask):
        batch_size, _ = decoder_mask.shape
        reshaped_decoder_mask = decoder_mask.view(-1, 1, 1, self.text_max_len)
        subsequence_mask = torch.tril(torch.ones(batch_size, 1, self.text_max_len, self.text_max_len).to(self.device))
        return torch.logical_and(reshaped_decoder_mask, subsequence_mask)
