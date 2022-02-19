import torch
from torch import nn
import tqdm

from encoder import TransformerEncoder
from decoder import TransformerDecoder


class Transformer(nn.Module):
    def __init__(self, vocab_size=30000, features_dim=512, head_num=8, text_max_len=256, device=torch.device('cpu')):
        super(Transformer, self).__init__()
        self.encoder = TransformerEncoder(
            vocab_size=vocab_size, features_dim=features_dim, head_num=head_num, text_max_len=text_max_len, device=device
        )
        self.decoder = TransformerDecoder(
            vocab_size=vocab_size, features_dim=features_dim, head_num=head_num, text_max_len=text_max_len, device=device
        )

    def forward(self, encoder_inputs, decoder_inputs, encoder_mask=None, decoder_mask=None):
        encoder_features = self.encoder(encoder_inputs, encoder_mask)
        predictive_words = self.decoder(decoder_inputs, encoder_features, decoder_mask, encoder_mask)
        return predictive_words


if __name__ == '__main__':

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'using device: {device}')
    encoder_test = torch.rand(2, 256).long().to(device)
    decoder_test = torch.rand(2, 256).long().to(device)
    model = Transformer(device=device)
    model.to(device)

    encoder_mask = torch.zeros(2, 256).to(device)
    encoder_mask[:, 0:35] = 1
    decoder_mask = torch.zeros(2, 256).to(device)
    for i in tqdm.tqdm(range(256)):
        decoder_mask[:, i] = 1
        decoder_test = model(encoder_test, decoder_test, encoder_mask, decoder_mask)
    print(decoder_test.shape)
    print(decoder_test)
