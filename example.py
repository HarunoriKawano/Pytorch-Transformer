import torch
from torchviz import make_dot

from models.transformer import Transformer


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    batch_num = 8
    encoder_vocab_size = 40000
    decoder_vocab_size = 30000
    text_max_len = 256

    model = Transformer(
        encoder_vocab_size=encoder_vocab_size, decoder_vocab_size=decoder_vocab_size,
        text_max_len=text_max_len, device=device
    )
    model.to(device)

    # Text ids to be translated
    encoder_ids = torch.rand(batch_num, text_max_len).long()

    # First input for decoding
    decoder_ids = torch.zeros(batch_num, text_max_len).long()

    # If the translation target is all 35 characters
    encoder_mask = torch.zeros(batch_num, text_max_len).long()
    encoder_mask[:, :35] = 1

    decoder_mask = torch.zeros(batch_num, text_max_len).long()

    # Predicted value including the next character
    decoder_out = model(encoder_ids, decoder_ids, encoder_mask, decoder_mask)

    graph = make_dot(decoder_out, params=dict(model.named_parameters()))
    graph.render("Transformer")

    # (batch_num, text_max_len, decoder_vocab_size)
    print(f'decoder out shape: {decoder_out.shape}')
