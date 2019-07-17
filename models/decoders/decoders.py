import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


class DiscDecoder(nn.Module):
    def __init__(self, config, vocabulary, embeddings):
        super().__init__()

        # SIZES
        if embeddings is None:
            self.emb_size = config['emb_size']
        else:
            self.emb_size = embeddings.shape[1]
            
        self.img_size = config["img_feature_size"]
        self.lstm_size = config['lstm_size']

        # PARAMS
        self.dropout = nn.Dropout(config['dropout'])
        self.vocabulary = vocabulary

        # LAYERS
        if embeddings is None:
            self.emb = nn.Embedding(len(vocabulary), self.emb_size, padding_idx=vocabulary.PAD_INDEX)
        else:
            self.emb = nn.Embedding.from_pretrained(embeddings)
        self.option_rnn = nn.LSTM(self.emb_size, self.lstm_size, batch_first=True)


    def forward(self, encoder_output, options):
        batch_size, num_rounds, num_options, max_sequence_length = options.size()
        options = options.view(batch_size * num_rounds * num_options, max_sequence_length)

        options_batch_lengths = torch.sum(torch.ne(options, self.vocabulary.PAD_INDEX), dim=1)
        options_packed = pack_padded_sequence(self.emb(options), options_batch_lengths,
                                              batch_first=True, enforce_sorted=False)

        _, (opt_hidden, _) = self.option_rnn(options_packed)
        opt_hidden = opt_hidden[-1]

        encoder_output = encoder_output.unsqueeze(2).repeat(1, 1, num_options, 1)
        encoder_output = encoder_output.view(batch_size * num_rounds * num_options, self.lstm_size)

        scores = torch.sum(opt_hidden.squeeze(0) * encoder_output, 1)
        scores = scores.view(batch_size, num_rounds, num_options)
        return scores



