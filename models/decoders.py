import torch
from pytorch_transformers import BertModel
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


class DiscDecoder(nn.Module):
    def __init__(self, config, vocabulary, embeddings, use_bert, bert_path):
        """

        Args:
            config (dict): Configuration dict with network params
            vocabulary (Vocabulary): Mapping between tokens and their ids
            embeddings (torch.Tensor): pre-trained embedding vectors matching token ids
                If None, the embedding layer is initialized randomly
            use_bert (bool): If true, pre-trained BERT is loaded from `bert_path`
            bert_path (str): Path to pre-trained binary BERT model
        """
        super().__init__()

        # SIZES
        self.use_bert = use_bert

        # SIZES
        if not self.use_bert:
            if embeddings is None:
                self.emb_size = config['emb_size']
            else:
                self.emb_size = embeddings.shape[1]
            self.lstm_size = config['lstm_size']
        else:
            model_state_dict = torch.load(bert_path)
            self.bert = BertModel.from_pretrained('bert-base-cased', state_dict=model_state_dict)
            self.lstm_size = self.bert.config.hidden_size
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.bert.to(device)

        self.img_size = config["img_feature_size"]

        # PARAMS
        self.dropout = nn.Dropout(config['dropout'])
        self.vocabulary = vocabulary

        # LAYERS
        if not self.use_bert:
            if embeddings is None:
                self.emb = nn.Embedding(len(vocabulary), self.emb_size, padding_idx=vocabulary.PAD_INDEX)
            else:
                self.emb = nn.Embedding.from_pretrained(embeddings, freeze=True)
            self.option_rnn = nn.LSTM(self.emb_size, self.lstm_size, batch_first=True)


    def forward(self, encoder_output, options):

        batch_size, num_options, max_sequence_length = options.size()
        options = options.view(batch_size * num_options, max_sequence_length) # [BN x Lopt]

        opt_hidden = self.init_opt_embedding(options)

        encoder_output = encoder_output.unsqueeze(2).repeat(1, 1, num_options, 1) # [1xBxHNx1]
        encoder_output = encoder_output.view(batch_size * num_options, self.lstm_size) # [BNxH]

        scores = torch.sum(opt_hidden.squeeze(0) * encoder_output, 1) # [BNx1]
        scores = scores.view(batch_size, num_options) # [BxN]
        return scores

    def init_opt_embedding(self, options):

        if not self.use_bert:
            options_batch_lengths = torch.sum(torch.ne(options, self.vocabulary.PAD_INDEX), dim=1)  # [BNx1]
            options_packed = pack_padded_sequence(self.emb(options), options_batch_lengths,
                                                  batch_first=True, enforce_sorted=False)

            _, (opt_hidden, _) = self.option_rnn(options_packed)
            opt_hidden = opt_hidden[-1]
        else:
            opt_hidden, _ = self.bert(options)
            opt_hidden = torch.mean(opt_hidden, dim=1)
        return opt_hidden

