import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from models.utils.utils import Q_ATT, H_ATT, V_Filter, GumbelSoftmax
from models.utils.modules import RvA_MODULE


class LateFusionEncoder(nn.Module):
    def __init__(self, config, vocabulary, embeddings=None):
        """

        Args:
            config:
            vocabulary:
        """
        super().__init__()

        # SIZES
        if embeddings is None:
            self.emb_size = config['emb_size']
        else:
            self.emb_size = embeddings.shape[1]
        self.lstm_size = config['lstm_size']
        self.img_size = config['img_feature_size']
        self.fusion_size = self.img_size + self.lstm_size * 2

        # VOCABULARY
        self.vocabulary = vocabulary

        # PARAMS
        self.dropout = nn.Dropout(config['dropout'])

        # LAYERS
        self.emb = nn.Embedding(len(vocabulary), self.emb_size, padding_idx=vocabulary.PAD_INDEX)
        self.hist_rnn = nn.LSTM(self.emb_size, self.lstm_size, batch_first=True)
        self.ques_rnn = nn.LSTM(self.emb_size, self.lstm_size, batch_first=True)
        self.img_lin = nn.Linear(self.img_size, self.lstm_size)
        self.attention_proj = nn.Linear(self.lstm_size, 1)
        self.fusion = nn.Linear(self.fusion_size, self.lstm_size)

        
    def forward(self, image, history, question):
        batch_size, num_rounds, _ = question.size()

        # embed questions
        ques_hidden = self.embed_question(question)

        # embed history
        hist_hidden = self.embed_history(history)

        # project down image features
        _, img_feat_size, _, _ = image.squeeze(1).size()
        image = image.view(batch_size, img_feat_size, -1).permute(0, 2, 1)
        image_features = self.img_lin(image)
    
        # repeat image feature vectors to be provided for every round
        image_features = image_features.view(batch_size, 1, -1, self.lstm_size) \
            .repeat(1, num_rounds, 1, 1).view(batch_size * num_rounds, -1, self.lstm_size)
    
        # computing attention weights
        projected_ques_features = ques_hidden.unsqueeze(1).repeat(1, image.shape[1], 1)
        projected_ques_image = (projected_ques_features * image_features)
        projected_ques_image = self.dropout(projected_ques_image)
        image_attention_weights = self.attention_proj(projected_ques_image).squeeze(-1)
        image_attention_weights = F.softmax(image_attention_weights, dim=-1)

        # multiply image features with their attention weights
        image = image.view(batch_size, 1, -1, self.img_size).repeat(1, num_rounds, 1, 1) \
            .view(batch_size * num_rounds, -1, self.img_size)
        image_attention_weights = image_attention_weights.unsqueeze(-1).repeat(1, 1, self.img_size)
        attended_image_features = (image_attention_weights * image).sum(1)
        image = attended_image_features

        # combining representations
        fused_vector = torch.cat((image, ques_hidden.squeeze(0), hist_hidden.squeeze(0)), 1)
        fused_vector = self.dropout(fused_vector)
        fused_embedding = torch.tanh(self.fusion(fused_vector))
        fused_embedding = fused_embedding.view(batch_size, num_rounds, -1)
        return fused_embedding

    def embed_question(self, question):
        """

        Args:
            question:

        Returns:

        """
        batch_size, num_rounds, _ = question.size()

        # reshaping
        question = question.view(batch_size * num_rounds, -1)

        # packing for RNN
        batch_lengths = torch.sum(torch.ne(question, self.vocabulary.PAD_INDEX), dim=1)
        ques_packed = pack_padded_sequence(self.emb(question), batch_lengths,
                                           batch_first=True, enforce_sorted=False)

        # getting last hidden state
        _, (ques_hidden, _) = self.ques_rnn(ques_packed)
        ques_hidden = ques_hidden[-1]
        return ques_hidden

    def embed_history(self, history):
        batch_size, num_rounds, _ = history.size()

        # reshaping
        history = history.view(batch_size * num_rounds, -1)

        # packing for RNN
        batch_lengths = torch.sum(torch.ne(history, self.vocabulary.PAD_INDEX), dim=1)
        hist_packed = pack_padded_sequence(self.emb(history), batch_lengths,
                                           batch_first=True, enforce_sorted=False)

        # getting last hidden state
        _, (hist_hidden, _) = self.hist_rnn(hist_packed)
        hist_hidden = hist_hidden[-1]
        return  hist_hidden


class RvAEncoder(nn.Module):
    # https://github.com/yuleiniu/rva
    def __init__(self, config, vocabulary, embeddings=None):
        """

        Args:
            config:
            vocabulary:
        """
        super().__init__()

        if embeddings is None:
            self.emb_size = config['emb_size']
        else:
            self.emb_size = embeddings.shape[1]
        self.lstm_size = config['lstm_size']
        self.img_size = config['img_feature_size']

        # VOCABULARY
        self.vocabulary = vocabulary

        # PARAMS
        self.dropout = nn.Dropout(config['dropout'])

        # LAYERS
        if embeddings is None:
            self.emb = nn.Embedding(len(vocabulary), self.emb_size, padding_idx=vocabulary.PAD_INDEX)
        else:
            self.emb = nn.Embedding.from_pretrained(embeddings, freeze=False)
        self.hist_rnn = nn.LSTM(self.emb_size, self.lstm_size, batch_first=True, bidirectional=True)
        self.ques_rnn = nn.LSTM(self.emb_size, self.lstm_size, batch_first=True, bidirectional=True)

        # self attention for question
        self.Q_ATT_ans = Q_ATT(config)
        self.Q_ATT_ref = Q_ATT(config)

        # question-based history attention
        self.H_ATT_ans = H_ATT(config)

        # modules
        self.RvA_MODULE = RvA_MODULE(config, self.emb_size)
        self.V_Filter = V_Filter(config, self.emb_size)

        # fusion layer
        self.fusion = nn.Sequential(
            nn.Dropout(p=config['dropout']),
            nn.Linear(
                self.img_size + self.emb_size + self.lstm_size * 2,
                self.lstm_size
            )
        )

        # other useful functions
        self.softmax = nn.Softmax(dim=-1)
        self.G_softmax = GumbelSoftmax()

        # initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)

    def forward(self, image, history, question):
        caption = history[:, :1, :]

        # embed questions
        ques_word_embed, ques_word_encoded, ques_not_pad, ques_encoded = self.init_q_embed(question)

        # embed history
        hist_word_embed, hist_encoded = self.init_h_embed(history)

        # embed caption
        cap_word_embed, cap_word_encoded, cap_not_pad = self.init_cap_embed(caption)

        # question features for RvA
        ques_ref_feat, ques_ref_att = self.Q_ATT_ref(ques_word_embed, ques_word_encoded, ques_not_pad)
        cap_ref_feat, _ = self.Q_ATT_ref(cap_word_embed, cap_word_encoded, cap_not_pad)

        # RvA module
        image = image.squeeze(1)
        batch_size, img_feat_size, _, _ = image.size()
        image = image.view(batch_size, img_feat_size, -1).permute(0, 2, 1)
        ques_feat = (cap_ref_feat, ques_ref_feat, ques_encoded)
        img_att, att_set = self.RvA_MODULE(image, ques_feat, hist_encoded)
        img_feat = torch.bmm(img_att, image)

        # ans_feat for joint embedding
        hist_ans_feat = self.H_ATT_ans(hist_encoded, ques_encoded)
        ques_ans_feat, ques_ans_att = self.Q_ATT_ans(ques_word_embed, ques_word_encoded, ques_not_pad)
        img_ans_feat = self.V_Filter(img_feat, ques_ans_feat)

        # combining representations
        fused_vector = torch.cat((img_ans_feat, ques_ans_feat, hist_ans_feat), -1)
        fused_embedding = torch.tanh(self.fusion(fused_vector))

        return fused_embedding

    def init_q_embed(self, question):
        batch_size, num_rounds, max_len = question.size()

        # reshaping
        question = question.view(-1, max_len)

        ques_word_embed = self.emb(question)
        ques_not_pad = torch.ne(question, self.vocabulary.PAD_INDEX)

        # packing
        batch_lengths = torch.sum(ques_not_pad, dim=1)
        ques_packed = pack_padded_sequence(ques_word_embed, batch_lengths, batch_first=True, enforce_sorted=False)

        # running through biRNN and getting all time steps
        packed_output, (ques_encoded, _) = self.ques_rnn(ques_packed)
        ques_encoded = torch.cat((ques_encoded[0], ques_encoded[1]), dim=-1)
        ques_word_encoded, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True,
                                                                      total_length=max_len)

        # reshaping
        ques_word_embed = ques_word_embed.view(-1, num_rounds, max_len, ques_word_embed.size(-1))
        ques_word_encoded = ques_word_encoded.view(-1, num_rounds, max_len, ques_word_encoded.size(-1))
        ques_not_pad = ques_not_pad.view(-1, num_rounds, max_len).float()
        ques_encoded = ques_encoded.view(batch_size, num_rounds, -1)

        return ques_word_embed, ques_word_encoded, ques_not_pad, ques_encoded


    def init_h_embed(self, history):
        batch_size, num_rounds, max_len = history.size()

        # reshaping
        history = history.view(batch_size * num_rounds, -1)

        hist_word_embed = self.emb(history)

        # packing
        batch_lengths = torch.sum(torch.ne(history, self.vocabulary.PAD_INDEX), dim=1)
        hist_packed = pack_padded_sequence(hist_word_embed, batch_lengths, batch_first=True, enforce_sorted=False)

        # getting last hidden state
        _, (hist_encoded, _) = self.hist_rnn(hist_packed)
        hist_encoded = torch.cat((hist_encoded[0], hist_encoded[1]))

        # reshaping
        hist_word_embed = hist_word_embed.view(-1, num_rounds, max_len, hist_word_embed.size(-1))
        hist_encoded = hist_encoded.view(batch_size, num_rounds, -1)

        return hist_word_embed, hist_encoded


    def init_cap_embed(self, caption):

        batch_size, _, max_len = caption.shape
        cap_word_embed = self.emb(caption.squeeze(1))

        # packing
        cap_not_pad = torch.ne(caption.view(batch_size, -1), self.vocabulary.PAD_INDEX)
        batch_lengths = torch.sum(cap_not_pad, dim=1)
        cap_packed = pack_padded_sequence(cap_word_embed, batch_lengths, batch_first=True, enforce_sorted=False)

        # getting all time steps of biRNN
        packed_output, _ = self.ques_rnn(cap_packed)

        cap_word_encoded, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True,
                                                                     total_length=max_len)

        # reshaping
        cap_word_encoded = cap_word_encoded.unsqueeze(1)
        cap_word_embed = cap_word_embed.unsqueeze(1)
        cap_not_pad = cap_not_pad.unsqueeze(1).float()

        return cap_word_embed, cap_word_encoded, cap_not_pad


class SANEncoder(nn.Module):
    # TODO include CNN for the question?
    def __init__(self, config, vocabulary, embeddings=None):
        """

        Args:
            config:
            vocabulary:
        """
        super().__init__()

        # SIZES
        if embeddings is None:
            self.emb_size = config['emb_size']
        else:
            self.emb_size = embeddings.shape[1]
        self.lstm_size = config['lstm_size']
        self.img_size = config['img_feature_size']
        self.hidden_size = config['hidden_size']

        # VOCABULARY
        self.vocabulary = vocabulary

        # PARAMS
        self.dropout = nn.Dropout(config['dropout'])

        # LAYERS

        # embedding layers
        if embeddings is None:
            self.emb = nn.Embedding(len(vocabulary), self.emb_size, padding_idx=vocabulary.PAD_INDEX)
        else:
            self.emb = nn.Embedding.from_pretrained(embeddings, freeze=False)
        self.ques_rnn = nn.LSTM(self.emb_size, self.lstm_size, batch_first=True)
        self.img_lin = nn.Linear(self.img_size, self.lstm_size)

        # first attention layer
        self.ques_fc_1 = nn.Linear(self.lstm_size, self.hidden_size)
        self.image_fc_1 = nn.Linear(self.lstm_size, self.hidden_size, bias=False)
        self.fc_att_1 = nn.Linear(self.hidden_size, 1)

        # second attention layer
        self.ques_fc_2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.image_fc_2 = nn.Linear(self.lstm_size, self.hidden_size, bias=False)
        self.fc_att_2 = nn.Linear(self.hidden_size, 1)

        # final
        self.final = nn.Linear(self.hidden_size, self.lstm_size)

        # ACTIVATIONS
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)


    def forward(self, image, history, question):
        # image dimension 1 -> squeeze?

        # Question embedding
        batch_size, num_rounds, max_sequence_length = question.size()

        question = question.view(batch_size * num_rounds, max_sequence_length)
        ques_batch_lengths = torch.sum(torch.ne(question, self.vocabulary.PAD_INDEX), dim=1)
        ques_packed = pack_padded_sequence(self.emb(question), ques_batch_lengths,
                                           batch_first=True, enforce_sorted=False)
        _, (ques_hidden, _) = self.ques_rnn(ques_packed)
        ques_hidden = ques_hidden[-1] # BxH

        # Image embedding
        _, img_feat_size, _, _ = image.squeeze(1).size()
        image = image.view(batch_size, img_feat_size, -1).transpose(1, 2)

        image_features = self.tanh(self.img_lin(image))
        image_features = self.dropout(image_features)

        # repeat image feature vectors to be provided for every round
        image_features = image_features.view(batch_size, 1, -1, self.lstm_size) \
            .repeat(1, num_rounds, 1, 1).view(batch_size * num_rounds, -1, self.lstm_size)

        # Attention 1st
        region_att_1 = self.tanh(self.image_fc_1(image_features) + self.ques_fc_1(ques_hidden).unsqueeze(1))
        region_weights_1 = self.softmax(self.fc_att_1(region_att_1))
        weighted_image_1 = (region_weights_1 * image_features).sum(1)
        query_1 = weighted_image_1 + ques_hidden

        # Attention 2nd
        region_att_2 = self.tanh(self.image_fc_2(image_features) + self.ques_fc_2(query_1).unsqueeze(1))
        region_weights_2 = self.softmax(self.fc_att_2(region_att_2))
        weighted_image_2 = (region_weights_2 * image_features).sum(1)
        query_2 = weighted_image_2 + query_1 # BN x H

        # Final representation
        fused_embedding = self.final(query_2)

        return fused_embedding
