import torch
import torch.nn.functional as F
from pytorch_transformers import BertModel
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from models.modules import Q_ATT, H_ATT, V_Filter, GumbelSoftmax, RvA_MODULE


class LateFusionEncoder(nn.Module):
    def __init__(self, config, vocabulary, embeddings=None):
        """

        Args:
            config (dict): Configuration dict with network params
            vocabulary (Vocabulary): Mapping between tokens and their ids
            embeddings (torch.Tensor): pre-trained embedding vectors matching token ids
                If None, the embedding layer is initialized randomly
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
        if embeddings is None:
            self.emb = nn.Embedding(len(vocabulary), self.emb_size, padding_idx=vocabulary.PAD_INDEX)
        else:
            self.emb = nn.Embedding.from_pretrained(embeddings, freeze=True)
        self.hist_rnn = nn.LSTM(self.emb_size, self.lstm_size, batch_first=True)
        self.ques_rnn = nn.LSTM(self.emb_size, self.lstm_size, batch_first=True)
        self.img_lin = nn.Linear(self.img_size, self.lstm_size)
        self.attention_proj = nn.Linear(self.lstm_size, 1)
        self.fusion = nn.Linear(self.fusion_size, self.lstm_size)

        
    def forward(self, image, history, question):
        """

        Args:
            image: [batch x image_dim x num_regions x num_regions]
            history: [batch x max_history_len]
            question: [batch x max_question_len]

        Returns:
            final representation of the dialog [batch_size x lstm_size]

        """
        batch_size, _ = question.size() # B x Lq

        # embed questions
        ques_hidden = self.embed_question(question) # [BxH]

        # embed history
        hist_hidden = self.embed_history(history) # [BxH]

        # project down image features
        image = image.float()
        if len(image.shape) == 5:
            image = image.squeeze(1)

        if len(image.size()) == 4:
            _, img_feat_size, _, _ = image.size()
            image = image.view(batch_size, img_feat_size, -1).permute(0, 2, 1) # [BxRxI]
        else:
            _, img_feat_size, _ = image.size()
        image_features = self.img_lin(image)  # [BxRxH]

        # computing attention weights
        projected_ques_features = ques_hidden.unsqueeze(1).repeat(1, image.shape[1], 1) # [BxRxH]
        projected_ques_image = (projected_ques_features * image_features) # [BxRxH]
        projected_ques_image = self.dropout(projected_ques_image) # [BxRxH]
        image_attention_weights = self.attention_proj(projected_ques_image).squeeze(-1) # [BxR]
        image_attention_weights = F.softmax(image_attention_weights, dim=-1) # [BxR]

        # multiply image features with their attention weights
        image_attention_weights = image_attention_weights.unsqueeze(-1).repeat(1, 1, self.img_size) # [BxRxI]
        attended_image_features = (image_attention_weights * image).sum(1) # [BxI]
        image = attended_image_features

        # combining representations
        fused_vector = torch.cat((image, ques_hidden.squeeze(0), hist_hidden.squeeze(0)), 1) # [Bx I+2H]
        fused_vector = self.dropout(fused_vector) # [Bx I+2H]
        fused_embedding = torch.tanh(self.fusion(fused_vector)) # [BxH]
        return fused_embedding

    def embed_question(self, question):
        """

        Args:
            question: [batch x max_history_len]

        Returns:
            question embedding [batch x lstm_size]

        """
        batch_size, _ = question.size() # [B x Lq]

        # packing for RNN
        batch_lengths = torch.sum(torch.ne(question, self.vocabulary.PAD_INDEX), dim=1)
        ques_packed = pack_padded_sequence(self.emb(question), batch_lengths,
                                           batch_first=True, enforce_sorted=False)

        # getting last hidden state
        _, (ques_hidden, _) = self.ques_rnn(ques_packed)
        ques_hidden = ques_hidden[-1]
        return ques_hidden

    def embed_history(self, history):
        batch_size, _ = history.size() # [B x Lh]

        # packing for RNN
        batch_lengths = torch.sum(torch.ne(history, self.vocabulary.PAD_INDEX), dim=1)
        hist_packed = pack_padded_sequence(self.emb(history), batch_lengths,
                                           batch_first=True, enforce_sorted=False)

        # getting last hidden state
        _, (hist_hidden, _) = self.hist_rnn(hist_packed)
        hist_hidden = hist_hidden[-1]
        return  hist_hidden


class RvAEncoder(nn.Module):
    def __init__(self, config, vocabulary, embeddings=None):
        """

        Args:
            config (dict): Configuration dict with network params
            vocabulary (Vocabulary): Mapping between tokens and their ids
            embeddings (torch.Tensor): pre-trained embedding vectors matching token ids
                If None, the embedding layer is initialized randomly
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
            self.emb = nn.Embedding.from_pretrained(embeddings, freeze=True)
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

    def forward(self, image, history, question, caption, turn):
        # caption = history[:, :1, :]
        batch_size = history.shape[0]

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
        image = image.float()
        if len(image.size()) == 4:
            _, img_feat_size, _, _ = image.size()
            image = image.view(batch_size, img_feat_size, -1).permute(0, 2, 1) # [BxRxI]
        else:
            _, img_feat_size, _ = image.size()
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

        output_vector = torch.stack([fused_embedding[i, turn[i], :] for i in range(batch_size)])

        return output_vector

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
        ques_word_embed = ques_word_embed.view(-1, num_rounds,  max_len, ques_word_embed.size(-1))
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

        batch_size, max_len = caption.shape
        cap_word_embed = self.emb(caption)

        # packing
        cap_not_pad = torch.ne(caption, self.vocabulary.PAD_INDEX)
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
    def __init__(self, config, vocabulary, embeddings=None, use_bert=False, bert_path=None):
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
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.bert.to(device)
            self.lstm_size = self.bert.config.hidden_size

        self.img_size = config['img_feature_size']
        self.hidden_size = config['hidden_size']

        # VOCABULARY
        self.vocabulary = vocabulary

        # PARAMS
        self.dropout = nn.Dropout(config['dropout'])

        # LAYERS

        # embedding layers
        if not self.use_bert:
            if embeddings is None:
                self.emb = nn.Embedding(len(vocabulary), self.emb_size, padding_idx=vocabulary.PAD_INDEX)
            else:
                self.emb = nn.Embedding.from_pretrained(embeddings, freeze=True)

            self.ques_rnn = nn.LSTM(self.emb_size, self.lstm_size, batch_first=True)
            self.hist_rnn = nn.LSTM(self.emb_size, self.lstm_size, batch_first=True)
            # self.cap_rnn = nn.LSTM(self.emb_size, self.lstm_size, batch_first=True)

        self.img_lin = nn.Linear(self.img_size, self.lstm_size)

        # first attention layer
        self.ques_fc_1 = nn.Linear(self.lstm_size, self.hidden_size)
        self.hist_fc_1 = nn.Linear(self.lstm_size, self.hidden_size)
        # self.cap_fc_1 = nn.Linear(self.lstm_size, self.hidden_size)
        self.image_fc_1 = nn.Linear(self.lstm_size, self.hidden_size, bias=False)
        self.fc_att_1 = nn.Linear(self.hidden_size, 1)

        # second attention layer
        self.ques_fc_2 = nn.Linear(self.lstm_size, self.hidden_size)
        self.image_fc_2 = nn.Linear(self.lstm_size, self.hidden_size, bias=False)
        self.fc_att_2 = nn.Linear(self.hidden_size, 1)

        # final
        self.final = nn.Linear(self.lstm_size, self.lstm_size)

        # ACTIVATIONS
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)


    def forward(self, image, history, question, caption):

        batch_size = history.shape[0]

        ques_hidden = self.init_ques_embedding(question)
        hist_hidden = self.init_hist_embedding(history)
        # cap_hidden = self.init_cap_embedding(caption)

        # Image embedding
        image = image.float()
        if len(image.size()) == 4:
            _, img_feat_size, _, _ = image.size()
            image = image.view(batch_size, img_feat_size, -1).permute(0, 2, 1) # [BxRxI]
        else:
            _, img_feat_size, _ = image.size()

        image_features = self.tanh(self.img_lin(image)) # [BxRxH]
        image_features = self.dropout(image_features) # [BxRxH]

        # Attention 1st
        region_att_1 = self.tanh(self.image_fc_1(image_features)  +
                                 self.ques_fc_1(ques_hidden).unsqueeze(1) +
                                 self.hist_fc_1(hist_hidden).unsqueeze(1)) # [BxRxH']
        region_weights_1 = self.softmax(self.fc_att_1(region_att_1)) # [BxRx1]
        weighted_image_1 = (region_weights_1 * image_features).sum(1) # [BxH]
        query_1 = weighted_image_1 + ques_hidden # [BxH]

        # Attention 2nd
        region_att_2 = self.tanh(self.image_fc_2(image_features) + self.ques_fc_2(query_1).unsqueeze(1)) # [BxRxH']
        region_weights_2 = self.softmax(self.fc_att_2(region_att_2)) # [BxRx1]
        weighted_image_2 = (region_weights_2 * image_features).sum(1) # [BxH]
        query_2 = weighted_image_2 + query_1 #  [BxH]

        # Final representation
        fused_embedding = self.final(query_2)

        return fused_embedding

    def init_ques_embedding(self, question):
        if not self.use_bert:
            ques_batch_lengths = torch.sum(torch.ne(question, self.vocabulary.PAD_INDEX), dim=1)
            ques_packed = pack_padded_sequence(self.emb(question), ques_batch_lengths,
                                               batch_first=True, enforce_sorted=False)
            _, (ques_hidden, _) = self.ques_rnn(ques_packed)
            ques_hidden = ques_hidden[-1]  # BxH

        else:
            ques_hidden, _ = self.bert(question)
            ques_hidden  = torch.mean(ques_hidden, dim=1)
        return ques_hidden

    def init_hist_embedding(self, history):
        if not self.use_bert:
            batch_lengths = torch.sum(torch.ne(history, self.vocabulary.PAD_INDEX), dim=1)
            hist_packed = pack_padded_sequence(self.emb(history), batch_lengths,
                                               batch_first=True, enforce_sorted=False)

            _, (hist_hidden, _) = self.hist_rnn(hist_packed)
            hist_hidden = hist_hidden[-1]
        else:
            hist_hidden, _ = self.bert(history)
            hist_hidden = torch.mean(hist_hidden, dim=1)
        return hist_hidden

    def init_cap_embedding(self, caption):
        if not self.use_bert:
            cap_batch_lengths = torch.sum(torch.ne(caption, self.vocabulary.PAD_INDEX), dim=1)
            cap_packed = pack_padded_sequence(self.emb(caption), cap_batch_lengths,
                                               batch_first=True, enforce_sorted=False)

            _, (cap_hidden, _) = self.cap_rnn(cap_packed)
            cap_hidden = cap_hidden[-1]
        else:
            cap_hidden, _ = self.bert(caption)
            cap_hidden = torch.mean(cap_hidden, dim=1)
        return cap_hidden