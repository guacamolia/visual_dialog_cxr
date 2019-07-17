import pickle

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, confusion_matrix
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from tqdm import tqdm

class GumbelSoftmax(nn.Module):
    def __init__(self, temperature=1, hard=False):
        super(GumbelSoftmax, self).__init__()
        self.hard = hard
        self.gpu = False

        self.temperature = temperature

    def cuda(self):
        self.gpu = True

    def cpu(self):
        self.gpu = False

    def sample_gumbel(self, shape, eps=1e-10):
        """Sample from Gumbel(0, 1)"""
        noise = torch.rand(shape)
        noise.add_(eps).log_().neg_()
        noise.add_(eps).log_().neg_()
        if self.gpu:
            return noise.detach().cuda()
        else:
            return noise.detach()

    def sample_gumbel_like(self, template_tensor, eps=1e-10):
        uniform_samples_tensor = template_tensor.clone().uniform_()
        gumble_samples_tensor = - torch.log(eps - torch.log(uniform_samples_tensor + eps))
        return gumble_samples_tensor

    def gumbel_softmax_sample(self, logits, temperature):
        """ Draw a sample from the Gumbel-Softmax distribution"""
        dim = logits.size(-1)
        gumble_samples_tensor = self.sample_gumbel_like(logits.detach())  # 0.4
        gumble_trick_log_prob_samples = logits + gumble_samples_tensor.detach()
        soft_samples = F.softmax(gumble_trick_log_prob_samples / temperature, -1)
        return soft_samples

    def gumbel_softmax(self, logits, temperature):
        """Sample from the Gumbel-Softmax distribution and optionally discretize.
        Args:
        logits: [batch_size, n_class] unnormalized log-probs
        temperature: non-negative scalar
        Returns:
        [batch_size, n_class] sample from the Gumbel-Softmax distribution.
        If hard=True, then the returned sample will be one-hot, otherwise it will
        be a probabilitiy distribution that sums to 1 across classes
        """
        if self.training:
            y = self.gumbel_softmax_sample(logits, temperature)
            _, max_value_indexes = y.detach().max(1, keepdim=True)
            y_hard = logits.detach().clone().zero_().scatter_(1, max_value_indexes, 1)
            y = y_hard - y.detach() + y
        else:
            _, max_value_indexes = logits.detach().max(1, keepdim=True)
            y = logits.detach().clone().zero_().scatter_(1, max_value_indexes, 1)
        return y

    def forward(self, logits, temperature=None):
        samplesize = logits.size()

        if temperature == None:
            temperature = self.temperature

        return self.gumbel_softmax(logits, temperature=temperature)


class GatedTrans(nn.Module):
    """docstring for GatedTrans"""

    def __init__(self, in_dim, out_dim):
        super(GatedTrans, self).__init__()

        self.embed_y = nn.Sequential(
            nn.Linear(
                in_dim,
                out_dim
            ),
            nn.Tanh()
        )
        self.embed_g = nn.Sequential(
            nn.Linear(
                in_dim,
                out_dim
            ),
            nn.Sigmoid()
        )

    def forward(self, x_in):
        x_y = self.embed_y(x_in)
        x_g = self.embed_g(x_in)
        x_out = x_y * x_g

        return x_out


class Q_ATT(nn.Module):
    """Self attention module of questions."""

    def __init__(self, config):
        super(Q_ATT, self).__init__()

        self.embed = nn.Sequential(
            nn.Dropout(p=config["dropout_fc"]),
            GatedTrans(
                config["lstm_size"] * 2,
                config["lstm_size"]
            ),
        )
        self.att = nn.Sequential(
            nn.Dropout(p=config["dropout_fc"]),
            nn.Linear(
                config["lstm_size"],
                1
            )
        )
        self.softmax = nn.Softmax(dim=-1)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)

    def forward(self, ques_word, ques_word_encoded, ques_not_pad):
        # ques_word shape: (batch_size, num_rounds, quen_len_max, word_embed_dim) YES
        # ques_embed shape: (batch_size, num_rounds, quen_len_max, lstm_hidden_size * 2) ques_word_encoded???
        # ques_not_pad shape: (batch_size, num_rounds, quen_len_max) YES
        # output: img_att (batch_size, num_rounds, embed_dim)
        batch_size = ques_word.size(0)
        num_rounds = ques_word.size(1)
        quen_len_max = ques_word.size(2)

        ques_embed = self.embed(ques_word_encoded)  # shape: (batch_size, num_rounds, quen_len_max, embed_dim)
        ques_norm = F.normalize(ques_embed, p=2, dim=-1)  # shape: (batch_size, num_rounds, quen_len_max, embed_dim) 

        att = self.att(ques_norm).squeeze(-1)  # shape: (batch_size, num_rounds, quen_len_max)
        # ignore <pad> word
        att = self.softmax(att)
        att = att * ques_not_pad  # shape: (batch_size, num_rounds, quen_len_max)
        att = att / torch.sum(att, dim=-1, keepdim=True)  # shape: (batch_size, num_rounds, quen_len_max)
        feat = torch.sum(att.unsqueeze(-1) * ques_word, dim=-2)  # shape: (batch_size, num_rounds, rnn_dim)

        return feat, att


class H_ATT(nn.Module):
    """question-based history attention"""

    def __init__(self, config):
        super(H_ATT, self).__init__()

        self.H_embed = nn.Sequential(
            nn.Dropout(p=config["dropout_fc"]),
            GatedTrans(
                config["lstm_size"] * 2,
                config["lstm_size"]
            ),
        )
        self.Q_embed = nn.Sequential(
            nn.Dropout(p=config["dropout_fc"]),
            GatedTrans(
                config["lstm_size"] * 2,
                config["lstm_size"]
            ),
        )
        self.att = nn.Sequential(
            nn.Dropout(p=config["dropout_fc"]),
            nn.Linear(
                config["lstm_size"],
                1
            )
        )
        self.softmax = nn.Softmax(dim=-1)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)

    def forward(self, hist, ques):
        # hist shape: (batch_size, num_rounds, rnn_dim)
        # ques shape: (batch_size, num_rounds, rnn_dim)
        # output: hist_att (batch_size, num_rounds, embed_dim)
        batch_size = ques.size(0)
        num_rounds = ques.size(1)

        hist_embed = self.H_embed(hist)  # shape: (batch_size, num_rounds, embed_dim)
        hist_embed = hist_embed.unsqueeze(1).repeat(1, num_rounds, 1,
                                                    1)  # shape: (batch_size, num_rounds, num_rounds, embed_dim)

        ques_embed = self.Q_embed(ques)  # shape: (batch_size, num_rounds, embed_dim)
        ques_embed = ques_embed.unsqueeze(2).repeat(1, 1, num_rounds,
                                                    1)  # shape: (batch_size, num_rounds, num_rounds, embed_dim)

        att_embed = F.normalize(hist_embed * ques_embed, p=2, dim=-1)  # (batch_size, num_rounds, num_rounds, embed_dim)
        att_embed = self.att(att_embed).squeeze(-1)
        att = self.softmax(att_embed)  # shape: (batch_size, num_rounds, num_rounds)
        att_not_pad = torch.tril(
            torch.ones(size=[num_rounds, num_rounds], requires_grad=False))  # shape: (num_rounds, num_rounds)
        att_not_pad = att_not_pad.cuda()
        att_masked = att * att_not_pad  # shape: (batch_size, num_rounds, num_rounds) 
        att_masked = att_masked / torch.sum(att_masked, dim=-1,
                                            keepdim=True)  # shape: (batch_size, num_rounds, num_rounds)
        feat = torch.sum(att_masked.unsqueeze(-1) * hist.unsqueeze(1),
                         dim=-2)  # shape: (batch_size, num_rounds, rnn_dim)

        return feat


class V_Filter(nn.Module):
    """docstring for V_Filter"""

    def __init__(self, config, emb_size=None):
        super(V_Filter, self).__init__()
        
        if emb_size is not None:
            self.emb_size = emb_size
        else:
            self.emb_size = config['emb_size']

        self.filter = nn.Sequential(
            nn.Dropout(p=config["dropout_fc"]),
            nn.Linear(
                self.emb_size,
                config["img_feature_size"]
            ),
            nn.Sigmoid()
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)

    def forward(self, img, ques):
        # img shape: (batch_size, num_rounds, i_dim)
        # ques shape: (batch_size, num_rounds, q_dim)
        # output: img_att (batch_size, num_rounds, embed_dim)

        batch_size = ques.size(0)
        num_rounds = ques.size(1)

        ques_embed = self.filter(ques)  # shape: (batch_size, num_rounds, embed_dim)

        # gated
        img_fused = img * ques_embed

        return img_fused


class DynamicRNN(nn.Module):
    def __init__(self, rnn_model):
        super().__init__()
        self.rnn_model = rnn_model

    def forward(self, seq_input, seq_lens, initial_state=None):
        """A wrapper over pytorch's rnn to handle sequences of variable length.
        Arguments
        ---------
        seq_input : torch.Tensor
            Input sequence tensor (padded) for RNN model.
            Shape: (batch_size, max_sequence_length, embed_size)
        seq_lens : torch.LongTensor
            Length of sequences (b, )
        initial_state : torch.Tensor
            Initial (hidden, cell) states of RNN model.
        Returns
        -------
            Single tensor of shape (batch_size, rnn_hidden_size) corresponding
            to the outputs of the RNN model at the last time step of each input
            sequence.
        """
        max_sequence_length = seq_input.size(1)
        sorted_len, fwd_order, bwd_order = self._get_sorted_order(seq_lens)
        sorted_seq_input = seq_input.index_select(0, fwd_order)
        packed_seq_input = pack_padded_sequence(
            sorted_seq_input, lengths=sorted_len, batch_first=True
        )

        if initial_state is not None:
            hx = initial_state
            assert hx[0].size(0) == self.rnn_model.num_layers
        else:
            sorted_hx = None

        self.rnn_model.flatten_parameters()

        outputs, (h_n, c_n) = self.rnn_model(packed_seq_input, sorted_hx)

        # pick hidden and cell states of last layer
        h_n = h_n[-1].index_select(dim=0, index=bwd_order)
        c_n = c_n[-1].index_select(dim=0, index=bwd_order)

        outputs = pad_packed_sequence(
            outputs, batch_first=True, total_length=max_sequence_length
        )[0].index_select(dim=0, index=bwd_order)

        return outputs, (h_n, c_n)

    @staticmethod
    def _get_sorted_order(lens):
        sorted_len, fwd_order = torch.sort(
            lens.contiguous().view(-1), 0, descending=True
        )
        _, bwd_order = torch.sort(fwd_order)
        sorted_len = list(sorted_len)
        return sorted_len, fwd_order, bwd_order
    
    
def report_metric(targets, outputs):
    targets = np.concatenate(targets).flatten()
    outputs = np.concatenate(outputs)
    predictions = np.argmax(outputs, axis=-1).flatten()
    f1_scores = f1_score(targets, predictions, average=None)
    micro_f1 = f1_score(targets, predictions, average='micro')
    conf_matrix = confusion_matrix(targets, predictions)
    return f1_scores, conf_matrix, micro_f1


def match_embeddings(vocabulary, embeddings):
    dim = list(embeddings.values())[0].shape[0]

    print("Matching word embeddings...")
    embs = []
    for i in tqdm(range(len(vocabulary))):
        token = vocabulary.index2word[i]
        token_emb = embeddings.get(token, np.random.uniform(-1, 1, dim))
        embs.append(token_emb)
    embs = np.stack(embs)
    return embs


def load_embeddings(path):
    with open(path, 'rb') as f:
        embeddings_dict = pickle.load(f)
    return embeddings_dict