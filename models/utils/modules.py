import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from models.utils.utils import GumbelSoftmax, GatedTrans


class ATT_MODULE(nn.Module):
    """docstring for ATT_MODULE"""

    def __init__(self, config, emb_size=None):
        super(ATT_MODULE, self).__init__()
        if emb_size is not None:
            self.emb_size = emb_size
        else:
            self.emb_size = config['emb_size']

        self.V_embed = nn.Sequential(
            nn.Dropout(p=config["dropout_fc"]),
            GatedTrans(
                config["img_feature_size"],
                config["lstm_size"]
            ),
        )
        self.Q_embed = nn.Sequential(
            nn.Dropout(p=config["dropout_fc"]),
            GatedTrans(
                self.emb_size,
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

    def forward(self, img, ques):
        # input
        # img - shape: (batch_size, num_proposals, img_feature_size)
        # ques - shape: (batch_size, num_rounds, word_embedding_size)
        # output
        # att - shape: (batch_size, num_rounds, num_proposals)

        batch_size = ques.size(0)
        num_rounds = ques.size(1)
        num_proposals = img.size(1)

        # img_embed = img.view(-1, img.size(-1))  # shape: (batch_size * num_proposals, img_feature_size)
        img_embed = img.reshape(batch_size*num_proposals, -1)
        img_embed = self.V_embed(img_embed)  # shape: (batch_size, num_proposals, lstm_hidden_size)
        img_embed = img_embed.view(batch_size, num_proposals,
                                   img_embed.size(-1))  # shape: (batch_size, num_proposals, lstm_hidden_size)
        img_embed = img_embed.unsqueeze(1).repeat(1, num_rounds, 1,
                                                  1)  # shape: (batch_size, num_rounds, num_proposals, lstm_hidden_size)

        ques_embed = ques.view(-1, ques.size(-1))  # shape: (batch_size * num_rounds, word_embedding_size)
        ques_embed = self.Q_embed(ques_embed)  # shape: (batch_size, num_rounds, lstm_hidden_size)
        ques_embed = ques_embed.view(batch_size, num_rounds,
                                     ques_embed.size(-1))  # shape: (batch_size, num_rounds, lstm_hidden_size)
        ques_embed = ques_embed.unsqueeze(2).repeat(1, 1, num_proposals,
                                                    1)  # shape: (batch_size, num_rounds, num_proposals, lstm_hidden_size)

        att_embed = F.normalize(img_embed * ques_embed, p=2,
                                dim=-1)  # (batch_size, num_rounds, num_proposals, lstm_hidden_size)
        att_embed = self.att(att_embed).squeeze(-1)  # (batch_size, num_rounds, num_proposals)
        att = self.softmax(att_embed)  # shape: (batch_size, num_rounds, num_proposals)

        return att


class PAIR_MODULE(nn.Module):
    """docstring for PAIR_MODULE"""

    def __init__(self, config):
        super(PAIR_MODULE, self).__init__()

        self.H_embed = nn.Sequential(
            nn.Dropout(p=config["dropout_fc"]),
            GatedTrans(
                config["lstm_size"] * 2,
                config["lstm_size"]),
        )
        self.Q_embed = nn.Sequential(
            nn.Dropout(p=config["dropout_fc"]),
            GatedTrans(
                config["lstm_size"] * 2,
                config["lstm_size"]),
        )
        self.MLP = nn.Sequential(
            nn.Dropout(p=config["dropout_fc"]),
            nn.Linear(
                config["lstm_size"] * 2,
                config["lstm_size"]),
            nn.Dropout(p=config["dropout_fc"]),
            nn.Linear(
                config["lstm_size"],
                1
            )
        )
        self.att = nn.Linear(2, 1)

        self.G_softmax = GumbelSoftmax()

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)

    def forward(self, hist, ques):
        # input
        # ques shape: (batch_size, num_rounds, lstm_hidden_size*2)
        # hist shape: (batch_size, num_rounds, lstm_hidden_size*2)
        # output
        # hist_gs_set - shape: (batch_size, num_rounds, num_rounds)

        batch_size = ques.size(0)
        num_rounds = ques.size(1)

        hist_embed = self.H_embed(hist)  # shape: (batch_size, num_rounds, lstm_hidden_size)
        hist_embed = hist_embed.unsqueeze(1).repeat(1, num_rounds, 1,
                                                    1)  # shape: (batch_size, num_rounds, num_rounds, lstm_hidden_size)

        ques_embed = self.Q_embed(ques)  # shape: (batch_size, num_rounds, lstm_hidden_size)
        ques_embed = ques_embed.unsqueeze(2).repeat(1, 1, num_rounds,
                                                    1)  # shape: (batch_size, num_rounds, num_rounds, lstm_hidden_size)

        att_embed = torch.cat((hist_embed, ques_embed), dim=-1)
        score = self.MLP(att_embed)

        delta_t = torch.tril(torch.ones(size=[num_rounds, num_rounds], requires_grad=False)).cumsum(
            dim=0)  # (num_rounds, num_rounds)
        delta_t = delta_t.view(1, num_rounds, num_rounds, 1).repeat(batch_size, 1, 1,
                                                                    1)  # (batch_size, num_rounds, num_rounds, 1)
        delta_t = delta_t.cuda()
        att_embed = torch.cat((score, delta_t), dim=-1)  # (batch_size, num_rounds, num_rounds, lstm_hidden_size*2)

        hist_logits = self.att(att_embed).squeeze(-1)  # (batch_size, num_rounds, num_rounds)

        # PAIR
        hist_gs_set = torch.zeros_like(hist_logits)
        for i in range(num_rounds):
            # one-hot
            hist_gs_set[:, i, :(i + 1)] = self.G_softmax(hist_logits[:, i, :(i + 1)])  # shape: (batch_size, i+1)

        return hist_gs_set


class INFER_MODULE(nn.Module):
    """docstring for INFER_MODULE"""

    def __init__(self, config, emb_size=None):
        super(INFER_MODULE, self).__init__()

        if emb_size is not None:
            self.emb_size = emb_size
        else:
            self.emb_size = config['emb_size']

        self.embed = nn.Sequential(
            nn.Dropout(p=config["dropout_fc"]),
            GatedTrans(
                self.emb_size,
                config["lstm_size"]),
        )
        self.att = nn.Sequential(
            nn.Dropout(p=config["dropout_fc"]),
            nn.Linear(
                config["lstm_size"],
                2
            )
        )

        self.softmax = nn.Softmax(dim=-1)
        self.G_softmax = GumbelSoftmax()

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)

    def forward(self, ques):
        # input
        # ques - shape: (batch_size, num_rounds, word_embedding_size)
        # output
        # ques_gs - shape: (batch_size, num_rounds, 2)
        # Lambda - shape: (batch_size, num_rounds, 2)

        batch_size = ques.size(0)
        num_rounds = ques.size(1)

        ques_embed = self.embed(ques)  # shape: (batch_size, num_rounds, quen_len_max, lstm_hidden_size)
        ques_embed = F.normalize(ques_embed, p=2,
                                 dim=-1)  # shape: (batch_size, num_rounds, quen_len_max, lstm_hidden_size)
        ques_logits = self.att(ques_embed)  # shape: (batch_size, num_rounds, 2)
        # ignore <pad> word
        ques_gs = self.G_softmax(ques_logits.view(-1, 2)).view(-1, num_rounds, 2)
        Lambda = self.softmax(ques_logits)

        return ques_gs, Lambda


class RvA_MODULE(nn.Module):
    """docstring for R_CALL"""

    def __init__(self, config, emb_size=None):
        super(RvA_MODULE, self).__init__()

        self.INFER_MODULE = INFER_MODULE(config, emb_size)
        self.PAIR_MODULE = PAIR_MODULE(config)
        self.ATT_MODULE = ATT_MODULE(config, emb_size)

    def forward(self, image, ques, hist):
        # img shape: [batch_size, num_proposals, i_dim]
        # img_att_ques shape: [batch_size, num_rounds, num_proposals]
        # img_att_cap shape: [batch_size, 1, num_proposals]
        # ques_gs shape: [batch_size, num_rounds, 2]
        # hist_logits shape: [batch_size, num_rounds, num_rounds]
        # ques_gs_prob shape: [batch_size, num_rounds, 2]

        cap_feat, ques_feat, ques_encoded = ques

        batch_size = ques_feat.size(0)
        num_rounds = ques_feat.size(1)
        num_proposals = image.size(1)

        ques_gs, ques_gs_prob = self.INFER_MODULE(ques_feat)  # (batch_size, num_rounds, 2)
        hist_gs_set = self.PAIR_MODULE(hist, ques_encoded)
        img_att_ques = self.ATT_MODULE(image, ques_feat)
        img_att_cap = self.ATT_MODULE(image, cap_feat)

        # soft
        ques_prob_single = torch.Tensor(data=[1, 0]).view(1, -1).repeat(batch_size, 1)  # shape: [batch_size, 2]
        ques_prob_single = ques_prob_single.cuda()
        ques_prob_single.requires_grad = False

        img_att_refined = img_att_ques.data.clone().zero_()  # shape: [batch_size, num_rounds, num_proposals]
        for i in range(num_rounds):
            if i == 0:
                img_att_temp = img_att_cap.view(-1, img_att_cap.size(-1))  # shape: [batch_size, num_proposals]
            else:
                hist_gs = hist_gs_set[:, i, :(i + 1)]  # shape: [batch_size, i+1]
                img_att_temp = torch.cat((img_att_cap, img_att_refined[:, :i, :]),
                                         dim=1)  # shape: [batch_size, i+1, num_proposals]
                img_att_temp = torch.sum(hist_gs.unsqueeze(-1) * img_att_temp,
                                         dim=-2)  # shape: [batch_size, num_proposals]
            img_att_cat = torch.cat((img_att_ques[:, i, :].unsqueeze(1), img_att_temp.unsqueeze(1)),
                                    dim=1)  # shape: [batch_size ,2, num_proposals]
            # soft
            ques_prob_pair = ques_gs_prob[:, i, :]
            ques_prob = torch.cat((ques_prob_single, ques_prob_pair), dim=-1)  # shape: [batch_size, 2]
            ques_prob = ques_prob.view(-1, 2, 2)  # shape: [batch_size, 2, 2]
            ques_prob_refine = torch.bmm(ques_gs[:, i, :].view(-1, 1, 2), ques_prob).view(-1, 1,
                                                                                          2)  # shape: [batch_size, num_rounds, 2]

            img_att_refined[:, i, :] = torch.bmm(ques_prob_refine, img_att_cat).view(-1,
                                                                                     num_proposals)  # shape: [batch_size, num_proposals]

        return img_att_refined, (ques_gs, hist_gs_set, img_att_ques)


# class CompactBilinearPooling(nn.Module):
#     """
#     Compute compact bilinear pooling over two bottom inputs.
#     Args:
#         output_dim: output dimension for compact bilinear pooling.
#         sum_pool: (Optional) If True, sum the output along height and width
#                   dimensions and return output shape [batch_size, output_dim].
#                   Otherwise return [batch_size, height, width, output_dim].
#                   Default: True.
#         rand_h_1: (Optional) an 1D numpy array containing indices in interval
#                   `[0, output_dim)`. Automatically generated from `seed_h_1`
#                   if is None.
#         rand_s_1: (Optional) an 1D numpy array of 1 and -1, having the same shape
#                   as `rand_h_1`. Automatically generated from `seed_s_1` if is
#                   None.
#         rand_h_2: (Optional) an 1D numpy array containing indices in interval
#                   `[0, output_dim)`. Automatically generated from `seed_h_2`
#                   if is None.
#         rand_s_2: (Optional) an 1D numpy array of 1 and -1, having the same shape
#                   as `rand_h_2`. Automatically generated from `seed_s_2` if is
#                   None.
#     """
# 
#     def __init__(self, input_dim1, input_dim2, output_dim,
#                  sum_pool=True, cuda=True,
#                  rand_h_1=None, rand_s_1=None, rand_h_2=None, rand_s_2=None):
#         super(CompactBilinearPooling, self).__init__()
#         self.input_dim1 = input_dim1
#         self.input_dim2 = input_dim2
#         self.output_dim = output_dim
#         self.sum_pool = sum_pool
# 
#         if rand_h_1 is None:
#             np.random.seed(1)
#             rand_h_1 = np.random.randint(output_dim, size=self.input_dim1)
#         if rand_s_1 is None:
#             np.random.seed(3)
#             rand_s_1 = 2 * np.random.randint(2, size=self.input_dim1) - 1
# 
#         self.sparse_sketch_matrix1 = Variable(self.generate_sketch_matrix(
#             rand_h_1, rand_s_1, self.output_dim))
# 
#         if rand_h_2 is None:
#             np.random.seed(5)
#             rand_h_2 = np.random.randint(output_dim, size=self.input_dim2)
#         if rand_s_2 is None:
#             np.random.seed(7)
#             rand_s_2 = 2 * np.random.randint(2, size=self.input_dim2) - 1
# 
#         self.sparse_sketch_matrix2 = Variable(self.generate_sketch_matrix(
#             rand_h_2, rand_s_2, self.output_dim))
# 
#         if cuda:
#             self.sparse_sketch_matrix1 = self.sparse_sketch_matrix1.cuda()
#             self.sparse_sketch_matrix2 = self.sparse_sketch_matrix2.cuda()
# 
#     def forward(self, bottom1, bottom2):
#         """
#         bottom1: 1st input, 4D Tensor of shape [batch_size, input_dim1, height, width].
#         bottom2: 2nd input, 4D Tensor of shape [batch_size, input_dim2, height, width].
#         """
#         assert bottom1.size(1) == self.input_dim1 and \
#             bottom2.size(1) == self.input_dim2
# 
#         batch_size, _, height, width = bottom1.size()
# 
#         bottom1_flat = bottom1.permute(0, 2, 3, 1).contiguous().view(-1, self.input_dim1)
#         bottom2_flat = bottom2.permute(0, 2, 3, 1).contiguous().view(-1, self.input_dim2)
# 
#         sketch_1 = bottom1_flat.mm(self.sparse_sketch_matrix1)
#         sketch_2 = bottom2_flat.mm(self.sparse_sketch_matrix2)
# 
#         fft1_real, fft1_imag = afft.Fft()(sketch_1, Variable(torch.zeros(sketch_1.size())).cuda())
#         fft2_real, fft2_imag = afft.Fft()(sketch_2, Variable(torch.zeros(sketch_2.size())).cuda())
# 
#         fft_product_real = fft1_real.mul(fft2_real) - fft1_imag.mul(fft2_imag)
#         fft_product_imag = fft1_real.mul(fft2_imag) + fft1_imag.mul(fft2_real)
# 
#         cbp_flat = afft.Ifft()(fft_product_real, fft_product_imag)[0]
# 
#         cbp = cbp_flat.view(batch_size, height, width, self.output_dim)
# 
#         if self.sum_pool:
#             cbp = cbp.sum(dim=1).sum(dim=1)
# 
#         return cbp
# 
#     @staticmethod
#     def generate_sketch_matrix(rand_h, rand_s, output_dim):
#         """
#         Return a sparse matrix used for tensor sketch operation in compact bilinear
#         pooling
#         Args:
#             rand_h: an 1D numpy array containing indices in interval `[0, output_dim)`.
#             rand_s: an 1D numpy array of 1 and -1, having the same shape as `rand_h`.
#             output_dim: the output dimensions of compact bilinear pooling.
#         Returns:
#             a sparse matrix of shape [input_dim, output_dim] for tensor sketch.
#         """
# 
#         # Generate a sparse matrix for tensor count sketch
#         rand_h = rand_h.astype(np.int64)
#         rand_s = rand_s.astype(np.float32)
#         assert(rand_h.ndim == 1 and rand_s.ndim ==
#                1 and len(rand_h) == len(rand_s))
#         assert(np.all(rand_h >= 0) and np.all(rand_h < output_dim))
# 
#         input_dim = len(rand_h)
#         indices = np.concatenate((np.arange(input_dim)[..., np.newaxis],
#                                   rand_h[..., np.newaxis]), axis=1)
#         indices = torch.from_numpy(indices)
#         rand_s = torch.from_numpy(rand_s)
#         sparse_sketch_matrix = torch.sparse.FloatTensor(
#             indices.t(), rand_s, torch.Size([input_dim, output_dim]))
#         return sparse_sketch_matrix.to_dense()
    
