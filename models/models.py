from torch import nn

from models.encoders import LateFusionEncoder, RvAEncoder, SANEncoder
from models.decoders import DiscDecoder

class LateFusionModel(nn.Module):
    """
    Implementation of LateFusion model
    https://arxiv.org/pdf/1611.08669.pdf
    Based on
    https://github.com/batra-mlp-lab/visdial-challenge-starter-pytorch
    """
    def __init__(self, config, vocabulary, embeddings=None):
        """

        Args:
            config (dict): Configuration dict with network params
            vocabulary (Vocabulary): Mapping between tokens and their ids
            embeddings (torch.Tensor): pre-trained embedding vectors matching token ids
                If None, the embedding layer is initialized randomly
        """
        super().__init__()

        self.encoder = LateFusionEncoder(config, vocabulary, embeddings)
        self.decoder = DiscDecoder(config, vocabulary, embeddings, use_bert=False, bert_path=None)

    def forward(self, image, history, question, options, caption, turn):
        encoder_output = self.encoder(image, history, question)
        scores = self.decoder(encoder_output, options)
        return scores

        
class StackedAttentionModel(nn.Module):
    """
    Implementation of Stacked Attention Network.
    https://arxiv.org/pdf/1511.02274.pdf
    Discriminative decoder from https://arxiv.org/pdf/1611.08669.pdf
    is used.
    """
    def __init__(self, config, vocabulary, embeddings=None, use_bert=False, bert_path=None):
        """

        Args:
            config (dict): Configuration dict with network params
            vocabulary (Vocabulary): Mapping between tokens and their ids
            embeddings (torch.Tensor): pre-trained embedding vectors matching token ids
                If None, the embedding layer is initialized randomly
        """
        super().__init__()

        self.encoder = SANEncoder(config, vocabulary, embeddings, use_bert, bert_path)
        self.decoder = DiscDecoder(config, vocabulary, embeddings, use_bert, bert_path)

    def forward(self, image, history, question, options, caption, turn):
        encoder_output = self.encoder(image, history, question, caption)
        scores = self.decoder(encoder_output, options)
        return scores


class RecursiveAttentionModel(nn.Module):
    """
    Implementation of RecursiveVisualAttention model.
    https://arxiv.org/pdf/1812.02664.pdf
    Based on
    https://github.com/yuleiniu/rva
    """
    def __init__(self, config, vocabulary, embeddings=None):
        """

        Args:
            config (dict): Configuration dict with network params
            vocabulary (Vocabulary): Mapping between tokens and their ids
            embeddings (torch.Tensor): pre-trained embedding vectors matching token ids
                If None, the embedding layer is initialized randomly
        """
        super().__init__()

        self.encoder = RvAEncoder(config, vocabulary, embeddings)
        self.decoder = DiscDecoder(config, vocabulary, embeddings, use_bert=False, bert_path=None)

    def forward(self, image, history, question, options, caption, turn):
        encoder_output = self.encoder(image, history, question, caption, turn)
        scores = self.decoder(encoder_output, options)
        return scores

