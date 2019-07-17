from torch import nn

from models.encoders.encoders import LateFusionEncoder, RvAEncoder, SANEncoder
from models.decoders.decoders import DiscDecoder

class LateFusionModel(nn.Module):
    def __init__(self, config, vocabulary, embeddings=None):
        """

        Args:
            config:
            vocabulary:
        """
        super().__init__()

        self.encoder = LateFusionEncoder(config, vocabulary, embeddings)
        self.decoder = DiscDecoder(config, vocabulary, embeddings)

    def forward(self, image, history, question, options):
        encoder_output = self.encoder(image, history, question)
        scores = self.decoder(encoder_output, options)
        return scores

        
class StackedAttentionModel(nn.Module):
    def __init__(self, config, vocabulary, embeddings=None):
        """
        
        Args:
            config:
            vocabulary:
        """
        super().__init__()

        self.encoder = SANEncoder(config, vocabulary, embeddings)
        self.decoder = DiscDecoder(config, vocabulary, embeddings)

    def forward(self, image, history, question, options):
        encoder_output = self.encoder(image, history, question)
        scores = self.decoder(encoder_output, options)
        return scores


class RecursiveAttentionModel(nn.Module):
    def __init__(self, config, vocabulary, embeddings=None):
        """

        Args:
            config:
            vocabulary:
        """
        super().__init__()

        self.encoder = RvAEncoder(config, vocabulary, embeddings)
        self.decoder = DiscDecoder(config, vocabulary, embeddings)

    def forward(self, image, history, question, options):
        encoder_output = self.encoder(image, history, question)
        scores = self.decoder(encoder_output, options)
        return scores
