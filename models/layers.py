import torch
import numpy as np

class EmbeddingLayer(torch.nn.Module):

    def __init__(self, field_dims, embed_dim):
        super().__init__()
        self.embedding = torch.nn.Embedding(sum(field_dims), embed_dim)
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]))
        torch.nn.init.xavier_uniform_(self.embedding.weight.data)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return self.embedding(x)

class EmbeddingLayer_wo_offset(torch.nn.Module):

    def __init__(self, field_dims, embed_dim):
        super().__init__()
        self.embedding = torch.nn.Embedding(field_dims, embed_dim)
        torch.nn.init.xavier_uniform_(self.embedding.weight.data)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        return self.embedding(x)

class MultiLayerPerceptron(torch.nn.Module):

    def __init__(self, input_dim, embed_dims, dropout, output_layer=True):
        super().__init__()
        layers = list()
        for embed_dim in embed_dims:
            layers.append(torch.nn.Linear(input_dim, embed_dim))
            layers.append(torch.nn.BatchNorm1d(embed_dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = embed_dim
        if output_layer:
            layers.append(torch.nn.Linear(input_dim, 1))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, embed_dim)``
        """
        return self.mlp(x)
    
class Attention(torch.nn.Module):
    """Self-attention layer for click and purchase"""

    def __init__(self, dim=32):
        super(Attention, self).__init__()
        self.dim = dim
        self.q_layer = torch.nn.Linear(dim, dim)
        self.k_layer = torch.nn.Linear(dim, dim)
        self.v_layer = torch.nn.Linear(dim, dim)
        self.softmax = torch.nn.Softmax(1)

    def forward(self, inputs):
        Q = self.q_layer(inputs)
        K = self.k_layer(inputs)
        V = self.v_layer(inputs)
        a = (Q * K).sum(-1) / (self.dim**0.5)
        a = self.softmax(a)
        outputs = (a.unsqueeze(-1) * V).sum(1)
        return outputs