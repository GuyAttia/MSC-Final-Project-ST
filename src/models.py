import torch
from torch import nn


class NMF(nn.Module):
    """
    Neural Matrix Factorizaion model implementation
    """

    def __init__(self, num_genes, num_spots, params):
        super().__init__()
        # Get relevant hyperparameters from the params dict
        latent_dim = params['latent_dim']

        # Initialize embedding layers for the users and for the items
        self.embedding_genes = torch.nn.Embedding(num_embeddings=num_genes+1, embedding_dim=latent_dim)
        self.embedding_spots = torch.nn.Embedding(num_embeddings=num_spots+1, embedding_dim=latent_dim)

    def forward(self, gene_indices, spot_indices):
        # Get the gene and spot vector using the embedding layers
        gene_embedding = self.embedding_genes(gene_indices)
        spot_embedding = self.embedding_spots(spot_indices)

        # Calculate the expression for the gene-spot combination
        output = (gene_embedding * spot_embedding).sum(1)
        return output


def get_model(model_name, params, dl_train):
    """
    Instantiate the proper model based on the model_name parameter. 
    Use the needed hyperparameters from params.
    Also, extract the needed data dimensions for building the models.
    """
    model = None

    if model_name == 'NMF':
        num_genes = dl_train.dataset.num_genes
        num_spots = dl_train.dataset.num_spots
        model = NMF(num_genes=num_genes, num_spots=num_spots, params=params)

    return model
