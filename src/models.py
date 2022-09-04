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


class EncoderLinear(nn.Module):
    """
    Encoder implementation (can be used for both, AutoRec and VAE)
    """

    def __init__(self, in_dim, params):
        super().__init__()
        # Get relevant hyperparameters from the params dict
        latent_dim = params.get('latent_dim', 10)
        layers_sizes = params.get('layers_sizes', [500, 250])
        layers_sizes.insert(0, in_dim)   # Insert the input dimension

        # Add deep layers
        modules = []
        for i in range(len(layers_sizes) - 1):
            modules.append(
                nn.Linear(layers_sizes[i], layers_sizes[i + 1], bias=True))

        # Add last layer for Z
        modules.append(nn.Linear(layers_sizes[-1], latent_dim, bias=True))

        # Generate the layers sequence foe easier implementation
        self.seq = nn.Sequential(*modules)

    def forward(self, x):
        return self.seq(x)


class DecoderLinear(nn.Module):
    """
    Decoder implementation (can be used for both, AutoRec and VAE)
    """

    def __init__(self, out_dim, params):
        super().__init__()
        # Get relevant hyperparameters from the params dict
        latent_dim = params.get('latent_dim', 10)
        layers_sizes = params.get('layers_sizes', [250, 500])
        layers_sizes.append(out_dim)   # append the output dimension

        modules = []
        # Add last layer for Z
        modules.append(nn.Linear(latent_dim, layers_sizes[0], bias=True))
        modules.append(nn.Sigmoid())

        # Add deep layers
        for i in range(len(layers_sizes) - 1):
            modules.append(
                nn.Linear(layers_sizes[i], layers_sizes[i + 1], bias=True))
            modules.append(nn.Sigmoid())

        # Generate the layers sequence foe easier implementation
        self.seq = nn.Sequential(*modules)

    def forward(self, z):
        return self.seq(z)


class AE(nn.Module):
    """
    AutoEncoder model implementation
    """

    def __init__(self, encoder, decoder):
        super().__init__()

        # Use external implementation of encoder & decoder (actually, using the above EncoderLinear & DecoderLinear)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        return self.decoder(self.encoder(x))


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
    elif model_name == 'AE':
        n_dim = dl_train.dataset.__getitem__(1).shape[0]
        linear_encoder = EncoderLinear(in_dim=n_dim, params=params)
        linear_decoder = DecoderLinear(out_dim=n_dim, params=params)
        model = AE(encoder=linear_encoder, decoder=linear_decoder)

    return model