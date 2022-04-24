import torch
from torch import nn


class GMF(nn.Module):
    def __init__(self, n_genes, n_spots, spots_pdist, device,
                 embedding_dimensions=8, embeddings_init_std=.1, dist_reg_factor=1, **kwargs):
        super().__init__()
        self.dist_reg_factor = dist_reg_factor
        self.spots_pdist = spots_pdist.to(device)
        self.gene_embeddings = nn.Embedding(n_genes, embedding_dimensions, device=device)
        self.spot_embeddings = nn.Embedding(n_spots, embedding_dimensions, device=device)
        self.gene_embeddings.weight = nn.Parameter(torch.normal(0, embeddings_init_std,
                                                                self.gene_embeddings.weight.shape).to(device))
        self.spot_embeddings.weight = nn.Parameter(torch.normal(0, embeddings_init_std,
                                                                self.spot_embeddings.weight.shape).to(device))
        self.dense = nn.Linear(embedding_dimensions, 1, device=device)
        
    def forward(self, gene_indices, spot_indices):
        gene_embeddings = self.gene_embeddings(gene_indices)
        spot_embeddings = self.spot_embeddings(spot_indices)
        x = torch.mul(gene_embeddings, spot_embeddings)
        return self.dense(x)[:, 0]