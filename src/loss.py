import torch
from torch import nn
import pandas as pd
import numpy as np


# --------------------------------------------------- Losses ---------------------------------------------------------------------
class RMSELoss(nn.Module):
    """
    Implement RMSE loss using PyTorch MSELoss existing class.
    """

    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, yhat, y, **kwargs):
        loss = torch.sqrt(self.mse(yhat, y) + self.eps)
        return loss


class NON_ZERO_RMSELoss(nn.Module):
    """
    Implement another RMSE loss that clear the zero indices.
    """

    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps  # Add eps to avoid devision by 0

    def forward(self, yhat, y, **kwargs):
        # Create mask for all non zero items in the tensor
        non_zero_mask = torch.nonzero(y, as_tuple=True)
        y_non_zeros = y[non_zero_mask]  # Keep only non zero in y
        yhat_non_zeros = yhat[non_zero_mask]    # Keep only non zero in y_hat

        loss = torch.sqrt(self.mse(yhat_non_zeros, y_non_zeros) + self.eps)
        return loss


class NON_ZERO_RMSELoss_Spatial(nn.Module):
    """
    Implement another RMSE loss that clear the zero indices while adding regularization of the spatial distance
    """

    def __init__(self, eps=1e-6, df_spots_neighbors=None, alpha=0.1, beta=0.1):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps  # Add eps to avoid devision by 0
        self.df_spots_neighbors = df_spots_neighbors
        self.alpha = alpha
        self.beta = beta

    def forward(self, yhat, y, spots, genes, **kwargs):
        # Create mask for all non zero items in the tensor
        non_zero_mask = torch.nonzero(y, as_tuple=True)
        y_non_zeros = y[non_zero_mask]  # Keep only non zero in y
        yhat_non_zeros = yhat[non_zero_mask]    # Keep only non zero in y_hat

        loss = torch.sqrt(self.mse(yhat_non_zeros, y_non_zeros) + self.eps)

        # Spatial loss
        df = pd.DataFrame({'gene': genes.cpu(), 'spot': spots.cpu(), 'yhat': yhat.detach()})

        first_degree_loss = 0
        second_degree_loss = 0

        for i, (gene, spot, yhat_) in df.iterrows():
            first_degree_neighbors = self.df_spots_neighbors.iloc[int(spot)][self.df_spots_neighbors.iloc[int(spot)]==1].index.values.astype(int)
            second_degree_neighbors = self.df_spots_neighbors.iloc[int(spot)][self.df_spots_neighbors.iloc[int(spot)]==2].index.values.astype(int)

            mask_spot = df['spot'] == spot
            for spot_gene in df.loc[mask_spot, 'gene'].unique():
                mask_gene = df['gene'] == spot_gene
                # First degree loss
                mask_neighbors = df['spot'].isin(first_degree_neighbors)
                neighbors_expression = df.loc[mask_neighbors & mask_gene, 'yhat']
                if len(neighbors_expression) > 0:
                    part_loss = np.sqrt(np.mean(np.square(neighbors_expression - yhat_)))
                    first_degree_loss += part_loss

                # Second degree loss
                mask_neighbors = df['spot'].isin(second_degree_neighbors)
                neighbors_expression = df.loc[mask_neighbors & mask_gene, 'yhat']
                if len(neighbors_expression) > 0:
                    part_loss = np.sqrt(np.mean(np.square(neighbors_expression - yhat_)))
                    first_degree_loss += part_loss


        loss = loss + (self.alpha * first_degree_loss) + (self.beta * second_degree_loss)

        return loss



class NON_ZERO_RMSELoss_Spatial_AE(nn.Module):
    """
    Implement another RMSE loss that clear the zero indices while adding regularization of the spatial distance
    """

    def __init__(self, eps=1e-6, df_spots_neighbors=None, alpha=0.1, beta=0.1):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps  # Add eps to avoid devision by 0
        self.df_spots_neighbors = df_spots_neighbors
        self.alpha = alpha
        self.beta = beta

        # Replace the neighbors closeness to the regularization value
        self.df_spots_neighbors.replace({1.: self.alpha, 2.: self.beta}, inplace=True)

    def forward(self, yhat, y, **kwargs):
        # Create mask for all non zero items in the tensor
        non_zero_mask = torch.nonzero(y, as_tuple=True)
        y_non_zeros = y[non_zero_mask]  # Keep only non zero in y
        yhat_non_zeros = yhat[non_zero_mask]    # Keep only non zero in y_hat

        non_zero_error = torch.square(yhat_non_zeros, y_non_zeros)
        squared_error = torch.square(y - yhat)

        # Spatial loss
        df_spatial_loss = np.zeros(shape=y.shape)
        # Iterate over all spots and find their neighbors sum loss
        for i, spot_neighbors in enumerate(self.df_spots_neighbors.itterrows()):
            df_spatial_loss[:, i] = squared_error @ spot_neighbors

        spatial_loss_non_zeros = df_spatial_loss[non_zero_mask]    # Keep only non zero in y_hat

        total_squared_error = non_zero_error + spatial_loss_non_zeros
        loss = torch.sqrt(torch.mean(total_squared_error) + self.eps)
        return loss