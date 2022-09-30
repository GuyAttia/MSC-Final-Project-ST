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

class NON_ZERO_RMSELoss_AE(nn.Module):
    """
    Implement another RMSE loss that clear the zero indices.
    """

    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps  # Add eps to avoid devision by 0

    def forward(self, yhat, y, batch_mask, **kwargs):
        # Keep only relevant samples (for validation and test datasets)
        y = y[batch_mask]
        yhat = yhat[batch_mask]
        
        # Create mask for all non zero items in the tensor
        non_zero_mask = torch.nonzero(y, as_tuple=True)
        y_non_zeros = y[non_zero_mask]  # Keep only non zero in y
        yhat_non_zeros = yhat[non_zero_mask]    # Keep only non zero in y_hat

        loss = torch.sqrt(self.mse(yhat_non_zeros, y_non_zeros) + self.eps)
        return loss

class NON_ZERO_RMSELoss_Spatial_AE(nn.Module):
    """
    Implement another RMSE loss that clear the zero indices while adding regularization of the spatial distance
    """

    def __init__(self, eps=1e-6, df_spots_neighbors=None, alpha=0.1, beta=0.1):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps  # Add eps to avoid devision by 0
        self.alpha = alpha
        self.beta = beta
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Replace the neighbors closeness to the regularization value
        df_spots_neighbors.replace({1.: self.alpha, 2.: self.beta}, inplace=True)
        self.spots_neighbors_tensor = torch.tensor(df_spots_neighbors.values, device=self.device, dtype=torch.float32)


    def forward(self, yhat, y, batch_mask, **kwargs):
        if batch_mask is not None:  # for valid and test sets
            actual_mask = batch_mask
        else:   # For training
            # Create mask for all non zero items in the tensor
            actual_mask = torch.nonzero(y, as_tuple=True)
        
        y_masked = y[actual_mask]  # Keep only masked in y
        yhat_masked = yhat[actual_mask]    # Keep only masked in y_hat

        masked_error = torch.square(yhat_masked - y_masked)
        squared_error = torch.square(y - yhat)

        # Spatial loss
        spatial_loss_tensor = torch.zeros(size=y.shape, device=self.device)
        # Iterate over all spots and find their neighbors sum loss
        for i, spot_neighbors in enumerate(self.spots_neighbors_tensor):
            spatial_loss_tensor[:, i] = torch.matmul(input=squared_error, other=spot_neighbors)

        spatial_loss_masked = spatial_loss_tensor[actual_mask]    # Keep only masked

        total_squared_error = masked_error + spatial_loss_masked
        loss = torch.sqrt(torch.mean(total_squared_error) + self.eps)
        return loss