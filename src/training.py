import torch
import numpy as np
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau, ExponentialLR, StepLR
from scipy.sparse import csr_matrix, vstack as sparse_vstack
import torch.optim.lbfgs

class AbstractModelTrainer:
    def __init__(self, train_dataloader, validation_dataloader,
                 model, device, optimizer_type=torch.optim.Adam, optimizer_kwargs=None,
                 lr=1e-3, n_epochs=4, verbose=True, patience=0, eps=1e-4, **kwargs):
        if optimizer_kwargs is None:
            optimizer_kwargs = {}

        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader
        self.model = model
        self.device = device
        self.optimizer = optimizer_type(model.parameters(), lr=lr, **optimizer_kwargs)
        self.n_epochs = n_epochs
        self.verbose = verbose
        self.patience = patience
        self.eps = eps

    def train(self):
        train_hist = []
        validation_hist = []
        train_rmse_hist = []
        validation_rmse_hist = []
        best_validation_mse = np.inf
        remaining_patience = self.patience
        for epoch in range(1, self.n_epochs + 1):
            self.model.train()
            epoch_train_losses = []
            epoch_train_mses = []
            pbar = tqdm(iter(self.train_dataloader),
                        desc=f'Train epoch {epoch}/{self.n_epochs}',
                        disable=not self.verbose)

            # Training epoch
            for batch in pbar:
                batch = batch[0].to(self.device)
                loss, mse = self.calc_loss(batch)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # train metrics - epoch
                epoch_train_losses.append(loss.item())
                epoch_train_mses.append(mse.item())
                pbar.set_postfix({'rmse':np.sqrt(np.nanmean(epoch_train_mses)), 'loss':np.sqrt(np.nanmean(epoch_train_losses))})

            pbar.close()
            
            # train metrics - total
            train_hist.append(np.mean(epoch_train_losses))
            train_rmse_hist.append(np.sqrt(np.nanmean(epoch_train_mses)))
            epoch_validation_losses = []
            epoch_validation_mses = []
            pbar = tqdm(iter(self.validation_dataloader),
                        desc=f'Validation', disable=not self.verbose)

            # Evaluating epoch
            self.model.eval()
            with torch.no_grad():
                for batch in pbar:
                    batch = batch[0].to(self.device)
                    val_loss, val_mse = self.calc_loss(batch)
                    epoch_validation_losses.append(val_loss.item())
                    epoch_validation_mses.append(val_mse.item())
                    pbar.set_postfix({'Loss': np.mean(epoch_validation_losses), 'rmse':str(round(np.sqrt(np.nanmean(epoch_validation_mses)), 5))})

            pbar.close()

            # validation metrics
            curr_validation_loss = np.nanmean(epoch_validation_losses)
            validation_hist.append(curr_validation_loss)

            curr_validation_mse = np.nanmean(epoch_validation_mses)
            validation_rmse_hist.append(np.sqrt(curr_validation_mse))

            # Early stopping
            if curr_validation_mse <= best_validation_mse - self.eps:
                best_validation_mse = curr_validation_mse
                remaining_patience = self.patience
            else:
                remaining_patience -= 1
                if remaining_patience == 0:
                    break

        return train_hist, validation_hist, train_rmse_hist, validation_rmse_hist
    
    def calc_loss(self, batch):
        raise NotImplementedError('Subclasses must implement calc_loss')
    

class GMFTrainer(AbstractModelTrainer):
    def calc_loss(self, batch):
        pred = self.model(batch[:, 0], batch[:, 1])
        mse = torch.mean((batch[:, 2] - pred) ** 2)
        reg_term = 0
        if (self.model.dist_reg_factor > 0):
              
#             embedding_pdist = torch.nn.functional.pdist(self.model.spot_embeddings.weight)
              embedding_pdist = torch.cdist(self.model.spot_embeddings.weight, self.model.spot_embeddings.weight)
              embedding_pdist = (embedding_pdist - embedding_pdist.min()) / (embedding_pdist.max() - embedding_pdist.min())
              reg_term = self.model.dist_reg_factor * torch.sum((embedding_pdist - self.model.spots_pdist) ** 2)
        loss = mse + reg_term
        return loss, mse