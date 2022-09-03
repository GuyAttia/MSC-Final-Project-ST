from os import path

import torch
from ignite.contrib.handlers import TensorboardLogger, global_step_from_engine
from ignite.engine import Engine, Events
from ignite.handlers import EarlyStopping, ModelCheckpoint
from ignite.metrics import Loss

from loss import *


import torch.optim as optim
from data import get_data
from models import get_model
import numpy as np
import pandas as pd
from os import path, listdir
from copy import deepcopy
import stlearn as st
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from torch import tensor
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from scanpy_stlearn_loaders import StlearnLoader
import trainer as trainer
from data import get_data

# We used the Ignite package for smarter building of our trainers.
# This package provide built-in loggers and handlers for different actions.

def trainer(model, optimizer, criterion, max_epochs, early_stopping, dl_train, dl_test, device, dataset_name, model_name):
    """
    Build a trainer for a model
    """
    model = model.to(device)

    def train_step(engine, batch):
        """
        Define the train step.
        Each sample in the batch is actually a tuple of (user, item, rating)
        """
        gens, spots, y = batch
        gens.to(device)
        spots.to(device)
        y = y.float().to('cpu')

        model.train()
        y_pred = model(gens, spots).to('cpu')
        loss = criterion(y_pred, y, spots, gens)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()

    trainer = Engine(train_step)    # Instantiate the trainer object

    def validation_step(engine, batch):
        """
        Define the validation step (without updating the parameters).
        """
        model.eval()

        with torch.no_grad():
            gens, spots, y = batch
            gens.to(device)
            spots.to(device)
            y.to('cpu')
            y_pred = model(gens, spots).to('cpu')
            return y_pred, y

    train_evaluator = Engine(validation_step)
    val_evaluator = Engine(validation_step)

    val_metrics = {
        "loss": Loss(criterion),
        'NonZeroRMSE': Loss(NON_ZERO_RMSELoss),
    }
    # Attach metrics to the evaluators
    val_metrics['loss'].attach(train_evaluator, 'loss')
    val_metrics['loss'].attach(val_evaluator, 'loss')
    val_metrics['NonZeroRMSE'].attach(val_evaluator, 'NonZeroRMSE')

    # Attach logger to print the training loss after each epoch
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(trainer):
        train_evaluator.run(dl_train)
        metrics = train_evaluator.state.metrics
        print(
            f"Training Results - Epoch[{trainer.state.epoch}] Avg loss: {metrics['loss']:.2f}")

    # Attach logger to print the validation loss after each epoch
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(trainer):
        val_evaluator.run(dl_test)
        metrics = val_evaluator.state.metrics
        print(
            f"Validation Results - Epoch[{trainer.state.epoch}] Avg loss: {metrics['loss']:.2f}  |  Avg NonZeroRMSE: {metrics['NonZeroRMSE']:.2f}")

    def score_function(engine):
        """
        Define the score function as the negative of our loss, because the ModelCheckpoint & Early-Stopping mechanisms are fixed to maximize.
        """
        val_loss = engine.state.metrics['loss']
        return -val_loss

    # Model Checkpoint
    checkpoint_dir = path.join("checkpoints", dataset_name)
    # Checkpoint to store n_saved best models wrt score function
    model_checkpoint = ModelCheckpoint(
        checkpoint_dir,
        n_saved=1,
        filename_prefix=f"best_{model_name}",
        score_function=score_function,
        score_name='neg_loss',
        global_step_transform=global_step_from_engine(trainer),  # helps fetch the trainer's state
        require_empty=False
    )
    # After each epoch if the validation results are better - save the model as a file
    val_evaluator.add_event_handler(
        Events.COMPLETED, model_checkpoint, {"model": model})

    # Early stopping
    if early_stopping > 0:
        handler = EarlyStopping(
            patience=early_stopping,
            score_function=score_function, 
            trainer=trainer
        )
        val_evaluator.add_event_handler(Events.COMPLETED, handler)

    # Tensorboard logger - log the training and evaluation losses as function of the iterations & epochs
    tb_logger = TensorboardLogger(
        log_dir=path.join('logs', dataset_name, model_name))
    tb_logger.attach_output_handler(
        trainer,
        event_name=Events.ITERATION_COMPLETED(every=100),
        tag="training",
        output_transform=lambda loss: {"batch_loss": loss},
    )

    for tag, evaluator in [("training", train_evaluator), ("validation", val_evaluator)]:
        tb_logger.attach_output_handler(
            evaluator,
            event_name=Events.EPOCH_COMPLETED,
            tag=tag,
            metric_names="all",
            global_step_transform=global_step_from_engine(trainer),
        )

    # Run the trainer
    trainer.run(dl_train, max_epochs=max_epochs)

    tb_logger.close()   # Close logger
    # Return the best validation score
    best_val_score = -handler.best_score
    return model, best_val_score



class ExpressionDataset(Dataset):
    """
    Generate expression dataset to use in the our models, where each sample should be a tuple of (gene, spot, expression)
    """

    def __init__(self, df, device):
        self.num_samples = len(df)
        self.genes = tensor(df['gene'].values).to(device)
        self.spots = tensor(df['spot'].values).to(device)
        self.labels = tensor(df['expression'].values)
        self.num_genes = df['gene'].max()
        self.num_spots = df['spot'].max()

    def __getitem__(self, index):
        gene = self.genes[index]
        spot = self.spots[index]
        label = self.labels[index].item()
        return gene, spot, label

    def __len__(self):
        return self.num_samples

    def get_all_data(self):
        return self.genes, self.spots, self.labels
        
# Only for testing
if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset_name = 'Visium_Mouse_Olfactory_Bulb'
    max_epochs = 2
    early_stopping = 15
    model_name = 'NMF'
    best_params = {
        'learning_rate': 0.01,
        'optimizer': "RMSprop",
        'latent_dim': 40,
        'batch_size': 128
    }

    obj = StlearnLoader().load_local_visum(path=path.join('/', 'data', dataset_name), count_file='filtered_feature_bc_matrix.h5')
    obj.layers["raw_count"] = obj.X
    x = obj.X.toarray()
    n_spots, n_genes = x.shape

    # Keep genes with at least 15% non zero spots
    min_cells = int(n_spots * 0.985)
    st.pp.filter_genes(obj, min_cells=min_cells)
    x = obj.X.toarray()
    n_spots, n_genes = x.shape

    obj.raw = obj
    x = np.log(x+1)
    
    spots_values = obj.obs.index.values
    genes_values = obj.var.index.values
    df_expressions_matrix = pd.DataFrame(x, columns=genes_values, index=spots_values)
    df_expressions = df_expressions_matrix.stack().reset_index()
    df_expressions.columns = ['spot', 'gene', 'expression']

    # Ordinal encoding the genes and spots for supported type
    oe_genes = OrdinalEncoder()
    df_expressions[['gene']] = oe_genes.fit_transform(df_expressions[['gene']].values)
    oe_spots = OrdinalEncoder()
    df_expressions[['spot']] = oe_spots.fit_transform(df_expressions[['spot']].values)
    df_expressions[['spot', 'gene']] = df_expressions[['spot', 'gene']].astype(int)

    df_train, df_test = train_test_split(df_expressions, test_size=0.10)
    df_train, df_valid = train_test_split(df_train, test_size=0.10)

    ds_train = ExpressionDataset(df=df_train, device=device)
    ds_valid = ExpressionDataset(df=df_valid, device=device)
    ds_test = ExpressionDataset(df=df_test, device=device)
    
    batch_size = 128
    dl_train = DataLoader(dataset=ds_train, batch_size=batch_size, shuffle=True)
    dl_valid = DataLoader(dataset=ds_valid, batch_size=batch_size, shuffle=True)
    dl_test = DataLoader(dataset=ds_test, batch_size=batch_size, shuffle=True)

    # dl_train, _, dl_test = get_data(dataset_name=dataset_name, batch_size=best_params['batch_size'], device=device)  # Get data
    model = get_model(model_name, best_params, dl_train)  # Build model
    optimizer = getattr(optim, best_params['optimizer'])(model.parameters(), lr=best_params['learning_rate'])  # Instantiate optimizer
    df_spots_neighbors = pd.read_csv(path.join('/', 'FPST', 'src', 'spots_neighbors.csv'))
    criterion = NON_ZERO_RMSELoss_Spatial(df_spots_neighbors=df_spots_neighbors, alpha=0.1, beta=0.1)
    # criterion = NON_ZERO_RMSELoss()
    # criterion = RMSELoss()
    test_loss = trainer(
                    model=model, 
                    optimizer=optimizer, 
                    criterion=criterion, 
                    max_epochs=max_epochs, 
                    early_stopping=early_stopping, 
                    dl_train=dl_train, 
                    dl_test=dl_valid, 
                    device=device, 
                    dataset_name=dataset_name, 
                    model_name=model_name
                )
                