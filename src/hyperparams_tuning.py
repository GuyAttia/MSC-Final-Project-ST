import optuna
import pandas as pd
import torch
import torch.optim as optim

import trainer as trainer
from data_ae import get_data
from models import get_model
from loss import *


def train(model_name, model, optimizer, criterion, epochs, dl_train, dl_test, device, dataset_name):
    """
    Execute the proper trainer with the right model, optimizer and relevant data loaders.
    """
    loss = trainer.trainer(
        model=model, 
        optimizer=optimizer,
        criterion=criterion, 
        max_epochs=epochs, 
        early_stopping=3,
        dl_train=dl_train, 
        dl_test=dl_test, 
        device=device, 
        dataset_name=dataset_name, 
        model_name=model_name
    )
    return loss, model


def tune_params(model_name, dataset_name, n_trials, max_epochs, device):
    """
    Use the Optuna package for hyperparameters tuning.
    - Define the ranges for the relevant hyperparameters
    - Sample different hyperparameters combinations using RandomSampler (can be changed)
    - Train the model using the sample and keep the validation loss.
    After many trials, decide on the best hyperparameters and return both the trials results and the entire study.
    """

    def objective(trial):
        if model_name == 'NMF':
            params = {
                # 'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-1),
                'learning_rate': trial.suggest_categorical('learning_rate', [0.001, 0.01, 0.1, 1]),
                'optimizer': trial.suggest_categorical("optimizer", ["SGD"]),
                # 'optimizer': trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"]),
                # 'latent_dim': trial.suggest_int("latent_dim", 10, 20),
                'latent_dim': trial.suggest_categorical("latent_dim", [10, 40, 100, 300, 500]),
                'batch_size': trial.suggest_categorical("batch_size", [512])
            }
        elif model_name == 'NMF2':
            params = {
                # 'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-1),
                'learning_rate': trial.suggest_categorical('learning_rate', [0.001, 0.01, 0.1, 1]),
                'optimizer': trial.suggest_categorical("optimizer", ["SGD"]),
                # 'optimizer': trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"]),
                # 'latent_dim': trial.suggest_int("latent_dim", 10, 20),
                'latent_dim': trial.suggest_categorical("latent_dim", [10, 40, 100, 300, 500]),
                'batch_size': trial.suggest_categorical("batch_size", [512])
            }
        elif model_name == 'NMF3':
            params = {
                # 'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-1),
                'learning_rate': trial.suggest_categorical('learning_rate', [0.001, 0.01, 0.1, 1]),
                'optimizer': trial.suggest_categorical("optimizer", ["SGD"]),
                # 'optimizer': trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"]),
                # 'latent_dim': trial.suggest_int("latent_dim", 10, 20),
                'latent_dim': trial.suggest_categorical("latent_dim", [10, 40, 100, 300, 500]),
                'batch_size': trial.suggest_categorical("batch_size", [512])
            }

        # params = params_dict[dataset_name][model_name]  # Get the relevant params range by the dataset and model
        dl_train, dl_valid, _ = get_data(
            dataset_name=dataset_name,
            batch_size=params['batch_size'], 
            device=device
        )
        model = get_model(model_name, params, dl_train)  # Build model
        optimizer = getattr(optim, params['optimizer'])(model.parameters(), lr=params['learning_rate'])  # Instantiate optimizer
        criterion = NON_ZERO_RMSELoss()
        # criterion = RMSELoss()
        valid_loss, _ = train(model_name, model, optimizer, criterion, max_epochs, dl_train, dl_valid, device, dataset_name)  # Train the model and calc the validation loss

        return valid_loss

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.RandomSampler())   # Build the study
    # Optimize (the actual tuning process)
    study.optimize(objective, n_trials=n_trials)
    # Extract the trials information as Pandas DataFrame
    df_trials_results = study.trials_dataframe()

    return study, df_trials_results


def test(model_name, dataset_name, best_params, max_epochs, device):
    """
    After we optimized and choosed the best hyperparameters for the model we want to prepare it for predicting the test set.
    - Use the best hyperparameters to build the final model
    - Train the final model on the train data set
    - Test it against the test set for final results
    """
    dl_train, _, dl_test = get_data(
        dataset_name=dataset_name, batch_size=best_params['batch_size'], device=device)
    model = get_model(model_name, best_params, dl_train)  # Build model
    optimizer = getattr(optim, best_params['optimizer'])(model.parameters(), lr=best_params['learning_rate'])  # Instantiate optimizer
    # Train the model on the full_train (train+valid) set and calc the test loss
    test_loss, final_model = train(model_name, model, optimizer, max_epochs, dl_train, dl_test, device, dataset_name)

    return test_loss, final_model


# Only for testing
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset_name = ''
    max_epochs = 2
    n_trials = 2
    model_name = 'NMF'

    study_ml, df_tuning_results = tune_params(
        model_name=model_name,
        dataset_name=dataset_name,
        max_epochs=max_epochs,
        n_trials=n_trials,
        device=device
    )

    best_params = study_ml.best_params
    print(f'Best params: {best_params}')
    print(df_tuning_results.sort_values(by='value').head(15))

    # Test
    test_loss, final_model = test(
        model_name=model_name,
        dataset_name=dataset_name,
        best_params=best_params,
        max_epochs=max_epochs,
        device=device
    )
    print(f'Final test loss = {test_loss}')
