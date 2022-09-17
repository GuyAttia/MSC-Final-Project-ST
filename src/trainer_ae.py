import torch
from os import path
from ignite.engine import Engine, Events
from ignite.metrics import Loss
from ignite.handlers import ModelCheckpoint, EarlyStopping
from ignite.contrib.handlers import TensorboardLogger, global_step_from_engine

from loss import *

# We used the Ignite package for smarter building of our trainers.
# This package provide built-in loggers and handlers for different actions.


def trainer_ae(model, optimizer, criterion, max_epochs, early_stopping, dl_train, dl_test, device, dataset_name, model_name):
    """
    Build a trainer for the AE model
    """
    model = model.to(device)
    # Define the loss function - AutoEnc data loaders are the genes/spots vectors, therefore contains a lot of non-relevant zeros,
    # so we used our custom RMSE which don't take them into account.
    # criterion = NON_ZERO_RMSELoss()

    def train_step(engine, batch):
        """
        Define the train step.
        Each sample in the batch is a user/item vector, which is also the target (what we want to reconstruct)
        """
        x = y = batch
        x.to(device)

        model.train()
        y_pred = model(x)
        loss = criterion(y_pred, y)

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
            x = y = batch
            x.to(device)
            y_pred = model(x)
            return y_pred, y

    train_evaluator = Engine(validation_step)
    val_evaluator = Engine(validation_step)

    # Generate training and validation evaluators to print results during running
    val_metrics = {
        "loss": Loss(criterion),
        'NonZeroRMSE': Loss(NON_ZERO_RMSELoss()),
    }

    # Attach metrics to the evaluators
    val_metrics['loss'].attach(train_evaluator, 'loss')
    val_metrics['loss'].attach(val_evaluator, 'loss')
    val_metrics['NonZeroRMSE'].attach(train_evaluator, 'NonZeroRMSE')
    val_metrics['NonZeroRMSE'].attach(val_evaluator, 'NonZeroRMSE')

    # Attach logger to print the training loss after each epoch
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(trainer):
        train_evaluator.run(dl_train)
        metrics = train_evaluator.state.metrics
        print(
            f"Training Results - Epoch[{trainer.state.epoch}] Avg loss: {metrics['loss']:.2f} |  Avg NonZeroRMSE: {metrics['NonZeroRMSE']:.2f}")

    # Attach logger to print the validation loss after each epoch
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(trainer):
        val_evaluator.run(dl_test)
        metrics = val_evaluator.state.metrics
        print(
            f"Validation Results - Epoch[{trainer.state.epoch}] Avg loss: {metrics['loss']:.2f}  |  Avg NonZeroRMSE: {metrics['NonZeroRMSE']:.2f}")

    # Model Checkpoint
    def score_function(engine):
        val_loss = engine.state.metrics['loss']
        return -val_loss

    checkpoint_dir = path.join("checkpoints", dataset_name)

    # Checkpoint to store n_saved best models wrt score function
    model_checkpoint = ModelCheckpoint(
        checkpoint_dir,
        n_saved=1,
        filename_prefix="best_ae",
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
        handler = EarlyStopping(patience=early_stopping,
                                score_function=score_function, trainer=trainer)
        val_evaluator.add_event_handler(Events.COMPLETED, handler)

    # Tensorboard logger - log the training and evaluation losses as function of the iterations & epochs
    tb_logger = TensorboardLogger(log_dir=path.join(
        'tb-logger', dataset_name, 'AE'))
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


# Only for testing
if __name__ == '__main__':
    from hyperparams_tuning import *
    from data import *

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset_name = 'Visium_Mouse_Olfactory_Bulb'

    max_epochs = 2
    early_stopping = 1
    model_name = 'AE'
    best_params = {
        'learning_rate': 0.001,
        'optimizer': "RMSprop",
        'latent_dim': 40,
        'batch_size': 128
    }
    dl_train, _, dl_test, df_spots_neighbors = get_data(model_name=model_name,
        dataset_name=dataset_name, batch_size=best_params['batch_size'], device=device)
    print('got data')
    model = get_model(model_name, best_params, dl_train)  # Build model
    optimizer = getattr(optim, best_params['optimizer'])(
        model.parameters(), lr=best_params['learning_rate'])  # Instantiate optimizer
    criterion = NON_ZERO_RMSELoss_Spatial_AE(df_spots_neighbors=df_spots_neighbors)
    test_loss = trainer_ae(
        model=model, 
        optimizer=optimizer, 
        criterion=criterion,
        max_epochs=max_epochs, 
        early_stopping=early_stopping,
        dl_train=dl_train,
        dl_test=dl_test, 
        device=device, 
        dataset_name=dataset_name,
        model_name=model_name
        )