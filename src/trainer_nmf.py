from os import path
import torch
import torch.optim as optim
from ignite.contrib.handlers import TensorboardLogger, global_step_from_engine
from ignite.engine import Engine, Events
from ignite.handlers import EarlyStopping, ModelCheckpoint
from ignite.metrics import Loss

from loss import *
from models import get_model

def train(model, optimizer, criterion, max_epochs, early_stopping, dl_train, dl_test, device, model_name):
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
    checkpoint_dir = "checkpoints"
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
        log_dir=path.join('logs', model_name))
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
    import data_nmf as get_data
    
    apply_log=False
    batch_size = 128
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_name = 'NMF'
    max_epochs = 2
    early_stopping = 15
    model_params = {
        'learning_rate': 0.01,
        'optimizer': "RMSprop",
        'latent_dim': 40,
        'batch_size': 128
    }

    dl_train, dl_valid, dl_test, _ = get_data.main(apply_log=apply_log, batch_size=batch_size, device=device)
    model = get_model(model_name, model_params, dl_train)
    optimizer = getattr(optim, model_params['optimizer'])(model.parameters(), lr=model_params['learning_rate'])
    criterion = NON_ZERO_RMSELoss()
    # criterion = RMSELoss()
    
    test_loss = train(
                    model=model, 
                    optimizer=optimizer, 
                    criterion=criterion,
                    max_epochs=max_epochs, 
                    early_stopping=early_stopping, 
                    dl_train=dl_train, 
                    dl_test=dl_test, 
                    device=device,
                    model_name=model_name
                )