import torch

from loss import *

def tester(model, dl_test, device):
    """
    Test a trained model
    """
    loss_fn = RMSELoss()
    test_samples = len(dl_test)

    model = model.to(device)
    model.eval()

    with torch.no_grad():
        total_loss = 0
        for batch in dl_test:
            gens, spots, y = batch
            gens.to(device)
            spots.to(device)
            y.to('cpu')
            y_pred = model(gens, spots).to('cpu')
            total_loss += loss_fn(y_pred, y)
            
    loss = total_loss / test_samples
    return loss


# Only for testing
if __name__ == '__main__':
    from data import get_data
    from models import get_model

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset_name = 'V1_Human_Lymph_Node'
    model_name = 'NMF'
    best_params = {
        'learning_rate': 0.001,
        'optimizer': "RMSprop",
        'latent_dim': 20,
        'batch_size': 512
    }

    dl_train, _, dl_test = get_data(dataset_name=dataset_name, batch_size=best_params['batch_size'], device=device)  # Get data
    model = get_model(model_name, best_params, dl_train)  # Build model
    test_loss = tester(model, dl_test, device)
