import pandas as pd
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

    all_gens, all_spots, y_list, y_pred_list = [], [], [], []

    with torch.no_grad():
        total_loss = 0
        for batch in dl_test:
            gens, spots, y = batch
            gens.to(device)
            spots.to(device)
            y.to('cpu')
            y_pred = model(gens, spots).to('cpu')
            total_loss += loss_fn(y_pred, y)

            all_gens.extend(gens.tolist())
            all_spots.extend(spots.tolist())
            y_list.extend(y.tolist())
            y_pred_list.extend(y_pred.tolist())
            
    loss = total_loss / test_samples
    df_test_preds = pd.DataFrame({'gene': all_gens, 'spot': all_spots, 'y': y_list, 'y_pred': y_pred_list})
    return loss, df_test_preds


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
