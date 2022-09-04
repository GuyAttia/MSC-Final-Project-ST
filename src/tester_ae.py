import pandas as pd
import torch

from loss import *

def tester_ae(model, dl_test, device, loss_fn):
    """
    Test a trained model
    """
    test_samples = len(dl_test)

    model = model.to(device)
    model.eval()

    # all_gens, all_spots, y_list, y_pred_list = [], [], [], []

    with torch.no_grad():
        total_loss = 0

        for batch in dl_test:
            x = y = batch
            x.to(device)
            y_pred = model(x)
            total_loss += loss_fn(y_pred, y)

            # all_gens.extend(x.tolist())
            # y_list.extend(y.tolist())
            # y_pred_list.extend(y_pred.tolist())
            
    loss = total_loss / test_samples
    # df_test_preds = pd.DataFrame({'gene': all_gens, 'spot': all_spots, 'y': y_list, 'y_pred': y_pred_list})
    return loss, pd.DataFrame() #df_test_preds


# Only for testing
if __name__ == '__main__':
    from data import get_data
    from models import get_model

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset_name = 'Visium_Mouse_Olfactory_Bulb'
    model_name = 'AE'
    best_params = {
        'learning_rate': 0.001,
        'optimizer': "RMSprop",
        'latent_dim': 20,
        'batch_size': 512
    }

    dl_train, _, dl_test = get_data(dataset_name=dataset_name, batch_size=best_params['batch_size'], device=device)  # Get data
    model = get_model(model_name, best_params, dl_train)  # Build model
    test_loss = tester_ae(model, dl_test, device)
