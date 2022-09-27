import pandas as pd
import torch

from loss import *

def test(model, criterion, dl_test, device):
    """
    Test a trained model
    """
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
            total_loss += criterion(y_pred, y)

            all_gens.extend(gens.tolist())
            all_spots.extend(spots.tolist())
            y_list.extend(y.tolist())
            y_pred_list.extend(y_pred.tolist())
            
    loss = total_loss / test_samples
    df_test_preds = pd.DataFrame({'gene': all_gens, 'spot': all_spots, 'y': y_list, 'y_pred': y_pred_list})
    return loss, df_test_preds


# Only for testing
if __name__ == '__main__':
    import data_nmf as get_data
    from models import get_model
    
    apply_log = False
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
    criterion = NON_ZERO_RMSELoss()
    test_loss = test(model, criterion, dl_test, device)
