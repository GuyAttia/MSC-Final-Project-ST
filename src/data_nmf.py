from os import path
from sklearn.model_selection import train_test_split
from torch import tensor
from torch.utils.data import DataLoader, Dataset
from scanpy_stlearn_loaders import StlearnLoader

import preprocessing

### ----------------------------------------------------------------------- Get Data -------------------------------------------------------------------------------- ###

def load_anndata_object():
    # Visium - Processed Visium Spatial Gene Expression data from 10x Genomics.
    obj = StlearnLoader().load_local_visum(path=path.join('/', 'data', 'Visium_Mouse_Olfactory_Bulb'),
                                      count_file='filtered_feature_bc_matrix.h5')
    obj.layers["raw_count"] = obj.X
    x = obj.X.toarray()
    n_spots, n_genes = x.shape
    print(f'# spots: {n_spots} | # genes: {n_genes}')
    return obj

def run_preprocessing(data_obj, apply_log=False):
    data_obj = preprocessing.filtering(data=data_obj, min_counts=10, min_cells=177)
    if apply_log:
        data_obj = preprocessing.log_transform(data=data_obj)
    
    df_expressions, oe_spots = preprocessing.encode_genes_and_spots(data_obj=data_obj)
    return data_obj, df_expressions, oe_spots

def get_expressions(apply_log):
    # Load the expressions data into a Pandas DataFrame
    data_obj = load_anndata_object()
    data_obj, df_expressions, oe_spots = run_preprocessing(data_obj=data_obj, apply_log=apply_log)

    print(f'Data shape: {df_expressions.shape}')
    print(f'Number of genes: {df_expressions["gene"].nunique()}')
    print(f'Number of spots: {df_expressions["spot"].nunique()}')
    return df_expressions, data_obj, oe_spots

def train_valid_test_split(df):
    """
    Split the data into train, validation, and test datasets.
    Creating the train dataset by replacing X% of the non-zero expressions into 0 while saving their index aside for the test evaluation.
    Doing the same on the created train dataset to create the validation dataset.
    """
    def set_random_zeros(df, random_size, dataset_mask):
        mask = df['expression'] > 0
        samp = df.loc[mask].sample(frac=random_size)

        mask_dataset = df.index.isin(samp.index)
        df.loc[mask_dataset, 'expression'] = 0
        df.loc[mask_dataset, 'dataset_mask'] = dataset_mask
        return df

    df['true_expression'] = df['expression']
    df = set_random_zeros(df, random_size=0.1, dataset_mask='test')
    df = set_random_zeros(df, random_size=0.1, dataset_mask='valid')

    df_train = df.loc[:, ['gene', 'spot', 'expression']]
    df_valid = df.loc[df['dataset_mask'] == 'valid', ['gene', 'spot', 'true_expression']].rename(columns={'true_expression': 'expression'})
    df_test = df.loc[df['dataset_mask'] == 'test', ['gene', 'spot', 'true_expression']].rename(columns={'true_expression': 'expression'})
    del df

    print(f'Train shape:{df_train.shape}')
    print(f'Valid shape:{df_valid.shape}')
    print(f'Test shape:{df_test.shape}')
    return df_train, df_valid, df_test

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


def main(apply_log, batch_size, device):
    expressions, _, _ = get_expressions(apply_log)

    df_train, df_valid, df_test = train_valid_test_split(df=expressions)

    print('Start creating the DataSets')
    ds_train = ExpressionDataset(df=df_train, device=device)
    ds_valid = ExpressionDataset(df=df_valid, device=device)
    ds_test = ExpressionDataset(df=df_test, device=device)

    print('Start creating the DataLoaders')
    dl_train = DataLoader(dataset=ds_train, batch_size=batch_size, shuffle=True)
    dl_valid = DataLoader(dataset=ds_valid, batch_size=batch_size, shuffle=True)
    dl_test = DataLoader(dataset=ds_test, batch_size=batch_size, shuffle=True)

    print('Finish loading the data')
    return dl_train, dl_valid, dl_test, None


# For testing only
if __name__ == '__main__':
    apply_log = False
    dl_train, dl_valid, dl_test, df_spots_neighbors = main(
        apply_log=apply_log, 
        batch_size=128, 
        device='cpu'
    )
