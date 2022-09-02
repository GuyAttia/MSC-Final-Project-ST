import stlearn as st
import scanpy as sc
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from torch import tensor
from torch.utils.data import DataLoader, Dataset

import preprocessing

### ----------------------------------------------------------------------- Get Data -------------------------------------------------------------------------------- ###

def load_anndata_object(dataset_name):
    # Visium - Processed Visium Spatial Gene Expression data from 10x Genomics.
    sc_anndata_obj = sc.datasets.visium_sge(sample_id=dataset_name)
    st_anndata_obj = st.convert_scanpy(sc_anndata_obj)
    return st_anndata_obj

def run_preprocessing(data_obj):
    data = preprocessing.filtering(data=data_obj, min_counts=15000, min_cells=1000)
    return data

def get_expressions(dataset_name='V1_Human_Lymph_Node'):
    # Load the expressions data into a Pandas DataFrame
    st_anndata_obj = load_anndata_object(dataset_name=dataset_name)
    st_anndata_obj_clean = run_preprocessing(data_obj=st_anndata_obj)

    spots = st_anndata_obj_clean.obs.index.values
    genes = st_anndata_obj_clean.var.index.values
    df_expressions_matrix = pd.DataFrame.sparse.from_spmatrix(st_anndata_obj_clean.X, columns=genes, index=spots)
    df_expressions = df_expressions_matrix.stack().reset_index()
    df_expressions.columns = ['spot', 'gene', 'expression']

    # Ordinal encoding the genes and spots for supported type
    oe = OrdinalEncoder()
    df_expressions[['spot', 'gene']] = oe.fit_transform(df_expressions[['spot', 'gene']].values)
    df_expressions[['spot', 'gene']] = df_expressions[['spot', 'gene']].astype(int)

    print(f'Data shape: {df_expressions.shape}')
    print(f'Number of genes: {df_expressions["gene"].nunique()}')
    print(f'Number of spots: {df_expressions["spot"].nunique()}')
    return df_expressions

### ----------------------------------------------------------------------- Split Data -------------------------------------------------------------------------------- ###

def train_valid_test_split(df):
    """
    Split the data into train, validation, test, and full-train sets
    """
    df_full_train, df_test = train_test_split(df, test_size=0.10)
    df_train, df_valid = train_test_split(df_full_train, test_size=0.10)
    print(f'Split to train, valid, full-train and test:\nTrain shape:{df_train.shape}\nValid shape:{df_valid.shape}\nTest shape:{df_test.shape}')

    return df_train, df_valid, df_test

### ----------------------------------------------------------------------- MF -------------------------------------------------------------------------------- ###

class ExpressionDataset(Dataset):
    """
    Generate expression dataset to use in the our models, where each sample should be a tuple of (gene, spot, expression)
    """

    def __init__(self, df, device):
        self.num_samples = len(df)
        self.genes = tensor(df['gene'].values).to(device)
        self.spots = tensor(df['spot'].values).to(device)
        self.labels = tensor(df['expression'].sparse.to_dense().values)
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


def dataloaders(dataset_name, device, batch_size: int = 128):
    """
    Generate the DataLoader objects for the models with the defined batch size
    """
    expressions = get_expressions(dataset_name=dataset_name)

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
    return dl_train, dl_valid, dl_test


def get_data(dataset_name, batch_size, device):
    """
    Get the train, validation, test, and full train data loaders for the relevant dataset
    """
    dl_train, dl_valid, dl_test = dataloaders(dataset_name=dataset_name, batch_size=batch_size, device=device)
    return dl_train, dl_valid, dl_test


# For testing only
if __name__ == '__main__':
    dataset_name = 'V1_Human_Lymph_Node'
    dl_train, dl_valid, dl_test = get_data(dataset_name=dataset_name, batch_size=128, device='cpu')
