from os import path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from torch import tensor
from torch.utils.data import DataLoader, Dataset
from scanpy_stlearn_loaders import StlearnLoader

import preprocessing

### ----------------------------------------------------------------------- Get Data -------------------------------------------------------------------------------- ###

def load_anndata_object(dataset_name):
    # Visium - Processed Visium Spatial Gene Expression data from 10x Genomics.
    obj = StlearnLoader().load_local_visum(path=path.join('/', 'data', dataset_name),
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
    return data_obj

def encode_genes_and_spots(data_obj):
    spots_values = data_obj.obs.index.values
    genes_values = data_obj.var.index.values
    x = data_obj.X.toarray()
    df_expressions_matrix = pd.DataFrame(x, columns=genes_values, index=spots_values).T
    df_expressions = df_expressions_matrix.stack().reset_index()
    df_expressions.columns = ['gene', 'spot', 'expression']
    
    # Ordinal encoding the genes and spots for supported type
    oe_genes = OrdinalEncoder()
    df_expressions[['gene']] = oe_genes.fit_transform(df_expressions[['gene']].values)
    oe_spots = OrdinalEncoder()
    df_expressions[['spot']] = oe_spots.fit_transform(df_expressions[['spot']].values)

    df_expressions[['spot', 'gene']] = df_expressions[['spot', 'gene']].astype(int)
    return df_expressions, oe_spots

def get_expressions(dataset_name):
    # Load the expressions data into a Pandas DataFrame
    data_obj = load_anndata_object(dataset_name=dataset_name)
    data_obj = run_preprocessing(data_obj=data_obj, apply_log=True)
    df_expressions, oe_spots = encode_genes_and_spots(data_obj)

    print(f'Data shape: {df_expressions.shape}')
    print(f'Number of genes: {df_expressions["gene"].nunique()}')
    print(f'Number of spots: {df_expressions["spot"].nunique()}')
    return df_expressions, data_obj, oe_spots

### ----------------------------------------------------------------------- Split Data -------------------------------------------------------------------------------- ###

def train_valid_test_split(df):
    """
    Split the data into train, validation, and test sets
    """
    df_full_train, df_test = train_test_split(df, test_size=0.10)
    df_train, df_valid = train_test_split(df_full_train, test_size=0.10)
    print(f'Split to train, valid, full-train and test:\nTrain shape:{df_train.shape}\nValid shape:{df_valid.shape}\nTest shape:{df_test.shape}')

    return df_train, df_valid, df_test


def set_random_zeros(df, random_size=0.1):
    mask = df['expression'] > 0
    samp = df.loc[mask].sample(frac=random_size)

    df.loc[df.index.isin(samp.index), 'expression'] = 0   
    return df

def train_valid_test_split_ae(df):
    """
    Split the data into train, validation, and test sets
    - The test is the original data set
    - The validation and the test are with 10% of random zeros (not real ones)
    """
    df_test = df.copy()
    df_train = set_random_zeros(df, random_size=0.1)
    df_valid = set_random_zeros(df_train, random_size=0.1)

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
    expressions, _, _ = get_expressions(dataset_name=dataset_name)

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


# ----------------------------------------------------- AE -----------------------------------------------------
def create_matrix(expressions, df_train, df_valid, df_test):
    """
    Change the expressions DataFrames into expressions matrixes.
    This method is used for AE where we want to get gene vector as an input.
    """
    # Pivot the full-data, train, valid and test DataFrames.
    expressions_pivot = expressions.pivot(
        index='gene', columns='spot', values='expression').fillna(0)
    train_pivot = df_train.pivot(
        index='gene', columns='spot', values='expression').fillna(0)
    valid_pivot = df_valid.pivot(
        index='gene', columns='spot', values='expression').fillna(0)
    test_pivot = df_test.pivot(
        index='gene', columns='spot', values='expression').fillna(0)
    del expressions, df_train, df_valid, df_test

    # Create empty matrixes in the shape of the full data
    expressions_train = pd.DataFrame(np.zeros(
        expressions_pivot.shape), index=expressions_pivot.index, columns=expressions_pivot.columns)
    expressions_valid = expressions_train.copy()
    expressions_test = expressions_train.copy()

    # Fill the relevant expressions by the correct expression matrix
    expressions_train.loc[train_pivot.index,
                      train_pivot.columns] = train_pivot.values
    expressions_valid.loc[valid_pivot.index,
                      valid_pivot.columns] = valid_pivot.values
    expressions_test.loc[test_pivot.index, test_pivot.columns] = test_pivot.values

    return expressions_train, expressions_valid, expressions_test

class VectorsDataSet(Dataset):
    """
    Generate vectors dataset to use in the AE model, where each sample should be a gene vector
    """

    def __init__(self, expressions_matrix, device) -> None:
        self.data = tensor(expressions_matrix.values).float().to(device)

    def __getitem__(self, index: int):
        vec = self.data[index]
        return vec

    def __len__(self) -> int:
        return self.data.shape[0]

    def get_all_data(self):
        return self.data


def dataloaders_ae(dataset_name, device, batch_size: int = 128):
    """
    Generate the DataLoader objects for AE model with the defined batch size
    """
    expressions, obj, oe_spots = get_expressions(dataset_name=dataset_name)
    df_spots_neighbors = find_neighbors(obj, oe_spots)
    df_train, df_valid, df_test = train_valid_test_split_ae(df=expressions)

    expressions_train, expressions_valid, expressions_test = create_matrix(
        expressions, df_train, df_valid, df_test)

    ds_train = VectorsDataSet(expressions_matrix=expressions_train, device=device)
    ds_valid = VectorsDataSet(expressions_matrix=expressions_valid, device=device)
    ds_test = VectorsDataSet(expressions_matrix=expressions_test, device=device)

    dl_train = DataLoader(dataset=ds_train, batch_size=batch_size, shuffle=True)
    dl_valid = DataLoader(dataset=ds_valid, batch_size=batch_size, shuffle=True)
    dl_test = DataLoader(dataset=ds_test, batch_size=batch_size, shuffle=True)
    
    return dl_train, dl_valid, dl_test, df_spots_neighbors


def find_neighbors(obj, oe_spots):
    df_spots = obj.obs[['array_row', 'array_col']].reset_index(drop=False).rename(columns={'index': 'spot_name'})
    df_spots[['spot_encoding']] = oe_spots.transform(df_spots[['spot_name']].values)
    df_spots['spot_encoding'] = df_spots['spot_encoding'].astype(int)

    def find_spots_neighbors(spot_encoding):   
        # Get spot location
        mask = df_spots['spot_encoding'] == spot_encoding
        row, col = df_spots.loc[mask, ['array_row', 'array_col']].values[0]
        
        # Get 1st degree neighbors
        first_degree_neighbors = []
        first_degree_tuples = [(2, 0), (-2, 0), (0, 2), (0, -2)]
        for tup in first_degree_tuples:
            mask = (df_spots['array_row'] == row+tup[0]) & (df_spots['array_col'] == col+tup[1])
            neighbor_encoding = df_spots.loc[mask, 'spot_encoding'].values
            if neighbor_encoding.size > 0:
                first_degree_neighbors.append(neighbor_encoding[0])
        
        # Get 1st degree neighbors
        second_degree_neighbors = []
        second_degree_tuples = [(4, 0), (-4, 0), (0, 4), (0, -4), (1, 1), (-1, -1), (1, -1), (-1, 1)]
        for tup in second_degree_tuples:
            mask = (df_spots['array_row'] == (row+tup[0])) & (df_spots['array_col'] == (col+tup[1]))
            neighbor_encoding = df_spots.loc[mask, 'spot_encoding'].values
            if neighbor_encoding.size > 0:
                second_degree_neighbors.append(neighbor_encoding[0])
    
        return first_degree_neighbors, second_degree_neighbors

    # Create spots-spots matrix
    n_spots = df_spots.shape[0]
    df_spots_neighbors = pd.DataFrame(np.zeros(shape=(n_spots, n_spots)))

    for spot_encoding in range(n_spots):
        first_degree_neighbors, second_degree_neighbors = find_spots_neighbors(spot_encoding=spot_encoding)

        if first_degree_neighbors:
            df_spots_neighbors.iloc[spot_encoding, first_degree_neighbors] = 1
        
        if second_degree_neighbors:
            df_spots_neighbors.iloc[spot_encoding].iloc[second_degree_neighbors] = 2

    return df_spots_neighbors


def get_data(model_name, dataset_name, batch_size, device):
    """
    Get the train, validation, test, and full train data loaders for the relevant dataset
    """
    if model_name == 'MF':
        dl_train, dl_valid, dl_test = dataloaders(dataset_name=dataset_name, batch_size=batch_size, device=device)
        df_spots_neighbors = None
    elif model_name == 'AE':
        dl_train, dl_valid, dl_test, df_spots_neighbors = dataloaders_ae(dataset_name=dataset_name, batch_size=batch_size, device=device)
    return dl_train, dl_valid, dl_test, df_spots_neighbors


# For testing only
if __name__ == '__main__':
    dataset_name = 'Visium_Mouse_Olfactory_Bulb'
    model_name = 'AE'
    dl_train, dl_valid, dl_test, df_spots_neighbors = get_data(model_name=model_name, dataset_name=dataset_name, batch_size=128, device='cpu')
