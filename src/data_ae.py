from os import path, mkdir
import numpy as np
import pickle
import pandas as pd
import torch
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

def run_preprocessing(data_obj, min_counts, min_cells, apply_log=False):
    data_obj = preprocessing.filtering(data=data_obj, min_counts=min_counts, min_cells=min_cells)
    if apply_log:
        data_obj = preprocessing.log_transform(data=data_obj)
    
    df_expressions, oe_spots, oe_genes = preprocessing.encode_genes_and_spots(data_obj=data_obj)
    return data_obj, df_expressions, oe_spots, oe_genes

def get_expressions(min_counts, min_cells, apply_log):
    # Load the expressions data into a Pandas DataFrame
    data_obj = load_anndata_object()
    data_obj, df_expressions, oe_spots, oe_genes = run_preprocessing(data_obj=data_obj, min_counts=min_counts, min_cells=min_cells, apply_log=apply_log)

    print(f'Data shape: {df_expressions.shape}')
    print(f'Number of genes: {df_expressions["gene"].nunique()}')
    print(f'Number of spots: {df_expressions["spot"].nunique()}')
    return df_expressions, data_obj, oe_spots, oe_genes

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
    df = set_random_zeros(df, random_size=0.2, dataset_mask='test')
    df = set_random_zeros(df, random_size=0.2, dataset_mask='valid')

    df_train = df.loc[:, ['gene', 'spot', 'expression']]
    
    mask_valid = df['dataset_mask'] == 'valid'
    df_valid = df.copy()
    df_valid.loc[mask_valid, 'expression'] = df_valid.loc[mask_valid, 'true_expression']
    df_valid.loc[mask_valid, 'dataset_mask'] = 1
    df_valid.loc[~mask_valid, 'dataset_mask'] = 0
    df_valid.drop(columns=['true_expression'], inplace=True)

    mask_test = df['dataset_mask'] == 'test'
    df_test = df.copy()
    df_test.loc[mask_test, 'expression'] = df_test.loc[mask_test, 'true_expression']
    df_test.loc[mask_test, 'dataset_mask'] = 1
    df_test.loc[~mask_test, 'dataset_mask'] = 0
    df_test.drop(columns=['true_expression'], inplace=True)

    del df

    print(f'Train shape:{df_train.shape}')
    print(f'Valid shape:{df_valid.shape}')
    print(f'Test shape:{df_test.shape}')
    return df_train, df_valid, df_test

def create_matrix(expressions, df_train, df_valid, df_test):
    """
    Change the expressions DataFrames into expressions matrixes.
    This method is used for AE where we want to get gene vector as an input.
    """
    # Pivot the train, valid and test DataFrames.
    expressions_pivot = expressions.pivot(
        index='gene', columns='spot', values='expression').fillna(0)
    train_pivot = df_train.pivot(
        index='gene', columns='spot', values='expression').fillna(0)
    valid_pivot = df_valid.pivot(
        index='gene', columns='spot', values='expression').fillna(0)
    test_pivot = df_test.pivot(
        index='gene', columns='spot', values='expression').fillna(0)
    
    # Pivot the valid and test DataFrames for masking
    valid_mask_pivot = df_valid.pivot(
        index='gene', columns='spot', values='dataset_mask').fillna(0)
    test_mask_pivot = df_test.pivot(
        index='gene', columns='spot', values='dataset_mask').fillna(0)
    
    del expressions, df_train, df_valid, df_test

    # Create empty matrixes in the shape of the full data
    expressions_train = pd.DataFrame(np.zeros(
        expressions_pivot.shape), index=expressions_pivot.index, columns=expressions_pivot.columns)
    expressions_mask_train = pd.DataFrame(np.ones(
        expressions_pivot.shape), index=expressions_pivot.index, columns=expressions_pivot.columns)
    
    expressions_valid = expressions_train.copy()
    expressions_mask_valid = expressions_train.copy()
    
    expressions_test = expressions_train.copy()
    expressions_mask_test = expressions_train.copy()

    # Fill the relevant expressions by the correct expression matrix
    expressions_train.loc[train_pivot.index,
                      train_pivot.columns] = train_pivot.values
    expressions_valid.loc[valid_pivot.index,
                      valid_pivot.columns] = valid_pivot.values
    expressions_test.loc[test_pivot.index, test_pivot.columns] = test_pivot.values
    expressions_mask_valid.loc[valid_mask_pivot.index, valid_mask_pivot.columns] = valid_mask_pivot.values
    expressions_mask_test.loc[test_mask_pivot.index, test_mask_pivot.columns] = test_mask_pivot.values

    return expressions_train, expressions_valid, expressions_test, expressions_mask_train, expressions_mask_valid, expressions_mask_test

class VectorsDataSet(Dataset):
    """
    Generate vectors dataset to use in the AE model, where each sample should be a gene vector
    """

    def __init__(self, expressions_matrix, expressions_mask, device) -> None:
        self.data = tensor(expressions_matrix.values).float().to(device)
        self.mask = tensor(expressions_mask.values, dtype=torch.bool).to(device)

    def __getitem__(self, index: int):
        vec = self.data[index]
        vec_mask = self.mask[index]
        return vec, vec_mask

    def __len__(self) -> int:
        return self.data.shape[0]

    def get_all_data(self):
        return self.data, self.mask


def main(min_counts, min_cells, apply_log, batch_size, device):
    if not path.isdir(path.join('/', 'data', 'AE')):
        mkdir(path.join('/', 'data', 'AE'))
        
    if path.isfile(path.join('/', 'data', 'AE', 'dl_train.pth')):
        dl_train = torch.load(path.join('/', 'data', 'AE', 'dl_train.pth'))
        dl_valid = torch.load(path.join('/', 'data', 'AE', 'dl_valid.pth'))
        dl_test = torch.load(path.join('/', 'data', 'AE', 'dl_test.pth'))
        df_spots_neighbors = pd.read_csv(path.join('/', 'data', 'AE', 'spots_neighbors.csv'))
    else:
        expressions, obj, oe_spots, oe_genes = get_expressions(min_counts, min_cells, apply_log)
        with open(path.join('/', 'data', 'AE', 'spots_encoder.pkl'), 'wb') as f:
            pickle.dump(oe_spots, f)
        with open(path.join('/', 'data', 'AE', 'genes_encoder.pkl'), 'wb') as f:
            pickle.dump(oe_genes, f)
        
        df_spots_neighbors = find_neighbors(obj, oe_spots)
        df_spots_neighbors.to_csv(path.join('/', 'data', 'AE', 'spots_neighbors.csv'), index=False)

        df_train, df_valid, df_test = train_valid_test_split(df=expressions)

        expressions_train, expressions_valid, expressions_test, expressions_mask_train, expressions_mask_valid, expressions_mask_test = create_matrix(
            expressions, df_train, df_valid, df_test)

        ds_train = VectorsDataSet(expressions_matrix=expressions_train, expressions_mask=expressions_mask_train, device=device)
        ds_valid = VectorsDataSet(expressions_matrix=expressions_valid, expressions_mask=expressions_mask_valid, device=device)
        ds_test = VectorsDataSet(expressions_matrix=expressions_test, expressions_mask=expressions_mask_test, device=device)

        dl_train = DataLoader(dataset=ds_train, batch_size=batch_size, shuffle=True)
        dl_valid = DataLoader(dataset=ds_valid, batch_size=batch_size, shuffle=True)
        dl_test = DataLoader(dataset=ds_test, batch_size=batch_size, shuffle=True)

        # Saving DataLoaders
        torch.save(dl_train, path.join('/', 'data', 'AE', 'dl_train.pth'))
        torch.save(dl_valid, path.join('/', 'data', 'AE', 'dl_valid.pth'))
        torch.save(dl_test, path.join('/', 'data', 'AE', 'dl_test.pth'))

    print('Finish loading the data')
    return dl_train, dl_valid, dl_test, df_spots_neighbors

# For testing only
if __name__ == '__main__':
    min_counts = 500
    min_cells = 177
    apply_log = False
    batch_size = 128
    device = 'cpu'

    dl_train, dl_valid, dl_test, df_spots_neighbors = main(
        min_counts=min_counts,
        min_cells=min_cells,
        apply_log=apply_log, 
        batch_size=batch_size, 
        device=device
    )