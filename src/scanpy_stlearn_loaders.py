from os import path
from string import ascii_uppercase
import stlearn as st
import scanpy as sc
import numpy as np
import pandas as pd


class ScanpyLoader:
    def __init__(self) -> None:
        pass

    def load_visium_dataset(self, sample_id='V1_Human_Lymph_Node'):
        # Visium - Processed Visium Spatial Gene Expression data from 10x Genomics.
        return sc.datasets.visium_sge(sample_id=sample_id)

    def load_local_visum(self, path_='', count_file='filtered_feature_bc_matrix.h5'):
        # 10x-Genomics-formatted visum dataset
        return sc.read_visium(path=path_, genome=None, count_file=count_file, library_id=None, load_images=True, source_image_path=None)
        
    def generate_sample_data(self):
        # Generate sample data
        n_obs = 1000 # number of observations
        obs = pd.DataFrame({'time': np.random.choice(['day 1', 'day 2', 'day 4', 'day 8'], n_obs)})
        var_names = [i*letter for i in range(1, 10) for letter in ascii_uppercase]
        n_vars = len(var_names)  # number of variables
        var = pd.DataFrame({'a':[0]*len(var_names)}, index=var_names)  # dataframe for annotating the variables
        X = np.arange(n_obs*n_vars).reshape(n_obs, n_vars)  # the data matrix of shape n_obs x n_vars

        # Todo: Build uns ourselfs
        uns = None

        # Create the AnnData object
        return sc.AnnData(X, obs=obs, var=var, uns=uns, dtype='int32')


class StlearnLoader:
    def __init__(self) -> None:
        pass

    def load_from_scanpy_object(self):
        scanpy_loader = ScanpyLoader()
        sc_data = scanpy_loader.load_visium_dataset()
        return st.convert_scanpy(sc_data)

    def load_from_st_datasets(self):
        # Visium - Processed Visium Spatial Gene Expression data from 10x Genomics.
        return st.datasets.example_bcba()

    def load_local_visum(self, path='data\\\V1_Human_Lymph_Node\\', count_file='filtered_feature_bc_matrix.h5'):
        # In addition to reading regular 10x output, this looks for the spatial folder and loads images, 
        # coordinates and scale factors.
        return st.Read10X(path, count_file=count_file, load_images=True)
        
    def generate_sample_data(self):
        # Generate sample data
        n_obs = 1000 # number of observations
        obs = pd.DataFrame({'time': np.random.choice(['day 1', 'day 2', 'day 4', 'day 8'], n_obs)})
        var_names = [i*letter for i in range(1, 10) for letter in ascii_uppercase]
        var = pd.DataFrame({'a':[0]*len(var_names)}, index=var_names)  # dataframe for annotating the variables
        n_vars = len(var_names)  # number of variables
        X = np.arange(n_obs*n_vars).reshape(n_obs, n_vars)  # the data matrix of shape n_obs x n_vars

        # count – Pandas Dataframe of count matrix with rows as barcodes and columns as gene names
        df_count = pd.DataFrame(X, columns=var_names, index=obs)
        # spatial – Pandas Dataframe of spatial location of cells/spots.
        df_spatial = pd.DataFrame({'imagecol': np.random.randn(n_obs),
                                'imagerow': np.random.randn(n_obs)})

        return st.create_stlearn(count=df_count, spatial=df_spatial, library_id="Sample_test")
