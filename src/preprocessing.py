import stlearn as st
from sklearn.preprocessing import OrdinalEncoder
import pandas as pd


def filtering(data, min_counts=None, max_counts=None, min_cells=None, max_cells=None):
    # Filter genes - Keep genes that have (Choose only 1):
    # - at least min_counts counts or 
    # - are expressed in at least min_cells cells or 
    # - have at most max_counts counts or 
    # - are expressed in at most max_cells cells. 
    if min_counts:
        # By minimum counts
        st.pp.filter_genes(data, min_counts=min_counts, inplace=True)
    if max_counts:
        # By maximum counts
        st.pp.filter_genes(data, max_counts=max_counts, inplace=True)
    if min_cells:
        # By minimum cells
        st.pp.filter_genes(data, min_cells=min_cells, inplace=True)
    if max_cells:
        # By maximum cells
        st.pp.filter_genes(data, max_cells=max_cells, inplace=True)
        
    print(f'New shape after filtering: {data.X.shape}')
    return data

def log_transform(data):
    # Log transform
    # Logarithmize the data matrix. Computes 𝑋=log(𝑋+1), where 𝑙𝑜𝑔 denotes the natural logarithm unless a different base is given.
    st.pp.log1p(data, copy=False)
    return data

def normalize(data):
    # Normalization
    # Normalize counts per cell. 
        # If choosing target_sum=1e6, this is CPM normalization. 
        # If exclude_highly_expressed=True, very highly expressed genes are excluded from the computation of the normalization factor (size factor) for each cell. 
    st.pp.normalize_total(data, inplace=True)
    return data

def scale(data):
    # Scaling
    # Scale data to unit variance and zero mean.
    st.pp.scale(data, copy=False)
    return data

def image_tiling(data):
    # Tiling H&E images to small tiles based on spot spatial location
    st.pp.tiling(data, inplace=True)
    return data

def extract_images_latent_features(data):
    # Extract latent morphological features from H&E images using pre-trained convolutional neural network base
    data = image_tiling(data)
    st.pp.extract_feature(data, inplace=True)
    return data

def run_pca(data):
    # run PCA for gene expression data
    st.em.run_pca(data, n_comps=50)
    return data

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