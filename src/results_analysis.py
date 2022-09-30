import matplotlib.pyplot as plt 
import seaborn as sns
import stlearn as st


def error_distribution(df_preds, mask=None):
    if mask is not None:
        df_preds = df_preds.loc[mask]

    print(df_preds['error'].describe())

    df_preds['error'].abs().plot.hist(logy=True, figsize=(15, 6), bins=30)
    plt.title('Errors Historgram')
    plt.show()


def spots_error_distribution(df_preds):
    print(df_preds.groupby('spot')['error'].mean().describe())


def genes_error_distribution(df_preds):
    print(df_preds.groupby('gene')['error'].mean().describe())


def error_heat_map(df_preds, vmin=None, vmax=None):
    df_pivot_error = df_preds.pivot(index='spot', columns='gene', values='error').fillna(0)
    print(df_pivot_error.shape)

    f = plt.figure(figsize=(15, 15))
    sns.heatmap(data=df_pivot_error.abs(), vmin=vmin, vmax=vmax)
    plt.title('Errors Heat Map')
    plt.show()

def pca_kmeans_clustering(data_obj, title):
    st.pp.normalize_total(data_obj)
    st.pp.log1p(data_obj)

    # run PCA for gene expression data
    st.em.run_pca(data_obj, n_comps=50)
    # K-means clustering
    st.tl.clustering.kmeans(data_obj, n_clusters=7, use_data="X_pca", key_added="X_pca_kmeans")
    
    colors_map_dict = {
    '#1f77b4': 1, # Blue
    '#f87f13': 0, # Orange
    '#359c62': 3, # Green
    '#d32929': 4, # Red
    '#69308e': 5, # Purple
    '#8c564c': 6, # Brown
    '#f33ca9': 2  # Pink
    }
    clusters_colors = [c[0] for c in sorted(colors_map_dict.items(), key=lambda i: i[1])]
    
    data_obj.uns['X_pca_kmeans_colors'] = clusters_colors
    
    f = plt.figure()
    st.pl.cluster_plot(data_obj, use_label="X_pca_kmeans")
    plt.title(title)
    plt.show()