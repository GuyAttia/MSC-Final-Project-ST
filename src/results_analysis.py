import matplotlib.pyplot as plt 
import seaborn as sns


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