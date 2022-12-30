# Spatial Transcriptomic Imputation Project
## Overview
Analysis of gene expression profiles in biological samples is a powerful tool used to study various biological systems and phenomena. Traditional assays measure bulk gene expression levels of relatively large samples (i.e. containing millions of cells) yielding robust measurements but hiding cell to cell variability. In the last decade, high throughput single cell RNA-Seq (scRNA-Seq) technologies were developed to capture this variability for thousands of cells simultaneously allowing for in-depth analysis of biological tissues. However, even with single cell resolution, the spatial information over the measured tissue is lost with scRNA-Seq.
Recently, new technologies measure gene expression profiles of biological tissues while maintaining spatial information. Spatial transcriptomics (ST) is a
new technology for measuring RNA expression in tissues while preserving spatial information. ST involves placing a thin slice of tissue on an array covered by a grid of barcoded spots and sequencing the mRNAs of cells within the spots.

![image](https://user-images.githubusercontent.com/31534531/210083755-e3cca461-e6fa-4c25-a677-7018b33700cc.png)

## Goal
The goal of this project is to deal with the depth limitation of the ST technology and perform data imputation on ST dataset to reduce the data sparsity. To do so, I will use DL models and techniques for using the spatial information to enrich the ST information.
To understand how hard is the imputation task, you can see the sparsity of the spots in the following graph.

<img width="976" alt="image" src="https://user-images.githubusercontent.com/31534531/210084107-b222f3ac-43f4-4133-bf32-d7ee96b49e65.png">

## Solution
There are different known ways to apply data imputation:
- Imputation using mean / median / mode values
- Imputation using KNN algorithm (based on features similarity)
- Extrapolation and interpolation methods
- More …
In this project I’ve turn into a DL technique to impute the missing values. The idea is to embed the gene expression dataset into a latent space and then reconstruct it to the original shape with the predicted values. 
The original proposed solution was to use the matrix factorization technique, while using the spatial information as a regularization term to better reconstruct the expression data, and require the reconstruction algorithm to generate “spatially smooth” predictions.
In this project the baseline model was a plain vanilla Neural Matrix Factorization (NMF) model without additional data processing or loss adjustments. Then, I’ve performed additional 4 trials on NMF and also on Auto-Encoder model (AE).

![image](https://user-images.githubusercontent.com/31534531/210084499-2a5c2901-b105-4006-9835-176f5aa451a6.png)

![image](https://user-images.githubusercontent.com/31534531/210084521-b0710410-852a-4953-a5ce-29b5f9f4e8f9.png)

The main goal of this project was to find a way to incorporate the spatial information in a way to optimize the model for a better reconstruction. In contrast of the original proposed solution I’ve used the spatial information inside the loss function calculation as a spatial contribution to each spot. The contribution will be the weighted spatial expression’s difference of the spot from its neighbors (sum over all genes). The weights will be by the spatial neighbors order (1 or 2).

<img width="658" alt="image" src="https://user-images.githubusercontent.com/31534531/210084743-dec38c77-2e06-45cb-833a-9c16d0b22421.png">
