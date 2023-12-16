.. These are examples of badges you might want to add to your README:
   please update the URLs accordingly

    .. image:: https://api.cirrus-ci.com/github/<USER>/exoplanets.svg?branch=main
        :alt: Built Status
        :target: https://cirrus-ci.com/github/<USER>/exoplanets
    .. image:: https://readthedocs.org/projects/exoplanets/badge/?version=latest
        :alt: ReadTheDocs
        :target: https://exoplanets.readthedocs.io/en/stable/
    .. image:: https://img.shields.io/coveralls/github/<USER>/exoplanets/main.svg
        :alt: Coveralls
        :target: https://coveralls.io/r/<USER>/exoplanets
    .. image:: https://img.shields.io/pypi/v/exoplanets.svg
        :alt: PyPI-Server
        :target: https://pypi.org/project/exoplanets/
    .. image:: https://img.shields.io/conda/vn/conda-forge/exoplanets.svg
        :alt: Conda-Forge
        :target: https://anaconda.org/conda-forge/exoplanets
    .. image:: https://pepy.tech/badge/exoplanets/month
        :alt: Monthly Downloads
        :target: https://pepy.tech/project/exoplanets
    .. image:: https://img.shields.io/twitter/url/http/shields.io.svg?style=social&label=Twitter
        :alt: Twitter
        :target: https://twitter.com/exoplanets

.. image:: https://img.shields.io/badge/-PyScaffold-005CA0?logo=pyscaffold
    :alt: Project generated with PyScaffold
    :target: https://pyscaffold.org/

|

==========
exoplanets
==========
    This README provides an overview of the machine learning methods and visualizations implemented in the Exoplanet Observability Project.

    The exoplanets PyScaffold contains the final project for MEMT680: Data Analysis and Machine Learning at Drexel University, Fall 2023.
    The goal of this project is to train machine learning models to determine and rank the relative observability of exoplanet atmospheres 
    for future studies based on their measured properties. We aim to predict how observable an exoplanet's atmosphere can be expected to be, 
    leveraging its basic features such as size, mass, surface temperature, and more. 

    PyScaffold Structure:

    Final Report: docs/finalreport.ipynb is a Jupyter Notebook file that contains the final report for this project. It includes:
    - Description of Dataset
    - Initial Data Visualization
    - Data Preprocessing Guide (data cleaning and normalization methods)
    - Visualization of Preprocessed Data
    - Machine Learning Challenge Description
    - Model Training and Hyperparameter Tuning
    - Model Evaluation Visualizations
    - Improvement Strategies

    Please run finalreport.ipynb for full access to the contents and results of this project.

    NOTE: All of the plots that were generated as part of this report can be accessed at docs/plots.

    The compiled code can also be found within src.
    hw4.py includes the initial visualizations for the project, including a heatmap, multiple scatterplots, a pie chart, and an interactive Dash scatterplot.
    hw6.py includes the remaining code, including preprocessing and the Supervised and Unsupervised Machine Learning Methods

    The dataset used for this project can be found at docs/search.csv. It is also uploaded on DataFed under parent collection c/500594589.
    The dataset includes the following features:

    Planet Name (Planet_Name)
    1.5 micron emission SNR (signal-to-noise ratio) relative to HD 209458 b (SNR_Emission_15_micron)
    5 micron emission SNR relative to HD 209458 b (SNR_Emission_5_micron)
    K-band Transmission SNR relative to HD 209458 b (SNR_Transmission_K_mag)
    Planet radius (Rp) [Jupiter radii]
    Planet mass (Mp) [Jupiter masses]
    Dayside temperature (Tday) [K]
    Planet equilibrium temperature (Teq) [K]
    Planet surface gravity (log10g_p) [log(cm/s^2)]
    Planet orbital period (Period) [days]
    Planet transit duration (Transit_Duration) [hours]
    K-band magnitude (K_mag) [mag]
    Distance to planet host star (Distance) [parsecs]
    Stellar effective temperature (Teff) [K]
    Stellar surface gravity (log10g_s) [log(cm/s^2)]
    Planet transit flag (Transit_Flag) - FALSE or TRUE
    Planet source catalog name (Catalog_Name)

    The code used to commit data to DataFed can be found at datapath/data.py

# Machine Learning and Visualization

 It covers Random Forest (RF), Gradient Boosting Regression (GBR), K-means Clustering, Principal Component Analysis (PCA), and t-distributed Stochastic Neighbor Embedding (t-SNE).

## Table of Contents
- [Machine Learning Methods](#machine-learning-methods)
  - [Random Forest (RF)](#random-forest-rf)
  - [Gradient Boosting Regression (GBR)](#gradient-boosting-regression-gbr)
  - [K-means Clustering](#k-means-clustering)
  - [Principal Component Analysis (PCA)](#principal-component-analysis-pca)
  - [t-distributed Stochastic Neighbor Embedding (t-SNE)](#t-distributed-stochastic-neighbor-embedding-t-sne)
- [Visualizations](#visualizations)
  - [Interactive Scatterplot](#interactive-scatterplot)
  - [Machine Learning Challenge Description](#machine-learning-challenge-description)
  - [Supervised Learning](#supervised-learning)

## Machine Learning Methods

### Random Forest (RF)
Random Forest is an ensemble learning method for both classification and regression tasks. It builds multiple decision trees and merges them together to get a more accurate and stable prediction.

#### Implementation Details:
- Used `sklearn.ensemble.RandomForestRegressor` for regression tasks.
- Conducted hyperparameter tuning using GridSearchCV.
- Evaluated performance using Mean Squared Error.

### Gradient Boosting Regression (GBR)
Gradient Boosting Regression is an ensemble learning method that builds a series of decision trees sequentially, with each tree correcting the errors made by the combined ensemble of preceding trees.

#### Implementation Details:
- Utilized `sklearn.ensemble.GradientBoostingRegressor`.
- Performed hyperparameter tuning with GridSearchCV.
- Evaluated performance using Mean Absolute Error.
- Applied cross-validated evaluation.

### K-means Clustering
K-means Clustering is an unsupervised learning algorithm that partitions data into clusters based on similarity.

#### Implementation Details:
- Used `sklearn.cluster.KMeans` for clustering.
- Employed GridSearchCV for hyperparameter tuning.
- Evaluated clusters using silhouette scores.
- Visualized results with scatter plots.

### Principal Component Analysis (PCA)
Principal Component Analysis is a dimensionality reduction technique that transforms data into a new coordinate system to reduce its dimensionality.

#### Implementation Details:
- Used `sklearn.decomposition.PCA`.
- Employed GridSearchCV for hyperparameter tuning.
- Visualized the explained variance ratio.
- Plotted scatter plots of PCA components.

### t-distributed Stochastic Neighbor Embedding (t-SNE)
t-SNE is a dimensionality reduction technique that visualizes high-dimensional data in two or three dimensions.

#### Implementation Details:
- Utilized `sklearn.manifold.TSNE`.
- Employed GridSearchCV for hyperparameter tuning.
- Visualized t-SNE results with scatter plots.

## Visualizations

### Interactive Scatterplot
An interactive scatterplot was created using Dash, allowing users to select variables for x and y axes and colormap. It visualizes exoplanet data with 'Transit Duration' represented by point size.

#### Implementation Details:
- Implemented using Dash for interactivity.
- Fixed x and y axes while allowing selection for colormap and point size.

### Machine Learning Challenge Description
Described the goal of the machine learning challenge, which is to predict the observability of an exoplanet's atmosphere based on measured properties.

#### Challenges Addressed:
- Data Complexity.
- Feature Selection.
- The Curse of Dimensionality.

### Supervised Learning
Applied supervised learning methods to map exoplanet parameters to Signal-to-Noise Ratio (SNR) values, predicting atmosphere observability.

#### Implementation Details:
- Utilized features like radius, mass, gravity, etc.
- Employed `sklearn.model_selection.train_test_split`.
- Used Gradient Boosting for predictions.
- Conducted hyperparameter tuning and cross-validated evaluation.

---




.. _pyscaffold-notes:


====

This project has been set up using PyScaffold 4.5. For details and usage
information on PyScaffold see https://pyscaffold.org/.
