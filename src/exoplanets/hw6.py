# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from dash import html
from dash.dependencies import Input, Output
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, accuracy_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import RobustScaler
from scipy.stats import zscore

# Load the dataset
df = pd.read_csv("docs/search.csv")  # Replace "your_dataset.csv" with the actual file name

# define the atmospheric height H based on existing formulas

df['H'] = 1000*8.3144598*df['Teq']/2.3/df['log10g_p']

# Preprocessing - remove rows with missing values
df.dropna(inplace=True)

# Import the MinMaxScaler from scikit-learn
from sklearn.preprocessing import MinMaxScaler

# Assuming X is your dataset
numeric_columns = df.select_dtypes(include=np.number).columns

# Create an instance of the MinMaxScaler
scaler = MinMaxScaler()

# Apply MinMax scaling to the numeric columns in the DataFrame
df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

# Define a threshold for outlier removal (adjust as needed)
threshold = 0.95

# Create a boolean mask identifying rows with values above the threshold for all numeric columns
outlier_mask = (df[numeric_columns] >= threshold).all(axis=1)

# Remove rows identified by the outlier mask (keeping rows where at least one value is below the threshold)
df = df[~outlier_mask]


# Static Visualizations

# Plot 1: Scatter plot of Planet Mass vs. 5 Micron Emission SNR
plt.figure(figsize=(10, 6))

# Scatter plot with colormap and size based on Transit Duration
plt.scatter(df['Mp'], df['SNR_Emission_5_micron'], c=df['SNR_Transmission_K_mag'], cmap='viridis', s=df['SNR_Emission_15_micron']*1000, alpha=0.8)  
plt.xlabel('Planet Mass (Log-scale)')  # X label
plt.ylabel('5 Micron Emission SNR (Signal to Noise Ratio, Log-scale)')  # Y label
plt.xscale('log')  # Set logarithmic scale for X-axis
plt.yscale('log')  # Set logarithmic scale for Y-axis
plt.title('Planet Mass vs 5 Micron Emission SNR with K-band Transmission SNR Colormap')  # Plot 1 Title
plt.colorbar(label='K-band Transmission SNR (Signal to Noise Ratio)')  # Set colorbar
plt.show()


# Plot 2: Scatter plot of Planet Radius vs. Planet Mass
plt.figure(figsize=(10, 6))

# Scatter plot with colormap and size based on Transit Duration
plt.scatter(df['Rp'], df['Mp'], c=df['SNR_Emission_15_micron'], cmap='viridis', s=df['Transit_Duration']*1000, alpha=0.8)  
plt.xlabel('Planet Radius (Log-scale)')  # X label
plt.ylabel('Planet Mass, Log-scale)')  # Y label
plt.xscale('log')  # Set logarithmic scale for X-axis
plt.yscale('log')  # Set logarithmic scale for Y-axis
plt.title('Planet Mass vs Planet Radius with 1.5 Micron Emission SNR Colormap')  # Plot 1 Title
plt.colorbar(label='1.5 Micron Emission SNR (Signal to Noise Ratio)')  # Set colorbar
plt.show()


# Plot 3: Pie Chart of Greatest SNR Values between the given 3

# Identify the column with the largest value for each row
df['Largest_SNR'] = df[['SNR_Emission_15_micron', 'SNR_Emission_5_micron', 'SNR_Transmission_K_mag']].idxmax(axis=1)  # Find the column with the largest SNR value for each row

# Count the occurrences of each category
counts = df['Largest_SNR'].value_counts()  # Count the occurrences of each category

# Create a static pie chart
plt.figure(figsize=(8, 8))
plt.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=140, colors=['skyblue', 'lightcoral', 'lightgreen'])  # Pie chart with percentages and colors
plt.title('Distribution Based on Largest SNR')  # Pie chart title
plt.show()

# Plot 4: Pair Plots
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming 'df' is your DataFrame with relevant columns
# Include the features and target variables in the pair plot
columns_to_plot = ['Rp', 'Mp', 'Distance', 'Period', 'log10g_p', 'Teq','SNR_Emission_15_micron', 'SNR_Emission_5_micron', 'SNR_Transmission_K_mag']

# Create a subset of the DataFrame with the selected columns
df_subset = df[columns_to_plot]

# Plotting pair plot
sns.pairplot(df_subset, diag_kind='kde', markers='o')
plt.show()


# SUPERVISED LEARNING


# Random Forest Regression



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# Features (X) and Target (y) for SNR prediction
X_regression = df[['Rp', 'Mp', 'Distance', 'log10g_p']]  # Add other features as needed
y_regression = df[['SNR_Emission_15_micron', 'SNR_Emission_5_micron']]

# Split the data into training and testing sets
X_train_regression, X_test_regression, y_train_regression, y_test_regression = train_test_split(
    X_regression, y_regression, test_size=0.2, random_state=42
)

# Standardize the features
scaler = StandardScaler()
X_train_regression_scaled = scaler.fit_transform(X_train_regression)
X_test_regression_scaled = scaler.transform(X_test_regression)

# Define the parameter grid for hyperparameter tuning
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Initialize the Random Forest Regressor
regressor = RandomForestRegressor(random_state=42)

# Initialize GridSearchCV
grid_search = GridSearchCV(regressor, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)

# Fit the model with the best parameters
grid_search.fit(X_train_regression_scaled, y_train_regression)

# Get the best parameters
best_params = grid_search.best_params_
print("Best Parameters:", best_params)

# Predict SNR values on the test set using the best model
best_regressor = grid_search.best_estimator_
y_pred_regression = best_regressor.predict(X_test_regression_scaled)

# Evaluate the regression model
mse_regression = mean_squared_error(y_test_regression, y_pred_regression)
print(f'Mean Squared Error (Regression): {mse_regression}')

# Get feature importance scores
feature_importances = best_regressor.feature_importances_

# Create a DataFrame with feature names and their importance scores
feature_importance_df = pd.DataFrame({'Feature': X_regression.columns, 'Importance': feature_importances})

# Sort the DataFrame by importance in descending order
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Plot the feature importance scores
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='viridis')
plt.title('Feature Importance Scores')
plt.show()

# Predict SNR values for the entire dataset
df['Predicted_SNR_Emission_15_micron'], df['Predicted_SNR_Emission_5_micron'] = best_regressor.predict(X_regression).T

# Plot training and predicted SNR 15 values against Rp
plt.figure(figsize=(12, 6))

# Training Data - SNR 15 vs. Rp
plt.subplot(1, 2, 1)
sns.scatterplot(data=df, x='Rp', y='SNR_Emission_15_micron', palette='viridis')
plt.yscale('log')
plt.title('Training Data - SNR 15 vs. Rp')

# Predicted Data - SNR 15 vs. Rp
plt.subplot(1, 2, 2)
sns.scatterplot(x=X_test_regression['Rp'], y=df['Predicted_SNR_Emission_15_micron'], palette='viridis')
plt.yscale('log')
plt.title('Predicted Data - SNR 15 vs. Rp')

plt.tight_layout()
plt.show()





# Gradient-Boosting Regression

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

# Function for model evaluation

def evaluate_model(model, X_train, X_test, y_train, y_test):
    """_summary_

    Args:
        model (_type_): _description_
        X_train (_type_): _description_
        X_test (_type_): _description_
        y_train (_type_): _description_
        y_test (_type_): _description_

    Returns:
        _type_: _description_
    """
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    mae_train = mean_absolute_error(y_train, y_train_pred)
    mae_test = mean_absolute_error(y_test, y_test_pred)
    return mae_train, mae_test, y_test_pred

# Function for hyperparameter tuning
def tune_hyperparameters(model, param_grid, X_train, y_train):
    """_summary_

    Args:
        model (_type_): _description_
        param_grid (_type_): _description_
        X_train (_type_): _description_
        y_train (_type_): _description_

    Returns:
        _type_: _description_
    """
    grid_search = GridSearchCV(model, param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    return best_params

# Function for cross-validated evaluation
def cross_validate_model(model, X, y):
    """_summary_

    Args:
        model (_type_): _description_
        X (_type_): _description_
        y (_type_): _description_

    Returns:
        _type_: _description_
    """
    mae_scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=5)
    return -mae_scores  # Negate the scores to obtain positive MAE values

# Function to plot the results
def plot_results(actual, predicted, feature_name):
    """_summary_

    Args:
        actual (_type_): _description_
        predicted (_type_): _description_
        feature_name (_type_): _description_
    """
    plt.figure(figsize=(12, 6))
    
    # Actual SNR 15 vs. Feature
    plt.subplot(1, 2, 1)
    sns.scatterplot(data=df, x=feature_name, y=actual, palette='viridis')
    plt.yscale('log')
    plt.title(f'Actual SNR 15 vs. {feature_name}')

    # Predicted SNR 15 vs. Feature
    plt.subplot(1, 2, 2)
    sns.scatterplot(x=X_test[feature_name], y=predicted, palette='viridis')
    plt.yscale('log')
    plt.title(f'Predicted SNR 15 vs. {feature_name}')

    plt.tight_layout()
    plt.show()

# 1a. Supervised Learning (Gradient Boosting Regression)
# Define features (X) and target variable (y)
features = ['Rp', 'Mp', 'Tday', 'Teq', 'log10g_p', 'Period', 'Transit_Duration', 'K_mag', 'Distance', 'Teff', 'log10g_s']
target = 'SNR_Emission_15_micron'

X = df[features]
y = df[target]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Gradient Boosting Regressor
gb_regressor = GradientBoostingRegressor(n_estimators=200, random_state=0)

# Hyperparameter tuning for regularization
param_grid = {
    'alpha': [0.1, 0.5, 0.9],
    'max_depth': [3, 4, 5],
    'learning_rate': [0.01, 0.1, 0.2]
}

# Tune hyperparameters
best_params = tune_hyperparameters(GradientBoostingRegressor(n_estimators=200, random_state=0), param_grid, X_train, y_train)
print("Best Parameters:", best_params)

# Fit the model with the best parameters
best_gb_regressor = GradientBoostingRegressor(n_estimators=200, random_state=0, **best_params)
best_gb_regressor.fit(X_train, y_train)
y_train_pred_best = best_gb_regressor.predict(X_train)
y_test_pred_best = best_gb_regressor.predict(X_test)

# Cross-validated evaluation
mae_scores_cv = cross_validate_model(best_gb_regressor, X, y)

# Print mean MAE scores across folds
print("Cross-validated Mean MAE:", mae_scores_cv.mean())

# Print MAE for training and testing sets
mae_train, mae_test, _ = evaluate_model(best_gb_regressor, X_train, X_test, y_train, y_test)
print(f"MAE on Training Set: {mae_train:.2f}")
print(f"MAE on Testing Set: {mae_test:.2f}")

# Scatter plot of predicted SNR values vs. actual SNR values
plot_results(y_test, y_test_pred_best, 'Rp')
plot_results(y_test, y_test_pred_best, 'Teq')
plot_results(y_test, y_test_pred_best, 'log10g_p')
plot_results(y_test, y_test_pred_best, 'Distance')




# UNSUPERVISED LEARNING


# K-Means Clustering



# 2. Unsupervised Learning (K-means Clustering)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import silhouette_score


# Features for clustering
X_cluster = df[['SNR_Emission_15_micron', 'SNR_Emission_5_micron', 'Rp']]

# Apply log transformation to the features
X_cluster_log = np.log1p(X_cluster)

# Impute missing values
imputer = SimpleImputer(strategy='mean')  
X_cluster_log_imputed = imputer.fit_transform(X_cluster_log)

# Standardize the features
scaler = StandardScaler()
X_cluster_scaled = scaler.fit_transform(X_cluster_log_imputed)

# KMeans pipeline
kmeans_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
    ('kmeans', KMeans())
])

# Hyperparameter grid for KMeans
param_grid = {
    'kmeans__n_clusters': [1,2,3,4],
    'kmeans__init': ['k-means++', 'random'],
    'kmeans__max_iter': [100, 200, 300],
    'kmeans__n_init': [10]
}
# GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(kmeans_pipeline, param_grid, cv=5)
grid_search.fit(X_cluster_scaled)

# Get the best parameters
best_params = grid_search.best_params_
print("Best Parameters:", best_params)

# Fit the pipeline with the best parameters
best_kmeans_pipeline = grid_search.best_estimator_

# Predict clusters
df['Cluster'] = best_kmeans_pipeline.named_steps['kmeans'].predict(X_cluster_scaled)

# Explore the resulting clusters
for cluster_label in range(best_params['kmeans__n_clusters']):
    print(f'\nCluster {cluster_label}:')
    print(df[df['Cluster'] == cluster_label])

# Evaluate the silhouette score
silhouette_score_kmeans = silhouette_score(X_cluster_scaled, df['Cluster'])
print(f"Silhouette Score on KMeans Clusters: {silhouette_score_kmeans}")

# Scatter plot of clustered data
plt.figure(figsize=(10, 6))

for cluster_label in range(best_params['kmeans__n_clusters']):
    cluster_data = X_cluster_scaled[df['Cluster'] == cluster_label]
    plt.scatter(cluster_data[:, 0], cluster_data[:, 2], label=f'Cluster {cluster_label}')

# Plot centroids
centroids = best_kmeans_pipeline.named_steps['kmeans'].cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 2], marker='X', s=200, c='red', label='Centroids')

plt.title('K-means Clustering Visualization')
plt.xlabel('SNR_Emission_15_micron (scaled)')
plt.ylabel('Planet Radius')
plt.legend()
plt.show()

# Scatter plot of SNR_Emission_15_micron vs. SNR_Emission_5_micron with color-coded KMeans clusters
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
sns.scatterplot(data=df, x='Teq', y='SNR_Emission_15_micron', hue='Cluster', palette='viridis')
plt.yscale('log')
plt.title('SNR_Emission_15_micron vs. Teq with KMeans Clusters (log scale)')

plt.subplot(1, 2, 2)
sns.scatterplot(data=df, x='Rp', y='SNR_Emission_15_micron', hue='Cluster', palette='viridis')
plt.yscale('log')
plt.xscale('log')
plt.title('SNR_Emission_15_micron vs. Rp with KMeans Clusters (log scale)')

plt.tight_layout()
plt.show()







# Principal Component Analysis (PCA)


from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming X_cluster_log_imputed is defined

# PCA pipeline
pca_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
    ('pca', PCA())
])

# Hyperparameter grid for PCA
param_grid_pca = {
    'pca__n_components': [2, 3, 4],
    'pca__svd_solver': ['auto', 'full', 'arpack', 'randomized'],  # Add solver options
    'pca__iterated_power': [5, 10, 15]  # Adjust iterated_power
}


# GridSearchCV for hyperparameter tuning
grid_search_pca = GridSearchCV(pca_pipeline, param_grid_pca, cv=5)
grid_search_pca.fit(X_cluster_log_imputed)

# Get the best parameters
best_params_pca = grid_search_pca.best_params_
print("Best Parameters for PCA:", best_params_pca)

# Fit and transform the data using PCA with the best parameters
X_pca = grid_search_pca.best_estimator_.transform(X_cluster_log_imputed)

# Access the principal components
principal_components = grid_search_pca.best_estimator_['pca'].components_

# Create a DataFrame to display the results
components_df = pd.DataFrame(principal_components.T, columns=[f'PC{i+1}' for i in range(principal_components.shape[0])])
print("Principal Components:")
print(components_df)

# Visualize the explained variance ratio
explained_variance = grid_search_pca.best_estimator_['pca'].explained_variance_ratio_
print("Explained Variance Ratio:", explained_variance)

pca_df = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(X_pca.shape[1])])
# Add the 'Cluster' column to the PCA DataFrame
pca_df['Cluster'] = df['Cluster']

# Scatter plot of PCA components
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=df['Cluster'], palette='viridis')
plt.title('PCA Components with KMeans Clusters')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()



# K-Means Clustering + PCA

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import silhouette_score

# Features for clustering
X_cluster = df[['SNR_Emission_15_micron', 'SNR_Emission_5_micron', 'Rp']]

# Apply log transformation to the features
X_cluster_log = np.log1p(X_cluster)

# Impute missing values
imputer = SimpleImputer(strategy='mean')  
X_cluster_log_imputed = imputer.fit_transform(X_cluster_log)

# Standardize the features
scaler = StandardScaler()
X_cluster_scaled = scaler.fit_transform(X_cluster_log_imputed)

# PCA pipeline
pca_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=2))  # Adjust the number of components as needed
])

# Fit and transform the data using PCA
X_pca = pca_pipeline.fit_transform(X_cluster_scaled)

# KMeans pipeline
kmeans_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=3)),  # Align the number of components with PCA
    ('kmeans', KMeans())
])

# Hyperparameter grid for KMeans
param_grid = {
    'kmeans__n_clusters': [1, 2, 3, 4],
    'kmeans__init': ['k-means++', 'random'],
    'kmeans__max_iter': [100, 200, 300]
}

# GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(kmeans_pipeline, param_grid, cv=5)
grid_search.fit(X_cluster_log_imputed)

# Get the best parameters
best_params = grid_search.best_params_
print("Best Parameters:", best_params)

# Fit the pipeline with the best parameters
best_kmeans_pipeline = grid_search.best_estimator_

# Predict clusters
df['Cluster'] = best_kmeans_pipeline.named_steps['kmeans'].predict(X_cluster_log_imputed)

# Evaluate the silhouette score
# silhouette_score_kmeans = silhouette_score(X_cluster_scaled, df['Cluster'])
# print(f"Silhouette Score on KMeans Clusters: {silhouette_score_kmeans}")

# Scatter plot of PCA components with color-coded KMeans clusters
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=df['Cluster'], palette='viridis')
plt.title('PCA Components with KMeans Clusters')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

plt.subplot(1, 2, 2)
sns.scatterplot(data=df, x='Rp', y='SNR_Emission_15_micron', hue='Cluster', palette='viridis')
plt.yscale('log')
plt.xscale('log')
plt.title('SNR_Emission_15_micron vs. SNR_Emission_5_micron with KMeans Clusters (log scale)')

plt.tight_layout()
plt.show()




# Manifold Learning - t-SNE 


# Manifold Learning

# Import t-SNE
from sklearn.manifold import TSNE
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming X_cluster_log_imputed is defined

# t-SNE pipeline
tsne_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
    ('tsne', TSNE())
])

# Hyperparameter grid for t-SNE
param_grid_tsne = {
    'tsne__n_components': [2],  # Adjust the number of components as needed
    'tsne__perplexity': [30, 50, 100],
    'tsne__learning_rate': [10, 50, 100]
}

# Specify scoring metric
scoring_metric = make_scorer(mean_squared_error, greater_is_better=False)

# GridSearchCV for hyperparameter tuning
grid_search_tsne = GridSearchCV(tsne_pipeline, param_grid_tsne, cv=5, scoring=scoring_metric)
grid_search_tsne.fit(X_cluster_log_imputed)

# Get the best parameters
best_params_tsne = grid_search_tsne.best_params_
print("Best Parameters for t-SNE:", best_params_tsne)

# Fit and transform the data using t-SNE with the best parameters
X_tsne = grid_search_tsne.best_estimator_['tsne'].fit_transform(X_cluster_log_imputed)

# Plot the t-SNE results
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=df['Cluster'], palette='viridis')
plt.title('t-SNE Visualization with Best Parameters')
plt.show()
