# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import dash
import dash_core_components as dcc
from dash import html
from dash.dependencies import Input, Output
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report


# Load the dataset
df = pd.read_csv("docs/search.csv")  # Replace "your_dataset.csv" with the actual file name


# define the atmospheric height H based on existing formulas

df['H'] = 1000*8.3144598*df['Teq']/2.3/df['log10g_p']


# Identify missing values
missing_values = df.isnull()

# Develop heatmap to visualize missing values

# Plot 0: Heatmap Showing Missing Values in Categories (Unprocessed/Raw Data)
heatmap = sns.heatmap(data = missing_values, yticklabels=False, cbar=False, cmap='viridis')
heatmap.set_title('Missing Data Before Preprocessing')
plt.show()



# Static Visualizations

# Plot 1: Scatter plot of Planet Mass vs. 5 Micron Emission SNR
plt.figure(figsize=(10, 6))

# Scatter plot with colormap and size based on Transit Duration
plt.scatter(df['Mp'], df['SNR_Emission_5_micron'], c=df['SNR_Transmission_K_mag'], cmap='viridis', s=df['SNR_Emission_15_micron'], alpha=0.8)  
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
plt.scatter(df['Rp'], df['Mp'], c=df['SNR_Emission_15_micron'], cmap='viridis', s=df['Transit_Duration']*10, alpha=0.8)  
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



# Plot 5: Interactive Visualization using Dash

# Preprocessing - remove rows with missing values
df.dropna(inplace=True)


app = dash.Dash(__name__)

# Define layout of the interactive app
app.layout = html.Div([
    dcc.Graph(id='interactive-plot'),
    html.Label('Select a feature for color scale:'),
    dcc.Dropdown(
        id='color-scale-dropdown',
        options=[{'label': col, 'value': col} for col in df.columns],
        value='Tday'
    )
])

# Define callback to update the interactive plot based on user input
@app.callback(
    Output('interactive-plot', 'figure'),
    [Input('color-scale-dropdown', 'value')]
)
def update_plot(selected_feature):
    """_summary_

    Args:
        selected_feature (_type_): _description_

    Returns:
        _type_: _description_
    """
    fig = px.scatter(df, x='Rp', y='Mp', color=selected_feature,
                     size='Transit_Duration', title='Interactive Scatter Plot of Planet Radius [Jupiter Radii] vs. Planet Mass [Jupiter Masses]',
                     labels={'Rp': 'Planet Radius (Log-scale)', 'Mp': 'Planet Mass (Log-scale)'},
                     color_continuous_scale= 'viridis', template='plotly_dark')
    
    # Set y-axis to logarithmic scale
    fig.update_layout(xaxis_type='log')
    fig.update_layout(yaxis_type='log')


    # Add a description
    description_text = (
        "This interactive scatter plot visualizes the relationship between 'Planet Radius' and 'Planet Mass'. "
        "The color scale is based on the selected feature, and the size of the points represents 'Transit Duration'."
    )

    fig.add_annotation(
        text=description_text,
        xref="paper", yref="paper",
        x=0.5, y=-0.3,
        showarrow=False,
        font=dict(size=10)
    )

    return fig
    

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)


