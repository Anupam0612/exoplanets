import unittest
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class TestVisualizations(unittest.TestCase):

    def setUp(self):
        # Load the dataset for testing
        self.df = pd.read_csv("/Users/anupammishra/exoplanets/docs/search.csv")

    def test_atmospheric_height_calculation(self):
        # Test the atmospheric height calculation
        self.df['H'] = 1000 * 8.3144598 * self.df['Teq'] / 2.3 / self.df['log10g_p']
        self.assertTrue('H' in self.df.columns)  # Check if the 'H' column is added

    def test_heatmap_visualization(self):
        # Test the heatmap visualization
        missing_values = self.df.isnull()
        heatmap = sns.heatmap(data=missing_values, yticklabels=False, cbar=False, cmap='viridis')
        self.assertIsInstance(heatmap, plt.Axes)  # Check if the heatmap is created as a Matplotlib Axes object

    def test_static_visualizations(self):
        # Test static visualizations
        # (You can create similar tests for Plot 1, Plot 2, Plot 3, and Plot 4)

        # Plot 1
        plt.figure(figsize=(10, 6))
        plt.scatter(self.df['Mp'], self.df['SNR_Emission_5_micron'], c=self.df['SNR_Transmission_K_mag'], cmap='viridis',
                    s=self.df['SNR_Emission_15_micron'], alpha=0.8)
        plt.xlabel('Planet Mass (Log-scale)')
        plt.ylabel('5 Micron Emission SNR (Signal to Noise Ratio, Log-scale)')
        plt.xscale('log')
        plt.yscale('log')
        plt.title('Planet Mass vs 5 Micron Emission SNR with K-band Transmission SNR Colormap')
        plt.colorbar(label='K-band Transmission SNR (Signal to Noise Ratio)')
        self.assertIsInstance(plt.gcf(), plt.Figure)  # Check if the figure is created as a Matplotlib Figure object

    def test_pair_plots(self):
        # Test pair plots
        columns_to_plot = ['Rp', 'Mp', 'Distance', 'Period', 'log10g_p', 'Teq', 'SNR_Emission_15_micron',
                           'SNR_Emission_5_micron', 'SNR_Transmission_K_mag']
        df_subset = self.df[columns_to_plot]
        pair_plot = sns.pairplot(df_subset, diag_kind='kde', markers='o', palette='viridis')
        self.assertIsInstance(pair_plot, sns.axisgrid.PairGrid)  # Check if the pair plot is created as a Seaborn PairGrid object

if __name__ == '__main__':
    unittest.main()
