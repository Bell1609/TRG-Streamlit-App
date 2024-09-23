import random
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import plotly.express as px
import squarify


from matplotlib.ticker import MaxNLocator
from matplotlib.patches import Patch

class Graph_Drawing():
    def rfm_component_graph(self, df_rfm, rfm_component, color):
        plt.figure()
        sns.histplot(df_rfm[rfm_component], bins=30, kde=True, color=color, edgecolor='pink')

        plt.xlabel(rfm_component)
        plt.ylabel('Number of Customers')
        plt.title(f"Number of Customers based on {rfm_component}")
        
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        
        return plt.gcf()
        
    
    #Duong update the function treemap_drawing
    def treemap_drawing(self, cluster_centers):
        plt.figure()
        total_customers = cluster_centers['Cluster Size'].sum()

        sns.set_style(style="whitegrid")  # Set Seaborn plot style

        sizes = cluster_centers['Cluster Size']  # Proportions of the categories

        # Generate random colors for each unique cluster
        unique_clusters = cluster_centers['Cluster'].unique()
        random.seed(50)  # Optional: Set seed for reproducibility
        colors = {cluster: f'#{random.randint(0, 0xFFFFFF):06x}' for cluster in unique_clusters}

        # Draw the treemap
        squarify.plot(
            sizes=sizes,
            alpha=0.6,
            color=[colors[cluster] for cluster in cluster_centers['Cluster']],
            label=cluster_centers['Cluster']
        ).axis('off')

        # Creating custom legend
        handles = []
        for i in cluster_centers.index:
            label = '{} \n{:.0f} days \n{:.0f} transactions \n${:,.0f} \n{:.0f} Customers ({:.1f}%)'.format(
                cluster_centers.loc[i, 'Cluster'], cluster_centers.loc[i, 'Recency'], cluster_centers.loc[i, 'Frequency'],
                cluster_centers.loc[i, 'Monetary'], cluster_centers.loc[i, 'Cluster Size'],
                cluster_centers.loc[i, 'Cluster Size'] / total_customers * 100
            )
            handles.append(Patch(facecolor=colors[cluster_centers.loc[i, 'Cluster']], label=label))

        
        
        plt.legend(handles=handles, loc='center left', bbox_to_anchor=(1, 0.5), fontsize='large')
        plt.title('RFM Segmentation Treemap', fontsize=20)

        return total_customers, plt.gcf()
        
        
        
    def scatter_3d_drawing(self, df_kmeans):
        df_scatter = df_kmeans.copy()
        
        # Select relevant columns
        df_review = df_scatter[['Recency', 'Frequency', 'Monetary', 'Ranking']]
        
        # Ensure the columns are of type float
        df_scatter[['Recency', 'Frequency', 'Monetary']] = df_review[['Recency', 'Frequency', 'Monetary']].astype(float)
        
        # Define a custom color sequence
        custom_colors = ['#e60049', '#0bb4ff', '#9b19f5', '#00bfa0', '#e6d800', '#8D493A', '#55AD9B', '#7ED7C1', '#EA8FEA']
        
        # Create the 3D scatter plot
        fig = px.scatter_3d(
            df_scatter, 
            x='Recency', 
            y='Frequency', 
            z='Monetary', 
            color='Ranking', 
            opacity=0.7,
            width=600,
            height=500,
            color_discrete_sequence=custom_colors
        )
        
        # Update marker size and text position
        fig.update_traces(marker=dict(size=6), textposition='top center')
        
        # Update layout template
        fig.update_layout(template='plotly_white')
        
        return fig

    