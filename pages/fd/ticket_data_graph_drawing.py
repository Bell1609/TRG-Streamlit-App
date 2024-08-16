import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import plotly.express as px
import squarify

from matplotlib.ticker import MaxNLocator
from matplotlib.patches import Patch

class Ticket_Graph_Drawing():
    def recency_graph(self, df_attributes):
        plt.figure()
        
        sns.histplot(df_attributes['Recency'], kde=True)
        
        plt.xlabel('Recency')
        plt.ylabel('Number of Customers')
        plt.title(f"Number of Customers based on recency")
        
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        
        return plt.gcf()
    
    def interactions_graph(self,df_attributes):
        plt.figure()
        sns.histplot(df_attributes[df_attributes['Customer interactions']< 10000]['Customer interactions'], 
                    color='red', 
                    kde=True, 
                    alpha=0.5,
                    label='Customer interactions'
        )
        sns.histplot(df_attributes[df_attributes['Agent interactions'] < 10000]['Agent interactions'], 
                    color='blue', 
                    kde=True, 
                    alpha=0.5,
                    label='Agent interactions'
        )
        
        plt.legend()
        
        return plt.gcf()
    
    def tickets_graph(self, df_attributes):
        plt.figure()
        sns.histplot(df_attributes[df_attributes['Ticket Count']<1000]['Ticket Count'], kde=True, color='orange', edgecolor='lightblue')
        
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        
        return plt.gcf()
    
    def treemap_drawing(self, cluster_centers):
        plt.figure()
        sns.set_style(style="whitegrid") # set seaborn plot style
        sizes= cluster_centers['Cluster Size']# proportions of the categories
        colors = {'Cluster 0': 'lightgreen', 'Cluster 1': 'pink','Cluster 2': 'blue', 'Cluster 3': 'indigo', 'Cluster 4' : 'red'}
        
        total_customers = cluster_centers['Cluster Size'].sum()

        squarify.plot(
            sizes=sizes,
            alpha=0.6, 
            color=colors.values(),
            label=colors.keys()
            # label=['{} \n{:.0f} days \n{:.0f} transactions \n${:.0f} \n{:.0f} customer(s)'.format(*cluster_centers.iloc[i]) for i in range(0, len(cluster_centers))]
        ).axis('off')

        # Creating custom legend
        handles = []
        for i in cluster_centers.index:
            label = '{} \n{:.0f} days \n{:.0f} tickets created\n{:.1f}% AMS, {:.1f}% CMS users \n{:.0f}/{:.0f} customer/agent interactions\n{:.0f} customers ({:.1f}%)'.format(
                cluster_centers.loc[i, 'Cluster'], cluster_centers.loc[i, 'Recency'], cluster_centers.loc[i, 'Ticket Count'], cluster_centers.loc[i, 'AMS'] * 100, 
                cluster_centers.loc[i, 'CMS'] * 100, cluster_centers.loc[i, 'Customer interactions'], cluster_centers.loc[i, 'Agent interactions'],
                cluster_centers.loc[i, 'Cluster Size'],
                cluster_centers.loc[i, 'Cluster Size'] / total_customers * 100
            )
            handles.append(Patch(facecolor=colors.get(f'Cluster {i}', 'grey'), label=label))
            
        plt.legend(handles=handles, loc='center left', bbox_to_anchor=(1, 0.5), fontsize='large')
        plt.title('Customer Ticketing Segmentation Treemap', fontsize=20)
        
        return plt.gcf()
    
    def scatter_3d_drawing(self, df_kmeans):
        df_scatter = df_kmeans
        df_scatter['Combined Interactions'] = df_kmeans['Agent interactions'] + df_kmeans['Customer interactions']
        df_scatter['Cluster'] = df_scatter['Cluster'].astype(str)

        # Define a custom color sequence
        custom_colors = ['#e60049', '#0bb4ff', '#9b19f5', '#00bfa0' , '#e6d800', '#8D493A', '#55AD9B', '#7ED7C1', '#EA8FEA'] 

        # Create the 3D scatter plot
        fig = px.scatter_3d(
            df_scatter, 
            x='Recency', 
            y='Ticket Count', 
            z='Combined Interactions', 
            color='Cluster',
            opacity=0.7,
            width=600,
            height=500,
            color_discrete_sequence=custom_colors
        )

        # Update marker size and text position
        fig.update_traces(marker=dict(size=6), textposition='top center')

        # Update layout template
        fig.update_layout(template='plotly_white')

        # Show the plot
        return fig