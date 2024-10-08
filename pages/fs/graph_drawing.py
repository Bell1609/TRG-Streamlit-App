import random
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import plotly.express as px
import squarify
import streamlit as st


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
    

    def pipeline_trend(self, df, start_date, end_date):
        """Generate the trend of total deal value and deal count in the pipeline grouped by month."""
        
        # Ensure 'Deal : Created at' and 'Deal : Closed date' columns are in datetime format
        df['Deal : Created at'] = pd.to_datetime(df['Deal : Created at'], errors='coerce')
        df['Deal : Closed date'] = pd.to_datetime(df['Deal : Closed date'], errors='coerce')

        # Generate a range of month-ends from start to end
        date_range = pd.date_range(start=start_date, end=end_date, freq='M')
        
        # Convert DatetimeIndex to a list to allow appending
        date_range_list = date_range.tolist()
        
        # Convert end_date to a pandas Timestamp if it is not already
        end_date_ts = pd.Timestamp(end_date)
        
        # If the exact end_date is not already in the date range, add it
        if end_date_ts not in date_range_list:
            date_range_list.append(end_date_ts)
        
        # Sort the list of dates to maintain chronological order
        date_range_list = sorted(date_range_list)
        
        # Convert the list back to a DatetimeIndex
        date_range = pd.DatetimeIndex(date_range_list)
        def pipeline_value_and_count_at_month(df, month_end):
            """Calculate total deal value and count of deals in the pipeline as of the end of a given month."""
            # Extend end_date to include the full day
            end_of_month = pd.to_datetime(month_end) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
            
            
            # Calculate the start of the month based on month_end
            month_start = month_end.replace(day=1)

            # Filter deals that were in the pipeline during the given month
            pipeline_deals = df[
                (df['Deal : Created at'] <= month_end) &  # Deal was created on or before the month end
                ((df['Deal : Closed date'].isna()) | (df['Deal : Closed date'] > month_end))  # Deal is still open or closed after the month end
            ]
            # st.write(f'Start: {month_start} - End: {end_of_month}')
            # st.write(f'Rows: {pipeline_deals["Deal : id"].count()}')
            # st.dataframe(pipeline_deals[['Deal : Name','Deal : Total Deal Value','Deal : Owner','Deal : Project type']])
            # Sum the total deal value for the filtered deals
            total_value = pipeline_deals['Deal : Total Deal Value'].sum()

            # Count deals created in the current month (between month_start and month_end)
            deals_created = df[
                (df['Deal : Created at'] >= month_start) &  
                (df['Deal : Created at'] <= end_of_month)
            ]
            deal_created_count = deals_created['Deal : id'].nunique()
            
            deals_closed = df[
                (df['Deal : Closed date'] >= month_start) &  
                (df['Deal : Closed date'] <= end_of_month) &
                (df['Deal : Deal stage'] == 'Won')
            ]
            deal_closed_count = deals_closed['Deal : id'].nunique()

            return total_value, deal_created_count, deal_closed_count

        # Initialize lists to store results
        months = []
        total_values = []
        deal_created_counts = []
        deal_closed_counts = []

        # Calculate total deal value and deal count for each month in the date range
        for month_end in date_range:
            total_value, deal_created_count, deal_closed_count = pipeline_value_and_count_at_month(df, month_end)
            months.append(month_end)
            total_values.append(total_value)  # Store total value
            deal_created_counts.append(deal_created_count)  # Store deal count
            deal_closed_counts.append(deal_closed_count)  # Store deal count

        # Create a DataFrame to return
        trend_df = pd.DataFrame({
            'Month': months,
            'Total Deal Value': total_values,
            'Deals Created Count': deal_created_counts,
            'Deals Closed Count': deal_closed_counts
        })

        return trend_df


    def plot_pipeline_trend(self, trend_df, start_month, end_month):
        """Plots 'Deals Closed Count' and 'Deals Created Count' on a bar chart, and 'Total Deal Value' on a line chart."""
        
        # Ensure the 'Month' column is in datetime format
        trend_df['Month'] = pd.to_datetime(trend_df['Month'])

        # Filter the DataFrame based on the selected month range
        filtered_trend_df = trend_df[
            (trend_df['Month'] >= pd.to_datetime(start_month)) & 
            (trend_df['Month'] <= pd.to_datetime(end_month))
        ]

        # Plot 1: Total Deal Value (line chart)
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        
        ax1.plot(filtered_trend_df['Month'], filtered_trend_df['Total Deal Value'], marker='o', linestyle='-', color='b', label='Total Deal Value')
        ax1.set_title('Total Deal Value in Pipeline by Month')
        ax1.set_xlabel('Month')
        ax1.set_ylabel('Total Deal Value')
        ax1.set_xticks(filtered_trend_df['Month'])
        ax1.set_xticklabels(filtered_trend_df['Month'].dt.strftime('%Y-%m'), rotation=45)
        ax1.grid(True)
        ax1.legend()

        # Display the first figure
        st.pyplot(fig1)

        # Plot 2: Deals Closed and Created Count (bar chart)
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        
        bar_width = 10  # Width of the bars, reduced for better separation
        
        # Plot Deals Closed Count
        ax2.bar(filtered_trend_df['Month'] - pd.DateOffset(days=4),  # Offset bars to the left slightly
                filtered_trend_df['Deals Closed Count'], 
                width=bar_width, color='r', alpha=0.7, label='Deals Closed Count')

        # Plot Deals Created Count
        ax2.bar(filtered_trend_df['Month'] + pd.DateOffset(days=5),  # Offset bars to the right slightly
                filtered_trend_df['Deals Created Count'], 
                width=bar_width, color='g', alpha=0.7, label='Deals Created Count')

        ax2.set_title('Deals Closed and Created Count by Month')
        ax2.set_xlabel('Month')
        ax2.set_ylabel('Count')
        ax2.set_xticks(filtered_trend_df['Month'])
        ax2.set_xticklabels(filtered_trend_df['Month'].dt.strftime('%Y-%m'), rotation=45)
        ax2.grid(True)
        
        # Add a legend for the bar chart
        ax2.legend()

        # Display the second figure
        st.pyplot(fig2)





    





   










    