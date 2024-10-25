import random
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import plotly.express as px
import squarify
import streamlit as st
import plotly.graph_objects as go
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import Patch
import yaml
import altair as alt



class Graph_Drawing():
    def format_number(self, num):
        if num > 1000000:
            if not num % 1000000:
                return f'{num // 1000000:,} M'
            return f'{round(num / 1000000, 1):,} M'
        return f'{num / 1000:,.0f} K'
        
    


    @st.cache_data(show_spinner=False)
    def rfm_component_graph(_self, df_rfm, rfm_component, color):
        plt.figure()
        sns.histplot(df_rfm[rfm_component], bins=30, kde=True, color=color, edgecolor='pink')

        plt.xlabel(rfm_component)
        plt.ylabel('Number of Customers')
        plt.title(f"Number of Customers based on {rfm_component}")
        
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        
        return plt.gcf()
        
    
    #Duong update the function treemap_drawing
    @st.cache_data(show_spinner=False)
    def treemap_drawing(_self, cluster_centers):
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
        
        
    @st.cache_data(show_spinner=False)
    def scatter_3d_drawing(_self, df_kmeans):
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
    
    @st.cache_data(show_spinner=False)
    def pipeline_trend(_self, df, start_date, end_date):
        """Generate the trend of total deal value and deal count in the pipeline grouped by month."""
        
        # Ensure 'Deal : Created at' and 'Deal : Closed date' columns are in datetime format
        df['Deal : Created at'] = pd.to_datetime(df['Deal : Created at'], errors='coerce')
        df['Deal : Closed date'] = pd.to_datetime(df['Deal : Closed date'], errors='coerce')


        # Generate a range of month-ends from start to end
        date_range = pd.date_range(start=start_date, end=end_date, freq='M')

        # Convert DatetimeIndex to a list to allow appending
        date_range_list = date_range.tolist()

        # Adjust the time to 23:59:59 for each date in the list
        date_range_list = [date.replace(hour=23, minute=59, second=59) for date in date_range_list]

        # Convert end_date to a pandas Timestamp if it is not already
        end_date_ts = pd.Timestamp(end_date).replace(hour=23, minute=59, second=59)

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
            #end_of_month = pd.to_datetime(month_end) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
            
            
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
                (df['Deal : Created at'] <= month_end)
            ]
            deal_created_count = deals_created['Deal : id'].nunique()
            
            deals_closed = df[
                (df['Deal : Closed date'] >= month_start) &  
                (df['Deal : Closed date'] <= month_end) &
                (df['Deal : Deal stage'] == 'Won')
            ]
            deal_closed_count = deals_closed['Deal : id'].nunique()

            return total_value, deal_created_count, deal_closed_count

        # Initialize lists to store results
        months = []
        as_at_date = []
        total_values = []
        deal_created_counts = []
        deal_closed_counts = []

        # Calculate total deal value and deal count for each month in the date range
        for month_end in date_range:
            total_value, deal_created_count, deal_closed_count = pipeline_value_and_count_at_month(df, month_end)
            months.append(month_end.strftime('%Y-%m'))
            as_at_date.append(month_end)
            total_values.append(total_value)  # Store total value
            deal_created_counts.append(deal_created_count)  # Store deal count
            deal_closed_counts.append(deal_closed_count)  # Store deal count

        
        # Create a DataFrame to return
        trend_df = pd.DataFrame({
            'Month': months,
            'As At Date': as_at_date,
            'Total Deal Value': total_values,
            'Deals Created Count': deal_created_counts,
            'Deals Closed Count': deal_closed_counts
        })
    
        return trend_df

    @st.cache_data(show_spinner=False)
    def plot_pipeline_trend(_self, trend_df, start_month, end_month):
        """Plots 'Deals Closed Count' and 'Deals Created Count' on a bar chart, and 'Total Deal Value' on a line chart."""
        
        # Ensure the 'Month' column is in datetime format
        #trend_df['As At Date'] = pd.to_datetime(trend_df['As At Date'])

        # Filter the DataFrame based on the selected month range
        filtered_trend_df = trend_df[
        (trend_df['Month'] >= start_month) & 
        (trend_df['Month'] <= end_month)
    ]
        st.write('Pipeline Data by Month')
        st.dataframe(filtered_trend_df)
        
        # Plot 1: Total Deal Value (line chart)
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        
        ax1.plot(filtered_trend_df['As At Date'], filtered_trend_df['Total Deal Value'], marker='o', linestyle='-', color='b', label='Total Deal Value')
        ax1.set_title('Total Deal Value in Pipeline by Month')
        ax1.set_xlabel('Month')
        ax1.set_ylabel('Total Deal Value')
        ax1.set_xticks(filtered_trend_df['As At Date'])
        ax1.set_xticklabels(filtered_trend_df['As At Date'].dt.strftime('%Y-%m'), rotation=45)
        ax1.grid(True)
        ax1.legend()

        # Display the first figure
        st.pyplot(fig1)

        # Plot 2: Deals Closed and Created Count (bar chart)
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        
        bar_width = 10  # Width of the bars, reduced for better separation
        
        # Plot Deals Closed Count
        ax2.bar(filtered_trend_df['As At Date'] - pd.DateOffset(days=4),  # Offset bars to the left slightly
                filtered_trend_df['Deals Closed Count'], 
                width=bar_width, color='r', alpha=0.7, label='Deals Closed Count')

        # Plot Deals Created Count
        ax2.bar(filtered_trend_df['As At Date'] + pd.DateOffset(days=5),  # Offset bars to the right slightly
                filtered_trend_df['Deals Created Count'], 
                width=bar_width, color='g', alpha=0.7, label='Deals Created Count')

        ax2.set_title('Deals Closed and Created Count by Month')
        ax2.set_xlabel('Month')
        ax2.set_ylabel('Count')
        ax2.set_xticks(filtered_trend_df['As At Date'])
        ax2.set_xticklabels(filtered_trend_df['As At Date'].dt.strftime('%Y-%m'), rotation=45)
        ax2.grid(True)
        
        # Add a legend for the bar chart
        ax2.legend()

        # Display the second figure
        st.pyplot(fig2)

    # # Function to calculate percentage of total deal value reaching target total revenue
    # @st.cache_data(show_spinner=False)
    # def calculate_revenue_percentage(_self, df, sales_targets):
    #     # Assuming 'total_deal_value' is the sum of 'Deal : Total Deal Value' in new_deals_data_filtered
    #     total_deal_value = df['Deal : Total Deal Value'].sum()
    #     target_total_revenue = sales_targets['deal_type']['AllDeal']
        
    #     if target_total_revenue > 0:
    #         percentage_reached = (total_deal_value / target_total_revenue) * 100
    #     else:
    #         percentage_reached = 0
    #     return percentage_reached

    # # Function to visualize the percentage on a donut chart
    # @st.cache_data(show_spinner=False)
    # def plot_total_revenue_donut_chart(_sefl, percentage_reached):
    #     remaining_percentage = 100 - percentage_reached if percentage_reached <= 100 else 0

    #     # Create the donut chart
    #     fig = go.Figure(data=[go.Pie(
    #         labels=['Achieved', 'Remaining'],
    #         values=[percentage_reached, remaining_percentage],
    #         hole=.6,  # Donut chart hole size
    #         marker_colors=['#4CAF50', '#FF4136']  # Custom colors for Achieved and Remaining
    #     )])

    #     # Customize layout
    #     fig.update_traces(textinfo='percent+label')
    #     fig.update_layout(title_text='Revenue Target Achievement', annotations=[dict(text=f'{percentage_reached:.1f}%', x=0.5, y=0.5, font_size=20, showarrow=False)])
        
    #     return fig
    
    # def visualize_metric_over_time(self, deals_data, metrics):
    #     """
    #     Function to visualize a selected revenue/cost type over time (monthly or quarterly) from the deals data.
        
    #     Args:
    #         deals_data (pd.DataFrame): The DataFrame containing the deals data.
    #         metrics (list): List of metric names (revenue/cost types) to select from.
    #     """
    #     col1, col2 = st.columns(2)
    #     # Drop-down box to allow the user to select a revenue/cost type
    #     with col1:
    #         selected_metric = st.selectbox(
    #             "Select Revenue or Cost Type to Visualize",
    #             options=metrics
    #         )
    #     with col2: 
    #         # Drop-down box to allow the user to select the view type (Month or Quarter)
    #         view_by = st.selectbox(
    #             "View By",
    #             options=["Month", "Quarter"]
    #         )

    #     # Check if the selected column exists in the DataFrame
    #     if selected_metric in deals_data.columns:
    #         # Ensure 'Deal : Expected close date' column is in datetime format
    #         deals_data['Date'] = pd.to_datetime(deals_data['Deal : Expected close date'], errors='coerce')

    #         # Sort by date to ensure proper ordering
    #         deals_data = deals_data.sort_values('Date')

    #         # Resample the data by either 'M' (month) or 'Q' (quarter) based on the selected view
    #         if view_by == "Month":
    #             trend_data = deals_data.resample('M', on='Date')[selected_metric].sum().reset_index()
    #             date_format = '%Y-%m'
    #             x_label = 'Month'
    #             plot_title = f'{selected_metric} Over Time (Monthly)'
    #         else:
    #             trend_data = deals_data.resample('Q', on='Date')[selected_metric].sum().reset_index()
    #             date_format = 'Q%q-%Y'
    #             x_label = 'Quarter'
    #             plot_title = f'{selected_metric} Over Time (Quarterly)'

    #         # Calculate the sum value for the metric across the selected timeframe
    #         sum_value = deals_data[selected_metric].sum()

    #         # Set x-tick frequency based on the number of data points
    #         total_periods = trend_data.shape[0]
    #         xtick_freq = 3 if total_periods > 12 else 1  # Adjust x-tick frequency

    #         # Plot the bar chart for each period's value
    #         plt.figure(figsize=(12, 6))  # Increase figure size for readability
    #         plt.bar(trend_data['Date'].dt.strftime(date_format), trend_data[selected_metric], 
    #                 label=f'{selected_metric} ({view_by})', color='skyblue')

    #         # Rotate the x-ticks for better readability
    #         plt.xticks(ticks=trend_data['Date'][::xtick_freq].dt.strftime(date_format), rotation=45, ha='right')

    #         # Enhance the plot with titles and labels
    #         plt.title(plot_title, fontsize=16)
    #         plt.xlabel(x_label, fontsize=14)
    #         plt.ylabel(selected_metric, fontsize=14)

    #         # Add a legend
    #         plt.legend()

    #         # Add gridlines for better readability
    #         plt.grid(visible=True, which='both', linestyle='--', linewidth=0.7)

    #         # Format sum with commas and currency (USD)
    #         formatted_sum_value = f"${sum_value:,.2f}"  # Formats with commas and adds a $ symbol

    #         # Display the sum value outside the chart as a label with currency and grouping
    #         plt.text(1.01, 0.95, f"Total: {formatted_sum_value}", 
    #                 transform=plt.gca().transAxes,
    #                 verticalalignment='top', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))

    #         # Adjust the layout to prevent clipping of tick labels
    #         plt.tight_layout()

    #         # Display the plot in Streamlit
    #         st.pyplot(plt)
    #     else:
    #         st.error(f"Selected metric '{selected_metric}' not found in data.")
    #Duong review

    def visualize_metric_over_time(self, deals_data, metrics):
        """
        Function to visualize a selected revenue/cost type over time (monthly or quarterly) from the deals data.
        
        Args:
            deals_data (pd.DataFrame): The DataFrame containing the deals data.
            metrics (list): List of metric names (revenue/cost types) to select from.
        """
        
        col1, col2 = st.columns(2)
        
        # Drop-down box to allow the user to select a revenue/cost type
        with col1:
            selected_metric = st.selectbox(
                "Select Revenue or Cost Type to Visualize",
                options=metrics
            )
        
        # Drop-down box to allow the user to select the view type (Month or Quarter)
        with col2:
            view_by = st.selectbox(
                "View By",
                options=["Month", "Quarter"]
            )
        
        # Ensure that the 'Deal : Closed date' column is in datetime format
        deals_data['Deal : Closed date'] = pd.to_datetime(deals_data['Deal : Closed date'])
        
        # Group by Month or Quarter
        if view_by == "Month":
            # Extract month and year for grouping
            deals_data['Month'] = deals_data['Deal : Closed date'].dt.strftime('%Y-%m')  # Format as Year-Month (e.g., 2023-07)
            grouped_data = deals_data.groupby('Month').agg({selected_metric: 'sum'}).reset_index()
            
            # Set x-axis label to show month names
            x_axis_label = 'Month'

        else:
            # Group by quarter
            deals_data['Quarter'] = deals_data['Deal : Closed date'].dt.to_period('Q').dt.strftime('Q%q %Y')  # Format as 'Q1 2023'
            grouped_data = deals_data.groupby('Quarter').agg({selected_metric: 'sum'}).reset_index()

            # Set x-axis label to show quarters
            x_axis_label = 'Quarter'

        # Plot the data using Plotly bar chart
        fig = px.bar(
            grouped_data,
            x=grouped_data.columns[0],  # Either 'Month' or 'Quarter' depending on selection
            y=selected_metric,
            title=f'{selected_metric} Over Time ({view_by})',
            labels={grouped_data.columns[0]: x_axis_label, selected_metric: selected_metric}
        )

        # Customize the layout to fit the width and height of the container
        fig.update_layout(
            autosize=True,
            height=500,
            width=800,
            margin=dict(l=40, r=40, t=40, b=40),
            #title_x=0.5,
            xaxis_title=x_axis_label,
            yaxis_title=selected_metric
        )

        return fig


    #Duong updating
    @st.cache_data
    def visualize_actual_vs_target_sales_gauge(_self,won_deals, sales_targets):
        """
        Visualize actual 'Deal : Total Deal Value' of 'Won' Deal, grouped by Owner, compared to target revenue using gauge charts.

        Parameters:
        - won_deals (pd.DataFrame): The DataFrame containing won deal data, including 'Deal : Owner', 'Deal : Total Deal Value', and 'Deal : Deal stage'.
        - sales_targets (dict): A dictionary containing target sales for each owner.

        Returns:
        - fig: A Plotly figure object containing the gauge charts for each sales representative.
        """
        # Group the data by 'Deal : Owner' and sum the 'Deal : Total Deal Value'
        owner_deal_value = won_deals.groupby('Deal : Owner', as_index=False)['Deal : Total Deal Value'].sum()

        # Extract revenue targets from sales_targets
        revenue_targets = sales_targets.get('revenue_targets', {})
        
        # Ensure sales_targets is a Series for mapping
        sales_targets_series = pd.Series(revenue_targets)

        # Add a 'Target' column by mapping each owner's name to the corresponding target sales
        owner_deal_value['Target'] = owner_deal_value['Deal : Owner'].map(sales_targets_series)
        
        # Remove sales reps who do not have defined revenue targets
        owner_deal_value = owner_deal_value[owner_deal_value['Deal : Owner'].isin(revenue_targets.keys())]

        # Ensure all owners with targets are included (even if they have no won deals)
        for owner in revenue_targets.keys():
            if owner not in owner_deal_value['Deal : Owner'].values:
                # Create a new DataFrame for the missing owner with no won deals
                new_row = pd.DataFrame({
                    'Deal : Owner': [owner], 
                    'Deal : Total Deal Value': [0], 
                    'Target': [revenue_targets[owner]]
                })
                owner_deal_value = pd.concat([owner_deal_value, new_row], ignore_index=True)

        # Fill NaN targets with 0 to avoid division errors
        owner_deal_value['Target'].fillna(0, inplace=True)

        # Calculate the percentage of won deal value vs target
        owner_deal_value['Percentage_Won'] = owner_deal_value.apply(
            lambda row: (row["Deal : Total Deal Value"] / row['Target']) * 100 if row['Target'] > 0 else 0,
            axis=1
        )

        # Initialize a list to store each owner's gauge chart
        figures = []

        # Create a gauge chart for each owner
        for index, row in owner_deal_value.iterrows():
            owner = row['Deal : Owner']
            won_value = row['Deal : Total Deal Value']
            target_value = row['Target']
            percentage_won = row['Percentage_Won']

            # Create a gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=percentage_won,
                title={'text': f"{owner}<br>Progress to Target", 'font': {'size': 20}},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "steelblue"},
                    'steps': [
                        {'range': [0, 100], 'color': "lightgray"},
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 100
                    }
                }
            ))

            st.plotly_chart(fig, use_container_width=True)


    def visualize_actual_vs_target_sales(_self, won_deals, sales_targets):
        """
        Visualize actual 'Deal : Total Deal Value' of 'Won' Deal, grouped by Owner, compared to target revenue.

        Parameters:
        - won_deals (pd.DataFrame): The DataFrame containing won deal data, including 'Deal : Owner', 'Deal : Total Deal Value', and 'Deal : Deal stage'.
        - sales_targets (dict): A dictionary containing target sales for each owner.

        Returns:
        - None: Displays the comparison visualization in Streamlit.
        """
        # Group the data by 'Deal : Owner' and sum the 'Deal : Total Deal Value'
        owner_deal_value = won_deals.groupby('Deal : Owner', as_index=False)['Deal : Total Deal Value'].sum()

        # Extract revenue targets from sales_targets
        revenue_targets = sales_targets.get('revenue_targets', {})
        
        # Ensure sales_targets is a Series for mapping
        sales_targets_series = pd.Series(revenue_targets)

        # Add a 'Target' column by mapping each owner's name to the corresponding target sales
        owner_deal_value['Target'] = owner_deal_value['Deal : Owner'].map(sales_targets_series)
        
        # Remove sales reps who do not have defined revenue targets
        owner_deal_value = owner_deal_value[owner_deal_value['Deal : Owner'].isin(revenue_targets.keys())]

        # Ensure all owners with targets are included (even if they have no won deals)
        for owner in revenue_targets.keys():
            if owner not in owner_deal_value['Deal : Owner'].values:
                # Create a new DataFrame for the missing owner with no won deals
                new_row = pd.DataFrame({
                    'Deal : Owner': [owner], 
                    'Deal : Total Deal Value': [0], 
                    'Target': [revenue_targets[owner]]
                })
                owner_deal_value = pd.concat([owner_deal_value, new_row], ignore_index=True)

        # Fill NaN targets with 0 to avoid division errors (though all should have targets by now)
        owner_deal_value['Target'].fillna(0, inplace=True)

        # Convert the 'Target' column to numeric
        owner_deal_value['Target'] = pd.to_numeric(owner_deal_value['Target'], errors='coerce')

        # Calculate the percentage of won deal value vs target
        owner_deal_value['Percentage_Won'] = owner_deal_value.apply(
            lambda row: (row["Deal : Total Deal Value"] / row['Target']) * 100 if row['Target'] > 0 else 0,
            axis=1
        )

        # Sort the owners by 'Percentage_Won' in descending order
        owner_deal_value = owner_deal_value.sort_values('Percentage_Won', ascending=False)

        # Visualize the data in Streamlit
        for index, row in owner_deal_value.iterrows():
            owner = row['Deal : Owner']
            won_value = row['Deal : Total Deal Value']
            target_value = row['Target']
            percentage_won = row['Percentage_Won']

            # Display owner name, progress bar, and target sales in three columns
            col1, col2, col3 = st.columns([1, 5, 1])  # Adjust column width ratios
            
            with col1:
                # Left-aligned owner name with no wrapping and small space between rows
                st.write(f"<div style='text-align: left; white-space: nowrap; margin-bottom: 10px;'>{owner}</div>", unsafe_allow_html=True)

            with col2:
                # Display the progress bar with larger size and percentage on it
                bar_color = 'steelblue' if percentage_won > 0 else 'lightgray'
                progress_text = f"{percentage_won:.2f}%"

                # Display bar with white text on progress bar for better contrast
                st.write(f"""
                    <div style="background-color: lightgray; height: 30px; position: relative; border-radius: 4px;">
                        <div style="width: {min(percentage_won, 100)}%; background-color: {bar_color}; height: 100%; border-radius: 4px; text-align: center; color: white;">
                            {progress_text}
                        </div>
                    </div>
                """, unsafe_allow_html=True)

            with col3:
                # Right-aligned target sales value using format_number function
                formatted_target_value = _self.format_number(target_value)
                st.write(f"<div style='text-align: left;'>{formatted_target_value}</div>", unsafe_allow_html=True)

    
    @st.cache_data
    def calculate_revenue_data(_self, df, sales_targets):
        """
        Calculate the revenue percentage for total, recurring, and non-recurring deals vs the pre-defined.
        
        Parameters:
        df: DataFrame containing deal data.
        sales_targets: Dictionary containing target values for different deal types.
        
        Returns:
        A dictionary containing total revenue, recurring revenue, and non-recurring revenue, 
        along with their corresponding progress percentages.
        """
        # Fetch targets from sales_targets dictionary
        target_total_revenue = sales_targets['deal_type']['AllDeal']
        target_non_recurring = sales_targets['deal_type']['NonRecurring']
        target_recurring = sales_targets['deal_type']['Recurring']
        
        # Define the recurring and non-recurring project types
        recurring_project_types = ['ARR']
        non_recurring_project_types = [ptype for ptype in df['Deal : Project type'].unique() if ptype not in recurring_project_types]

        # Calculate achieved revenue for each deal type
        total_deal_value = df['Deal : Total Deal Value'].sum()
        recurring_revenue = df[df['Deal : Project type'].isin(recurring_project_types)]['Deal : Total Deal Value'].sum()
        non_recurring_revenue = df[df['Deal : Project type'].isin(non_recurring_project_types)]['Deal : Total Deal Value'].sum()

        # Calculate progress percentages
        total_progress = (total_deal_value / target_total_revenue) * 100 if target_total_revenue > 0 else 0
        recurring_progress = (recurring_revenue / target_recurring) * 100 if target_recurring > 0 else 0
        non_recurring_progress = (non_recurring_revenue / target_non_recurring) * 100 if target_non_recurring > 0 else 0

        return {
            'total_progress': total_progress,
            'recurring_progress': recurring_progress,
            'non_recurring_progress': non_recurring_progress,
            'total_deal_value': total_deal_value,
            'recurring_revenue': recurring_revenue,
            'non_recurring_revenue': non_recurring_revenue
        }

    @st.cache_data        
    def create_donut_chart(_self, progress, title, color):
        """
        Create donut charts for revenue target achievement and progress using Altair.
        
        Parameters:
        revenue_data: Dictionary containing calculated revenue percentages.
        
        Returns:
        Altair charts showing total revenue, recurring, and non-recurring progress as donut charts.
        """
        remaining_percentage = 100 - progress if progress <= 100 else 0
        chart_color = color

        source = pd.DataFrame({
            "Topic": ['', title],
            "% value": [remaining_percentage, progress]
        })

        source_bg = pd.DataFrame({
            "Topic": ['', title],
            "% value": [100, 0]
        })

        plot = alt.Chart(source).mark_arc(innerRadius=45, cornerRadius=25).encode(
            theta="% value",
            color=alt.Color("Topic:N",
                            scale=alt.Scale(
                                domain=[title, ''],
                                range=chart_color),
                            legend=None),
        ).properties(width=130, height=130)

        text = plot.mark_text(align='center', color=chart_color[0], font="Lato", fontSize=20, fontWeight=700).encode(
            text=alt.value(f'{progress:.1f}%')
        )

        plot_bg = alt.Chart(source_bg).mark_arc(innerRadius=45, cornerRadius=20).encode(
            theta="% value",
            color=alt.Color("Topic:N",
                            scale=alt.Scale(
                                domain=[title, ''],
                                range=chart_color),
                            legend=None),
        ).properties(width=130, height=130)

        return plot_bg + plot + text



    
    def plot_avg_days_to_close(self, df):
        # Filter out rows where 'Deal : Owner' contains 'Inactive'
        df_filtered = df[~df['Deal : Owner'].str.contains('Inactive', na=False)]
        
        # Calculate 'Days to Close'
        df_filtered['Days to Close'] = (df_filtered['Deal : Closed date'] - df_filtered['Deal : Created at']).dt.days
        
        # Calculate average 'Days to Close' by 'Deal : Owner'
        avg_days_to_close = df_filtered.groupby('Deal : Owner', as_index=False)['Days to Close'].mean()
        
        # Sort values from largest to smallest
        avg_days_to_close = avg_days_to_close.sort_values(by='Days to Close', ascending=False)
        
        # Plot the results
        fig = px.bar(avg_days_to_close, x='Deal : Owner', y='Days to Close')
        st.plotly_chart(fig)
        
    @st.cache_data
    def plot_deal_value_growth_rate(_self, df):       
        # Extract month from 'Deal : Closed date'
        df['Month'] = pd.to_datetime(df['Deal : Closed date']).dt.to_period('M')
        
        # Calculate total deal value growth by month
        deal_value_growth = df.groupby('Month', as_index=False)['Deal : Total Deal Value'].sum()
        
        # Format the month for better readability
        deal_value_growth['Month'] = deal_value_growth['Month'].dt.strftime('%Y-%m')
        
        # Plot the results
        fig = px.line(deal_value_growth, x='Month', y='Deal : Total Deal Value')
        st.plotly_chart(fig)
        
    #Function to visualize current year pipeline deal value compare to last year
    @st.cache_data
    def visualize_pipeline_deal_values_current_last_year(_self, df, view_by='Monthly'):
        # Convert 'Month' column to datetime if not already in datetime format
        df['Month'] = pd.to_datetime(df['Month'], format='%Y-%m')

        # Extract current and last year from the data
        last_year = df['Month'].dt.year.max()
        current_year = last_year + 1

        # Resample data based on the view_by selection (monthly or quarterly)
        if view_by == 'Monthly':
            # Resample by month and aggregate numeric columns
            data_resampled = df.set_index('Month').resample('M').sum(numeric_only=True).reset_index()

            # Generate a complete range of months from January to the max month in the data
            min_month = pd.Timestamp(f'{last_year}')
            max_month = data_resampled['Month'].max()

            # Generate the x-axis labels from January to max month of the current year
            x_axis = pd.date_range(start=min_month, end=max_month, freq='M')

            # Set the tick format for the x-axis
            xaxis_tickformat = '%b'
            xaxis_title = 'Month'

        elif view_by == 'Quarterly':
            # Resample by quarter and aggregate numeric columns
            data_resampled = df.set_index('Month').resample('Q').sum(numeric_only=True).reset_index()

            # Generate a complete range of quarters from Q1 to the max quarter in the data
            min_quarter = pd.Timestamp(f'{current_year}-01-01')
            max_quarter = data_resampled['Month'].max()

            # Generate the x-axis labels from Q1 to the max quarter of the current year
            x_axis = pd.date_range(start=min_quarter, end=max_quarter, freq='Q')

            # Set the tick format for the x-axis
            xaxis_tickformat = 'Q%q'
            xaxis_title = 'Quarter'

        else:
            raise ValueError("view_by must be either 'Monthly' or 'Quarterly'")

        # Create the bar chart using Plotly
        fig = go.Figure()

        # Add bars for 'Total Deal Value' (current year)
        fig.add_trace(go.Bar(
            x=x_axis,
            y=data_resampled['Current Year Total Deal Value'],
            name=f'{current_year}',
            marker_color='blue'
        ))

        # Add bars for 'Last Year Total Deal Value' (last year)
        fig.add_trace(go.Bar(
            x=x_axis,
            y=data_resampled['Total Deal Value'],
            name=f'{last_year}',
            marker_color='orange'
        ))

        # Update layout of the plot
        fig.update_layout(
            title={
                'text': f'Pipeline Deals Accumulated Value Back Date: {current_year} vs {last_year}',  # Add a title
                'font': {
                    'size': 20,  # Font size for the title (equivalent to st.markdown("#### Text value"))
                    'color': 'black'  # Optional: specify the color
                },
                'x': 0.7,  # Center the title
                'xanchor': 'right'  # Center anchor
            },
            xaxis_title=xaxis_title,
            yaxis_title='Total Deal Value',
            margin=dict(t=30),
            height=450,
            barmode='group',  # Group bars next to each other
            xaxis_tickformat=xaxis_tickformat,  # Format based on view_by
            template='plotly_white',
            xaxis_tickmode='array',
            xaxis_tickvals=x_axis,  # Set x-axis ticks manually
            xaxis_ticktext=[tick.strftime('%b') if view_by == 'Monthly' else f'Q{tick.quarter}' for tick in x_axis],  # Customize tick labels
            #legend_title="Year Comparison",
            #xaxis_tickangle=-45  # Rotate x-axis labels for better readability
        )

        return fig
















    





    





   










    