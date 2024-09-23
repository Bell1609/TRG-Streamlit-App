import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import squarify
import sweetviz as sv

from matplotlib.ticker import MaxNLocator
from matplotlib.patches import Patch

class Ticket_Graph_Drawing():
    def recency_graph(self, id_field, df_attributes):
        plt.figure()
        if id_field == 'Contact ID':
            sns.histplot(df_attributes[df_attributes['Recency']<3000]['Recency'], kde=True)
        else:
            sns.histplot(df_attributes[df_attributes['Recency']<500]['Recency'], kde=True)
        
        plt.xlabel('Recency')
        plt.ylabel('Number of customers')
        plt.title(f"Number of customers based on recency")
        
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        
        return plt.gcf()
    

    def interactions_graph(self, df_attributes, selected_volume_columns):
        plt.figure()

        # Initialize plotting variables
        customer_interactions_selected = 'Customer interactions' in selected_volume_columns
        agent_interactions_selected = 'Agent interactions' in selected_volume_columns

        # Prepare the data for plotting
        title = "Interactions Distribution"
        bins = np.arange(0, 90, 10)  # Bins from 0 to 90, adjust as needed
        width = 4  # Width of bars for side-by-side placement

        # Case 1: Plot only Customer Interactions
        if customer_interactions_selected and not agent_interactions_selected:
            customer_data = df_attributes['Customer interactions'].clip(lower=0)

            # Plot histogram for Customer Interactions
            counts, edges = np.histogram(customer_data, bins=bins)
            plt.bar(edges[:-1], counts, width=width, color='red', label='Customer interactions')

            title = "Customer Interactions Distribution"

        # Case 2: Plot only Agent Interactions
        elif agent_interactions_selected and not customer_interactions_selected:
            agent_data = df_attributes['Agent interactions'].clip(lower=0)

            # Plot histogram for Agent Interactions
            counts, edges = np.histogram(agent_data, bins=bins)
            plt.bar(edges[:-1], counts, width=width, color='blue', label='Agent interactions')

            title = "Agent Interactions Distribution"

        # Case 3: Plot both Customer and Agent Interactions side by side
        elif customer_interactions_selected and agent_interactions_selected:
            customer_data = df_attributes['Customer interactions'].clip(lower=0)
            agent_data = df_attributes['Agent interactions'].clip(lower=0)

            # Get histogram counts for customer and agent interactions
            customer_counts, edges = np.histogram(customer_data, bins=bins)
            agent_counts, _ = np.histogram(agent_data, bins=bins)

            # Create bin centers and adjust for side-by-side bar placement
            bin_centers = (edges[:-1] + edges[1:]) / 2

            # Plot customer interactions on one side of the bins
            plt.bar(bin_centers - width/2, customer_counts, width=width, color='red', label='Customer interactions')

            # Plot agent interactions on the other side of the bins
            plt.bar(bin_centers + width/2, agent_counts, width=width, color='blue', label='Agent interactions')

            # Add line plot to show the total interaction trend (optional)
            total_interactions = customer_counts + agent_counts
            plt.plot(bin_centers, total_interactions, color='green', label='Total interactions', linestyle='--')

            title = "Customer and Agent Interactions Distribution"

        else:
            raise ValueError("Neither 'Customer interactions' nor 'Agent interactions' selected for plotting.")

        # Set labels and title
        plt.xlabel('Interactions Count')
        plt.ylabel('Number of Customers')
        plt.title(title)
        plt.legend()

        # Ensure x-axis ticks are integers
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

        return plt.gcf()

   
    def tickets_graph(self, id_field, df_attributes):
        plt.figure()
        if id_field == 'Contact ID':
            sns.histplot(df_attributes[df_attributes['Ticket ID']<50]['Ticket ID'], kde=True, color='orange', edgecolor='lightblue')
        else:
            sns.histplot(df_attributes[df_attributes['Ticket ID']<500]['Ticket ID'], kde=True, color='orange', edgecolor='lightblue')
        
        plt.xlabel('Ticket Count')
        plt.ylabel('Number of customers')
        plt.title(f"Number of customers based on ticket count")
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        
        return plt.gcf()

    def time_tracked_graph(self, df_attributes):
        plt.figure()
        
        # Plot histogram for 'Time tracked'
        sns.histplot(df_attributes[df_attributes['Time tracked'] < 20]['Time tracked'], kde=True, color='blue')
        
        plt.xlabel('Time tracked')
        plt.ylabel('Number of customers')
        plt.title(f"Number of customers based on time tracked")
        
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        
        return plt.gcf()

    def first_response_time_graph(self, df_attributes):
        plt.figure()
        
        # Convert 'First response time (in hrs)' to minutes
        df_attributes['First response time (in mins)'] = df_attributes['First response time (in hrs)'] * 60
        
        # Plot histogram for 'First response time (in mins)'
        sns.histplot(df_attributes[df_attributes['First response time (in mins)'] < 90]['First response time (in mins)'], kde=True, color='green')
        #sns.histplot(df_attributes['First response time (in mins)'], kde=True, color='green')
        
        plt.xlabel('First response time (in minutes)')
        plt.ylabel('Number of customers')
        plt.title('Number of customers based on first response time')
        
        # # Set x-axis limits to range from 0 to 90 minutes
        # plt.xlim(0, 90)
        
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        
        return plt.gcf()


    def resolution_time_graph(self, df_attributes):
        plt.figure()
        
        # Plot histogram for 'Resolution time (in hrs)'
        sns.histplot(df_attributes[df_attributes['Resolution time (in hrs)'] < 50]['Resolution time (in hrs)'],  kde=True, color='purple')
        
        plt.xlabel('Resolution time (in hrs)')
        plt.ylabel('Number of customers')
        plt.title(f"Number of customers based on resolution time")
        
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        
        return plt.gcf()


    def treemap_drawing(self, cluster_centers, selected_volume_columns):
        plt.figure()
        sns.set_style(style="whitegrid")  # Set seaborn plot style
        sizes = cluster_centers['Cluster Size']  # Proportions of the categories

        # Generate random colors for each cluster
        unique_clusters = cluster_centers['Cluster'].unique()
        random.seed(50)  # Optional: Set seed for reproducibility
        colors = {cluster: f'#{random.randint(0, 0xFFFFFF):06x}' for cluster in unique_clusters}

        total_customers = cluster_centers['Cluster Size'].sum()

        squarify.plot(
            sizes=sizes,
            alpha=0.6, 
            color=[colors[cluster] for cluster in cluster_centers['Cluster']],
            label=cluster_centers['Cluster']
        ).axis('off')

        # Creating custom legend
        handles = []

        for i in cluster_centers.index:
            label = f"{cluster_centers.loc[i, 'Cluster']}"

            # Add fields dynamically based on selected_volume_columns
            label += f"\n{cluster_centers.loc[i, 'Recency']:.0f} days"
            if 'Ticket ID' in selected_volume_columns:
                label += f"\n{cluster_centers.loc[i, 'Ticket ID']:.0f} tickets created"
            if 'Time tracked' in selected_volume_columns:
                label += f"\n{cluster_centers.loc[i, 'Time tracked']:.1f} hours tracked"
            if 'First response time (in hrs)' in selected_volume_columns:
                label += f"\nFirst response: {cluster_centers.loc[i, 'First response time (in hrs)']:.1f} hrs"
            if 'Resolution time (in hrs)' in selected_volume_columns:
                label += f"\nResolution time: {cluster_centers.loc[i, 'Resolution time (in hrs)']:.1f} hrs"
            if 'Customer interactions' in selected_volume_columns and 'Agent interactions' in selected_volume_columns:
                label += f"\n{cluster_centers.loc[i, 'Customer interactions']:.0f}/{cluster_centers.loc[i, 'Agent interactions']:.0f} Customer/agent interactions"

            # Cluster size and percentage of total customers
            label += f"\n{cluster_centers.loc[i, 'Cluster Size']:.0f} Customers ({cluster_centers.loc[i, 'Cluster Size'] / total_customers * 100:.1f}%)"

            # Append to handles for the legend
            handles.append(Patch(facecolor=colors[cluster_centers.loc[i, 'Cluster']], label=label))

        plt.legend(handles=handles, loc='center left', bbox_to_anchor=(1, 0.5), fontsize='large')
        plt.title('Ticketing Customer Segmentation Treemap', fontsize=20)

        return total_customers, plt.gcf()


    def visualize_ticket_activity_by_month(self, df):
        # Ensure 'Created time' is in datetime format
        if 'Created time' in df.columns:
            df['Created time'] = pd.to_datetime(df['Created time'], errors='coerce')

            # Create a new column 'Created month' with the format 'YYYY-MM'
            df['Created month'] = df['Created time'].dt.to_period('M').astype(str)

            # Group by 'Created month' and count unique 'Contact ID', 'Company Name', and 'Agent'
            monthly_activity = df.groupby('Created month').agg(
                contacts=('Contact ID', 'nunique'),
                companies=('Company Name', 'nunique'),
                agents=('Agent', 'nunique')  # Count the unique number of agents per month
            ).reset_index()

            # Create the plot
            fig, ax1 = plt.subplots(figsize=(10, 6))

            # Plot contacts and companies on the left y-axis
            ax1.set_xlabel('Month (YYYY-MM)')
            ax1.set_ylabel('Count (Contacts & Companies)', color='tab:blue')
            ax1.plot(monthly_activity['Created month'], monthly_activity['contacts'], label='Contacts', marker='o', color='tab:blue')
            ax1.plot(monthly_activity['Created month'], monthly_activity['companies'], label='Companies', marker='s', color='tab:green')
            ax1.tick_params(axis='y', labelcolor='tab:blue')

            # Create a second y-axis to plot the agent count
            ax2 = ax1.twinx()
            ax2.set_ylabel('Agent Count', color='tab:orange')
            ax2.plot(monthly_activity['Created month'], monthly_activity['agents'], label='Agents', marker='^', color='tab:orange')
            ax2.tick_params(axis='y', labelcolor='tab:orange')

            # Rotate x-axis labels to be vertical
            ax1.set_xticks(range(len(monthly_activity['Created month'])))
            ax1.set_xticklabels(monthly_activity['Created month'], rotation=90, ha='center')

            # Title and layout adjustments
            fig.suptitle('Monthly Ticket Activity by Contacts, Companies, and Agents')
            plt.tight_layout()

            return fig  # Return the created figure for further use
        else:
            raise ValueError("The 'Created time' column is missing from the data.")



    def visualize_ticket_volume_by_month(self, df):
        # Ensure 'Created time' is in datetime format
        df['Created time'] = pd.to_datetime(df['Created time'], errors='coerce')

        # Extract the month in 'yyyy-mm' format
        df['Created month'] = df['Created time'].dt.to_period('M').astype(str)

        # Group by 'Created month' and count ticket volume and the number of unique agents
        monthly_data = df.groupby('Created month').agg(
            ticket_count=('Ticket ID', 'size'),  # Count the number of tickets
            agent_count=('Agent', 'nunique')  # Count unique agents
        ).reset_index()

        # Plot the data
        fig, ax1 = plt.subplots(figsize=(10, 6))

        # Plot ticket volume (count) on the left y-axis
        ax1.set_xlabel('Month')
        ax1.set_ylabel('Ticket Count', color='tab:blue')
        ax1.plot(monthly_data['Created month'], monthly_data['ticket_count'], label='Ticket Volume', marker='o', color='tab:blue')
        ax1.tick_params(axis='y', labelcolor='tab:blue')

        # Create another y-axis to plot agent count
        ax2 = ax1.twinx()
        ax2.set_ylabel('Agent Count', color='tab:orange')
        ax2.plot(monthly_data['Created month'], monthly_data['agent_count'], label='Agent Count', marker='s', color='tab:orange')
        ax2.tick_params(axis='y', labelcolor='tab:orange')

        # Set x-axis labels to be vertical
        ax1.set_xticks(range(len(monthly_data['Created month'])))
        ax1.set_xticklabels(monthly_data['Created month'], rotation=90, ha='center')


        # Title and layout
        plt.title('Monthly Ticket Volume and Agent Count')
        plt.tight_layout()

        return fig  # Return the figure for rendering


    def visualize_helpdesk_performance_column(self, df_helpdesk_performance, column_name):
        # Set the plotting theme
        sns.set_theme(style="whitegrid")

        # Check if the provided column name exists in the dataframe
        if column_name not in df_helpdesk_performance.columns:
            raise ValueError(f"Column '{column_name}' not found in the dataframe.")

        # Check if 'Agent_Count', 'Capacity_Needed', and 'Agent_Needed' columns exist
        if 'Agent_Count' not in df_helpdesk_performance.columns:
            raise ValueError("The 'Agent_Count' column is missing from the dataframe.")
        if 'Capacity_Needed' not in df_helpdesk_performance.columns:
            raise ValueError("The 'Capacity_Needed' column is missing from the dataframe.")
        if 'Agent_Needed' not in df_helpdesk_performance.columns:
            raise ValueError("The 'Agent_Needed' column is missing from the dataframe.")

        # Create a new figure
        fig, ax1 = plt.subplots(figsize=(10, 6))

        # Plot the selected column on the primary y-axis (left)
        sns.lineplot(
            x='Month', y=column_name, data=df_helpdesk_performance, ax=ax1, marker='o', label=column_name.replace("_", " ")
        )
        ax1.set_title(f'Monthly {column_name.replace("_", " ")} with Agent Count, Capacity Needed, and Agent Needed')
        ax1.set_xlabel('Month')
        ax1.set_ylabel(column_name.replace("_", " "), color='blue')
        ax1.tick_params(axis='x', rotation=90)  # Rotate x-axis labels for readability
        ax1.tick_params(axis='y', labelcolor='blue')

        # Create a secondary y-axis for the agent count and capacity needed (right)
        ax2 = ax1.twinx()
        
        # Plot Agent Count with a solid line style
        sns.lineplot(
            x='Month', y='Agent_Count', data=df_helpdesk_performance, ax=ax2, marker='o', color='orange', label='Agent Count', linestyle='-'
        )
        
        # Plot Capacity Needed with a dashed line style
        sns.lineplot(
            x='Month', y='Capacity_Needed', data=df_helpdesk_performance, ax=ax2, color='green', label='Capacity Needed', linestyle='--'
        )
        
        # Plot Agent Needed with a dotted line style in red
        sns.lineplot(
            x='Month', y='Agent_Needed', data=df_helpdesk_performance, ax=ax2, color='red', label='Agent Needed', linestyle='--'
        )

        ax2.set_ylabel('Agents', color='orange')
        ax2.tick_params(axis='y', labelcolor='orange')

        # Combine legends from both axes
        # Set legend for ax1 only (excluding ax2 labels)
        lines, labels = ax1.get_legend_handles_labels()
        ax1.legend(lines, labels, loc='upper left')

        # Set the x-axis ticks manually to display every 3rd month
        months = df_helpdesk_performance['Month'].unique()
        ax1.set_xticks(range(0, len(months), 3))  # Tick positions every 3 months
        ax1.set_xticklabels(months[::3], rotation=90, ha='right')

        # Adjust layout for better visualization
        plt.tight_layout()

        # Return both the figure and the primary axis
        return fig, ax1




    def visualize_performance_with_employees(self, df_performance, column_name):
        # Set the plotting theme
        sns.set_theme(style="whitegrid")

        # Check if the provided column name exists in the dataframe
        if column_name not in df_performance.columns:
            raise ValueError(f"Column '{column_name}' not found in the dataframe.")

        # Check if 'Employed Count' column exists
        if 'Employed Count' not in df_performance.columns:
            raise ValueError("The 'Employed Count' column is missing from the dataframe.")

        # Create a new figure
        fig, ax1 = plt.subplots(figsize=(10, 6))

        # Plot the selected column on the primary y-axis (left)
        sns.lineplot(
            x='Month', y=column_name, data=df_performance, ax=ax1, marker='o', label=column_name.replace("_", " ")
        )
        ax1.set_title(f'Monthly {column_name.replace("_", " ")} and Employed Count')
        ax1.set_xlabel('Month')
        ax1.set_ylabel(column_name.replace("_", " "), color='blue')
        ax1.tick_params(axis='x', rotation=90)  # Rotate x-axis labels for readability
        ax1.tick_params(axis='y', labelcolor='blue')

        # Create a secondary y-axis for the employed count (right)
        ax2 = ax1.twinx()
        sns.lineplot(
            x='Month', y='Employed Count', data=df_performance, ax=ax2, marker='o', color='orange', label='Employed Count'
        )
        ax2.set_ylabel('Employed Count', color='orange')
        ax2.tick_params(axis='y', labelcolor='orange')

        # Combine legends from both axes
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc='upper left')

        # Set the x-axis ticks manually to display every 3rd month
        months = df_performance['Month'].unique()
        ax1.set_xticks(range(0, len(months), 3))  # Tick positions every 3 months
        ax1.set_xticklabels(months[::3], rotation=45, ha='right')

        # Adjust layout for better visualization
        plt.tight_layout()

        # Return both the figure and the primary axis
        return fig, ax1


    