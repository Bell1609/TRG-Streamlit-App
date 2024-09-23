import pandas as pd
import functools as ft
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

class Ticket_Data():
    def get_raw(self, file):
        try:
            raw_data = pd.read_csv(file)
        except Exception:
        # try:
            raw_data = pd.read_excel(file)
        except:
            print("Use .csv or .xlsx files only!")
            return
        return raw_data
    
    def create_ticket_dataframe(self, raw_data):
        #raw_data.loc[~raw_data['Group Company'].isnull(), 'Client code'] = raw_data['Group Company']
        raw_data.loc[~raw_data['Brand'].isnull(), 'Client code'] = raw_data['Brand']
        #fd_customer = raw_data[raw_data['TRG Customer']==True]
        
        return raw_data
    
           
    # update the function to flexible using the id field
    def create_ticket_dataframe_to_download(self, df_kmeans, raw_data, id_field='Contact ID'):
        # Rename 'Ticket ID' column in df_kmeans to 'Ticket Count' to avoid conflicts
        df_kmeans = df_kmeans.rename(columns={'Ticket ID': 'Ticket Count'})

        # Merge raw_data with df_kmeans on id_field to include 'Ranking', 'Recency', and 'Ticket Count'
        download_data = raw_data.merge(
            df_kmeans[[id_field, 'Ranking', 'Recency', 'Ticket Count']], 
            on=id_field, 
            how='left'
        )

        # Reorder columns to ensure 'id_field', 'Ranking', 'Recency', and 'Ticket Count' appear first
        columns_order = [id_field, 'Ranking', 'Recency', 'Ticket Count'] + \
                        [col for col in raw_data.columns if col not in [id_field, 'Ranking', 'Recency', 'Ticket Count']]

        # Reorder the DataFrame based on the specified column order
        download_data = download_data[columns_order]

        # Remove any duplicate rows
        download_data = download_data.drop_duplicates()

        # Remove rows where all values are NaN
        download_data = download_data.dropna(how='all')

        return download_data

    #End of change
    
    def create_kmeans_dataframe(self, df_attributes, fd_data, selected_volume_columns, id_field='Contact ID'): 
        def create_clustered_data(kmeans, features_to_scale, scaler):
            # Create a DataFrame with cluster centers, inverse transforming to get original scale
            cluster_centers = pd.DataFrame(
                scaler.inverse_transform(kmeans.cluster_centers_), 
                columns=features_to_scale
            )

            # Add cluster size based on the counts in df_clusters
            cluster_sizes = df_clusters['Cluster'].value_counts().sort_index().values
            if len(cluster_centers) != len(cluster_sizes):
                raise ValueError(f"Mismatch between number of clusters ({len(cluster_centers)}) and cluster sizes ({len(cluster_sizes)})")
            cluster_centers['Cluster Size'] = cluster_sizes

            # Label the clusters
            for i in range(len(cluster_centers)):
                cluster_centers.loc[i, 'Cluster'] = f'Cluster {i}'

            # Reorder columns (including 'Cluster Size')
            cluster_centers = cluster_centers[['Cluster'] + features_to_scale + ['Cluster Size']]

            return cluster_centers

        # Add 'Recency' to the features to be scaled
        features_to_scale = ['Recency'] + selected_volume_columns if 'Recency' not in selected_volume_columns else selected_volume_columns
    
        # Prepare the features for scaling
        df_features = df_attributes[features_to_scale].copy()
        
        # Scaling
        scaler = StandardScaler()
        df_standard = scaler.fit_transform(df_features)

        # Initialize variables for best results
        best_silhouette = -1
        best_kmeans = None
        best_k = None
        best_random_state = None
        best_labels = None

        # Compare if being standardized makes better predictions or not
        # Loop through k values from 3 to 7
        for k in range(3, 8):
            for random_state in range(1, 50):
                kmeans = KMeans(n_clusters=k, random_state=random_state)
                cluster_labels = kmeans.fit_predict(df_standard)
                silhouette_avg = silhouette_score(df_standard, cluster_labels)
                if silhouette_avg > best_silhouette:
                    best_silhouette = silhouette_avg
                    best_kmeans = kmeans
                    best_k = k
                    best_labels = cluster_labels
                    best_random_state = random_state

        # Create DataFrame with the best cluster labels
        clustered_data = pd.DataFrame({id_field: df_attributes[id_field], 'Cluster': best_labels})
        df_clusters = df_attributes.merge(clustered_data, on=id_field, how='left')

        # Add additional company/brand data
        if id_field == 'Contact ID':
            df_clusters_name = df_clusters.merge(fd_data[[id_field, 'AMS','CMS','Systems Environment', 'Valid Maintenance', 'FS TRG Customer', 'Country', 'Industry']], on=id_field, how='left')
        else:
            df_clusters_name = df_clusters.merge(fd_data[[id_field, 'Brand','AMS','CMS','Systems Environment', 'Valid Maintenance', 'FS TRG Customer', 'Country', 'Industry', 'License Qty']], on=id_field, how='left')

        df_clusters_name = df_clusters_name.drop_duplicates()

        # Add cluster rankings
        for i in range(0, best_k):
            df_clusters_name.loc[df_clusters_name['Cluster'] == i, 'Ranking'] = f'Cluster {i}'

        # Generate the cluster centers with inverse-transformed values
        cluster_centers = create_clustered_data(best_kmeans, features_to_scale, scaler)

        return df_clusters_name, cluster_centers, best_silhouette, best_k, best_random_state

    
    def create_df_rfm_grouped_by_id(self, fd_customer, id_field='Contact ID', selected_volume_columns=None):
        # Sub-function to calculate recency
        def create_recency():
            fd_customer['Created time'] = pd.to_datetime(fd_customer['Created time'])
            df_recency = fd_customer.groupby(id_field)['Created time'].max().reset_index()
            df_recency['Recency'] = (df_recency['Created time'].max() - df_recency['Created time']).dt.days
            return df_recency

        # Sub-function to calculate volume
        def create_volume(selected_volume_columns):
            # Check if 'Ticket ID' is in selected_volume_columns and handle it separately
            if 'Ticket ID' in selected_volume_columns:
                df_volume = fd_customer.groupby(id_field)['Ticket ID'].count().reset_index()
                #df_volume = df_volume.rename(columns={'Ticket ID': 'Ticket Volume'})
                # Remove 'Ticket ID' from selected_volume_columns to avoid double processing
                selected_volume_columns = [col for col in selected_volume_columns if col != 'Ticket ID']
            else:
                # Initialize df_volume to an empty DataFrame if 'Ticket ID' is not part of the selected columns
                df_volume = pd.DataFrame({id_field: fd_customer[id_field].unique()})

            # Calculate volume for other selected columns
            if selected_volume_columns:
                for column in selected_volume_columns:
                    if pd.api.types.is_numeric_dtype(fd_customer[column]):
                        volume = fd_customer.groupby(id_field)[column].mean().reset_index()
                    else:
                        volume = fd_customer.groupby(id_field)[column].count().reset_index()
                    # Merge with df_volume
                    df_volume = df_volume.merge(volume, on=id_field, how='left')
            
            return df_volume


        # Generate the recency and volume data
        df_recency = create_recency()
        df_volume = create_volume(selected_volume_columns)

        # Combine the recency and volume data
        df_list = [df_recency[[id_field, 'Recency']], df_volume]

        # Merge all the attributes together
        df_attributes = ft.reduce(lambda left, right: pd.merge(left, right, on=id_field), df_list)

        return df_attributes

   
    def create_helpdesk_performance(self, processed_data, support_percentage, avg_time_tracked):
        # Ensure 'Created time' is in datetime format
        if 'Created time' not in processed_data.columns:
            raise ValueError("The 'Created time' column is missing from the data.")

        processed_data['Created time'] = pd.to_datetime(processed_data['Created time'], errors='coerce')

        # Extract 'Month' from 'Created time' in 'YYYY-MM' format
        processed_data['Month'] = processed_data['Created time'].dt.to_period('M').astype(str)

        # Group data by 'Month'
        helpdesk_performance = processed_data.groupby('Month').agg(
            # 1. Count the unique contacts who have tickets during that month
            Contact_Count=('Contact ID', 'nunique'),
            # 2. Count the unique companies who have tickets during that month
            Company_Count=('Company Name', 'nunique'),
            # 3. Count the number of tickets raised during that month
            Ticket_Count=('Ticket ID', 'count'),
            # 4. Average 1st response time without filtering x > 0
            Average_1st_Response_Time=('First response time (in hrs)', lambda x: x.mean()),
            # 5. Average resolution time without filtering x > 0
            Average_Resolution_Time=('Resolution time (in hrs)', lambda x: x.mean()),
            # 6. FCR calculation (percentage of tickets resolved after first customer interaction)
            FCR=('Ticket ID', lambda x: ((processed_data.loc[x.index, 'Customer interactions'] == 1).sum()) / len(x) * 100),
            # 7. Average time tracked per month
            Average_Time_Tracked=('Time tracked', lambda x: x[x > 0].sum() / (x[x > 0].count()) if x[x > 0].count() > 0 else 0),
            # 8. Count of unique agents handling tickets in the month
            Agent_Count=('Agent', 'nunique')
        ).reset_index()

        # Calculate working days in each month (assuming a standard 5-day work week)
        helpdesk_performance['Working_Days'] = helpdesk_performance['Month'].apply(
            lambda x: pd.date_range(start=x, end=(pd.Period(x) + 1).to_timestamp() - pd.Timedelta(days=1), freq='B').size
        )

       # Calculate the mean value of 'Average_Time_Tracked'
        #third_quartile_average_time_tracked = helpdesk_performance['Average_Time_Tracked'].quantile(0.75)
        third_quartile_ticket_count = helpdesk_performance['Ticket_Count'].quantile(0.75)


        # Calculate Agent_Needed based on the mean value
        helpdesk_performance['Agent_Needed'] = (avg_time_tracked * third_quartile_ticket_count) / (helpdesk_performance['Working_Days'].mean() * 8)

        # Calculate Capacity_Needed based on support_percentage
        helpdesk_performance['Capacity_Needed'] = (helpdesk_performance['Agent_Needed'] * 100) / support_percentage

        return helpdesk_performance

    def create_ticket_and_contact_grouped_by_company(self, processed_data):
        # Ensure 'Created time' and 'Company Name' columns exist
        if 'Created time' not in processed_data.columns:
            raise ValueError("The 'Created time' column is missing from the data.")
        if 'Company Name' not in processed_data.columns:
            raise ValueError("The 'Company Name' column is missing from the data.")
        
        # Ensure 'Created time' is in datetime format
        processed_data['Created time'] = pd.to_datetime(processed_data['Created time'], errors='coerce')

        # Extract 'Month' from 'Created time' in 'YYYY-MM' format
        processed_data['Month'] = processed_data['Created time'].dt.to_period('M').astype(str)

        # Group data by 'Month' and 'Company Name'
        ticket_and_contact_grouped_by_company = processed_data.groupby(['Month', 'Company Name']).agg(
            # 1. Count the number of tickets for each company in each month
            Ticket_Count=('Ticket ID', 'count'),
            # 2. Count the unique contacts for each company in each month
            Contact_Count=('Contact ID', 'nunique'),
            # 3. Calculate FCR for each company in each month (First Call Resolution percentage)
            
        )


        # Reset the index to have 'Month' and 'Company Name' as regular columns
        ticket_and_contact_grouped_by_company = ticket_and_contact_grouped_by_company.reset_index()

        return ticket_and_contact_grouped_by_company
    
    def load_and_transform_employees(self, file):
        # Load the Excel file
        employees = pd.read_excel(file, engine='openpyxl')

        # Parse the dates and fill missing 'Last Day' with today's date
        employees['Joining Date'] = pd.to_datetime(employees['Joining Date'], format='%Y-%m-%d')
        employees['Last Day'] = pd.to_datetime(employees['Last Day'], format='%Y-%m-%d', errors='coerce')
        employees['Last Day'].fillna(pd.Timestamp.today(), inplace=True)

        # Find the min and max months based on 'Joining Date' and 'Last Day'
        min_month = employees['Joining Date'].min().to_period('M')
        max_month = pd.Timestamp.today().to_period('M')

        # Create a list of months from min to max month
        month_range = pd.period_range(min_month, max_month, freq='M')

        # Generate the new dataframe
        rows = []
        for _, row in employees.iterrows():
            for month in month_range:
                status = 'Employed'
                if month > row['Last Day'].to_period('M'):
                    status = 'Unemployed'
                elif month < row['Joining Date'].to_period('M'):
                    status = 'Unemployed'
                # Append the new row with 'Month' converted to string format (YYYY-MM)
                rows.append({
                    'Month': month.strftime('%Y-%m'),  # Convert to string format
                    'Staff Name': row['Staff Name'],
                    'Dept': row['Dept'],
                    'Status': status,
                    'Type': row['Type']  # Include the Type column
                })

        employees_transformed = pd.DataFrame(rows)
        return employees_transformed

    







