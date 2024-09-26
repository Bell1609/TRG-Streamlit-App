import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from datetime import datetime


class Data_Handling():
    def get_raw(self, file):
        try:
            raw_data = pd.read_csv(file)
        except Exception:
        # try:
            raw_data = pd.read_excel(file)
        except:
            print("Use .csv or .xlsx files only!")
            return
        # raw_data['AccountName'] = raw_data['AccountName'].str.strip()
        return raw_data
    
    
    """ def create_rfm_dataframe(self, df, id_field):
        # Initialize the RFM DataFrame using the unique account IDs
        df_rfm = pd.DataFrame(df[id_field].unique())
        df_rfm.columns = [id_field]
        
        # Convert 'Deal : Closed date' to datetime
        df['Deal : Expected close date'] = pd.to_datetime(df['Deal : Expected close date'], dayfirst=True, format='mixed')
        
        # Calculate Recency
        last_purchase = df.groupby(id_field)['Deal : Expected close date'].max().reset_index()
        last_purchase.columns = [id_field, 'CloseDateMax']
        last_purchase['Recency'] = (last_purchase['CloseDateMax'].max() - last_purchase['CloseDateMax']).dt.days
        df_rfm = pd.merge(df_rfm, last_purchase[[id_field, 'Recency']], how='left', on=id_field)
        
        # Calculate Frequency
        df_freq = df.dropna(subset=[id_field]).groupby(id_field)['Deal : Expected close date'].count().reset_index()
        df_freq.columns = [id_field, 'Frequency']
        df_rfm = pd.merge(df_rfm, df_freq, on=id_field)
        
        # Calculate Monetary
        df['Deal : Total Deal Value'] = df['Deal : Total Deal Value'].astype(str).replace('[\$,]', '', regex=True).astype(float)
        df_mone = df.groupby(id_field)['Deal : Total Deal Value'].sum().reset_index()
        df_mone.columns = [id_field, 'Monetary']
        df_rfm = pd.merge(df_rfm, df_mone, on=id_field)
        
        return df_rfm """


    def create_rfm_dataframe(self, df, id_field):
        # Initialize the RFM DataFrame using the unique account IDs
        df_rfm = pd.DataFrame(df[id_field].unique())
        df_rfm.columns = [id_field]

        # Get today's date
        today = pd.to_datetime(datetime.today().date())

        # Convert 'Deal : Expected close date' to datetime
        df['Deal : Expected close date'] = pd.to_datetime(df['Deal : Expected close date'], dayfirst=True, errors='coerce')

        # Adjust 'Expected close date' greater than today
        df['Adjusted Close Date'] = df['Deal : Expected close date'].apply(lambda x: today if x > today else x)

        # Calculate Recency (if expected close date > today, recency will be negative)
        last_purchase = df.groupby(id_field)['Adjusted Close Date'].max().reset_index()
        last_purchase.columns = [id_field, 'CloseDateMax']
        last_purchase['Recency'] = (today - last_purchase['CloseDateMax']).dt.days

        # If the original expected close date is greater than today, set Recency as negative
        last_purchase['Recency'] = last_purchase.apply(
            lambda row: -(row['Recency']) if row['CloseDateMax'] == today else row['Recency'], axis=1
        )

        # Merge Recency into RFM DataFrame
        df_rfm = pd.merge(df_rfm, last_purchase[[id_field, 'Recency']], how='left', on=id_field)

        # Calculate Frequency
        df_freq = df.dropna(subset=[id_field]).groupby(id_field)['Deal : Expected close date'].count().reset_index()
        df_freq.columns = [id_field, 'Frequency']
        df_rfm = pd.merge(df_rfm, df_freq, on=id_field)

        # Calculate Monetary
        df['Deal : Total Deal Value'] = df['Deal : Total Deal Value'].astype(str).replace('[\$,]', '', regex=True).astype(float)
        df_mone = df.groupby(id_field)['Deal : Total Deal Value'].sum().reset_index()
        df_mone.columns = [id_field, 'Monetary']
        df_rfm = pd.merge(df_rfm, df_mone, on=id_field)

        return df_rfm


    
    def create_kmeans_dataframe(self, df_rfm, id_field):
        def create_clustered_data(kmeans):
            # Create a DataFrame with cluster centers
            cluster_centers = pd.DataFrame(
                scaler.inverse_transform(kmeans.cluster_centers_), 
                columns=['Recency', 'Frequency', 'Monetary']
            )

            # Add cluster size
            cluster_sizes = df_kmeans['Cluster'].value_counts().sort_index().values
            if len(cluster_centers) != len(cluster_sizes):
                raise ValueError(f"Mismatch between number of clusters ({len(cluster_centers)}) and cluster sizes ({len(cluster_sizes)})")
            cluster_centers['Cluster Size'] = cluster_sizes
            cluster_centers['Recency'] = np.abs(cluster_centers['Recency'])

            for i in range(len(cluster_centers)):
                cluster_centers.loc[i, 'Cluster'] = f'Cluster {i}'
            cluster_centers = cluster_centers[['Cluster', 'Recency', 'Frequency', 'Monetary', 'Cluster Size']]

            return cluster_centers

        # Copy the original DataFrame
        df_rfm_copy = df_rfm.copy()

        # Select the relevant columns for clustering
        rfm_selected = df_rfm[['Recency', 'Frequency', 'Monetary']]
        
        # Invert the Recency for clustering
        rfm_selected['Recency'] = np.abs(rfm_selected['Recency']) * -1
        
        # Scale the features
        scaler = StandardScaler()
        rfm_standard = scaler.fit_transform(rfm_selected)

        # Initialize variables for the best results
        best_silhouette = -1
        best_kmeans = None
        best_k = None
        best_random_state = None
        best_labels = None

        for c in range(3, 8):
            for n in range(1, 50):
                kmeans = KMeans(n_clusters=c, random_state=n)
                cluster_labels = kmeans.fit_predict(rfm_standard)
                silhouette_avg = silhouette_score(rfm_standard, cluster_labels)
                if best_silhouette < silhouette_avg:
                    best_silhouette = silhouette_avg
                    best_k = c
                    best_random_state = n
                    best_labels = cluster_labels
                    best_kmeans = kmeans

        # Create a DataFrame with the account ID and their corresponding cluster
        clustered_data = pd.DataFrame({id_field: df_rfm_copy[id_field], 'Cluster': best_labels})

        # Merge the clustered data with the original RFM DataFrame
        df_kmeans = pd.merge(df_rfm, clustered_data, on=id_field)

        # Assign cluster rankings
        for i in range(0, best_k):
            df_kmeans.loc[df_kmeans['Cluster'] == i, 'Ranking'] = f'Cluster {i}'

        # Generate cluster centers data
        cluster_centers = create_clustered_data(best_kmeans)

        return df_kmeans, cluster_centers, best_silhouette, best_k, best_random_state

    
    def create_dataframe_to_download(self, df_kmeans, raw_data, selected_accounts_columns, id_field):
        # Merge the kmeans data with the raw data on the specified id_field
        download_data = raw_data.merge(
            df_kmeans[[id_field, 'Ranking', 'Recency', 'Frequency', 'Monetary']], 
            on=id_field, 
            how='left'
        )

        # Ensure that the selected accounts columns are included in the final DataFrame
        columns_order = [id_field, 'Ranking', 'Recency', 'Frequency', 'Monetary'] + \
                        [col for col in selected_accounts_columns if col != id_field]

        # Reorder the DataFrame to place kmeans data and selected accounts columns at the beginning
        download_data = download_data[columns_order]
        
        # Remove any duplicate rows
        download_data = download_data.drop_duplicates()

        # Remove rows where all values are NaN
        download_data = download_data.dropna(how='all')

        return download_data

    # Function to add 'Deal : Account ID' column to Deals DataFrame
    def add_account_id_column(self, deals_df, accounts_df):
        # Create a mapping from 'Account : Name' to 'SalesAccount : id'
        account_id_mapping = dict(zip(accounts_df['Account : Name'], accounts_df['SalesAccount : id']))
        
        # Map 'Deal : Account name' to 'SalesAccount : id' and create a new column
        deals_df['Deal : Account ID'] = deals_df['Deal : Account name'].map(account_id_mapping)
        
        # Ensure the 'Deal : Account ID' column is of string type
        deals_df['Deal : Account ID'] = deals_df['Deal : Account ID'].astype(str)
        
        return deals_df
