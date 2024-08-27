import pandas as pd
import numpy as np
import functools as ft
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

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
    
    def create_dataframe(self, raw_data):
        raw_data.loc[~raw_data['Group Company'].isnull(), 'Client code'] = raw_data['Group Company']
        raw_data.loc[~raw_data['Brand'].isnull(), 'Client code'] = raw_data['Brand']
        fd_customer = raw_data[raw_data['TRG Customer']==True]
        
        return fd_customer
    
    def create_df_with_relevant_info(self, fd_customer):
        def create_recency():
            fd_customer['Closed time'] = pd.to_datetime(fd_customer['Closed time'])
            df_recency = fd_customer.groupby('Client code')['Closed time'].max().reset_index()
            df_recency['Recency'] = (df_recency['Closed time'].max() - df_recency['Closed time']).dt.days
            return df_recency
        
        def create_volume():
            df_volume = fd_customer.groupby('Client code')['Closed time'].count().reset_index()
            df_volume = df_volume.rename(columns=({'Closed time': 'Ticket Count'}))
            df_volume = df_volume.loc[df_volume['Client code']!='']
            return df_volume
        
        def create_interactions():
            df_interactions = fd_customer.groupby('Client code')[['Customer interactions', 'Agent interactions']].sum().reset_index()
            return df_interactions
        
        def create_ams_cms():
            df_ams_cms = fd_customer.groupby('Client code')[['AMS', 'CMS']].all().reset_index()
            return df_ams_cms
            
        df_recency = create_recency()
        df_volume = create_volume()
        df_interactions = create_interactions()
        df_ams_cms = create_ams_cms()

        df_list = [df_recency[['Client code', 'Recency']], df_volume, df_ams_cms, df_interactions]
        df_attributes = ft.reduce(lambda left, right: pd.merge(left, right, on='Client code'), df_list)
        return df_attributes
    
    def create_kmeans_dataframe(self, df_attributes, fd_data):
        def create_clustered_data(kmeans):
            # Create a DataFrame with cluster centers
            cluster_centers = pd.DataFrame(kmeans.cluster_centers_, columns=['Recency', 'Ticket Count', 'AMS', 'CMS', 'Customer interactions','Agent interactions'])

            # Add cluster size
            cluster_centers['Cluster Size'] = df_clusters['Cluster'].value_counts().sort_index().values

            for i in range(len(cluster_centers)):
                cluster_centers.loc[i, 'Cluster'] = f'Cluster {i}'
            return cluster_centers
        
        
        scaler = StandardScaler()
        df_compute = df_attributes.drop(columns='Client code')
        df_standard = scaler.fit_transform(df_compute)
        # We compare if being standardized makes better predictions or not
        comp = 0
        for df in [df_compute, df_standard]:
            for n in range(1, 300):
                kmeans = KMeans(n_clusters=5, random_state=n)
                cluster_labels = kmeans.fit_predict(df)
                silhouette_avg = silhouette_score(df, cluster_labels)
                if comp < silhouette_avg:
                    comp = silhouette_avg
                    state = n
                    df_suitable = df
                    cluster_labels_ans = cluster_labels
                    kmeans_ans = kmeans

        clustered_data = pd.DataFrame({'Client code': df_attributes['Client code'], 'Cluster': cluster_labels_ans})
        df_clusters = df_attributes.merge(clustered_data, on='Client code', how='left')
        df_clusters_name = df_clusters.merge(fd_data[['Client code', 'Company Name', 'Group Company', 'Brand']], on='Client code', how='left')
        df_clusters_name = df_clusters_name.drop_duplicates()
        
        for i in range(0, 5):
            df_clusters_name.loc[df_clusters_name['Cluster'] == i, 'Ranking'] = f'Cluster {i}'
        
        cluster_centers = create_clustered_data(kmeans_ans)
        return df_clusters_name, cluster_centers, comp
    
    def create_dataframe_to_download(self, df_kmeans, raw_data):
        download_data = raw_data.merge(df_kmeans[['Ranking', 'Client code']], on='Client code', how='left')
        download_data = download_data.drop_duplicates()
        return download_data
