import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


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
    
    def create_dataframe(self, raw_data):
        raw_data['AccountID'] = raw_data['AccountID'].apply(lambda x: "{:.0f}".format(x)).astype(str)
        sub_list = [ 'AccountID', 'CloseDate', 'DealValue']
        df_sort = raw_data[(raw_data['DealStage']=='Won')]
        df = df_sort[sub_list].reset_index(drop=True)
        
        # df_sort = raw_data[(raw_data['Deal : Deal stage']=='Won')].dropna(subset=['Deal : Account name', 'Account ID', 'Deal : Closed date', 'Deal : Deal value'])
        # df = df_sort[['Deal : Account name', 'Account ID', 'Deal : Closed date', 'Deal : Deal value']].reset_index(drop=True)

        # df = df.rename(columns={
        #     'Deal : Account name': 'AccountName',
        #     'Account ID': 'AccountID',
        #     'Deal : Closed date': 'CloseDate', 
        #     'Deal : Deal value in Base Currency': 'DealValue'
        # })
        # df['AccountID'] = df['AccountID'].apply(lambda x: "{:.0f}".format(x)).astype(str)
        return df
    
    def create_rfm_dataframe(self, df):
        df_rfm = pd.DataFrame(df['AccountID'].unique())
        df_rfm.columns = ['AccountID']
        # df_rfm = df_rfm.merge(df_compare, how='right', on='AccountID')
        
        # Getting recency values
        df['CloseDate'] = pd.to_datetime(df['CloseDate'], dayfirst=True, format='mixed')
        last_purchase = df.groupby('AccountID').CloseDate.max().reset_index()
        last_purchase.columns = ['AccountID', 'CloseDateMax']

        last_purchase['Recency'] = (last_purchase['CloseDateMax'].max() - last_purchase['CloseDateMax']).dt.days

        df_rfm = pd.merge(df_rfm, last_purchase[['AccountID','Recency']],how='left', on='AccountID')
        
        
        # Getting frequency values
        df_freq = df.dropna(subset=['AccountID']).groupby('AccountID').CloseDate.count().reset_index()
        df_freq.columns = ['AccountID','Frequency']

        df_rfm = pd.merge(df_rfm,df_freq,on='AccountID')
        df_freq['AccountID'].replace('nan', np.nan, inplace=True)

        # Drop rows where 'AccountID' is NaN
        df_freq = df_freq.dropna(subset=['AccountID'], how='any')
        
        
        # Getting monetary values
        df['DealValue'] = df['DealValue'].astype(str).replace('[\$,]','',regex=True).astype(float)
        df_mone = df.groupby('AccountID').DealValue.sum().reset_index()
        df_mone.columns = ['AccountID','Monetary']

        df_rfm = pd.merge(df_rfm,df_mone,on='AccountID')
        
        return df_rfm
    
    
    
    # we do kmeans next
    def create_kmeans_dataframe(self, df_rfm):
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
            
            
        df_rfm_copy = df_rfm.copy()
        rfm_selected = df_rfm[['Recency','Frequency','Monetary']]
        # Times -1 to invert the recency
        rfm_selected['Recency'] = np.abs(rfm_selected['Recency']) * -1
        # Scale the features
        scaler = StandardScaler()
        rfm_standard = scaler.fit_transform(rfm_selected)
        
        comp = 0
        cluster_labels_ans = None
        kmeans_ans = None
        for c in range(5,7):
            for n in range(1, 400):
            # for k in range(2, 10):
                kmeans = KMeans(n_clusters=c, random_state=n)
                cluster_labels = kmeans.fit_predict(rfm_standard)
                silhouette_avg = silhouette_score(rfm_standard, cluster_labels)
                if comp < silhouette_avg:
                    comp = silhouette_avg
                    cluster = c
                    state = n
                    cluster_labels_ans = cluster_labels
                    kmeans_ans = kmeans
                    
        print(f'random state: {state}, cluster {cluster} with {comp}')

        clustered_data = pd.DataFrame({'AccountID': df_rfm_copy.AccountID, 'Cluster': cluster_labels_ans})
        
        df_kmeans = pd.merge(df_rfm, clustered_data, on='AccountID')

        for i in range(0, cluster):
            df_kmeans.loc[df_kmeans['Cluster'] == i, 'Ranking'] = f'Cluster {i}'
            
        cluster_centers = create_clustered_data(kmeans_ans)
        return df_kmeans, cluster_centers, comp
    
    def create_dataframe_to_download(self, df_kmeans, raw_data):
        download_data = raw_data.merge(df_kmeans[['Ranking', 'AccountID']], on='AccountID', how='left')
        download_data = download_data.drop_duplicates()
        return download_data