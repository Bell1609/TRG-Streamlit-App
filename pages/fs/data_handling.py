from io import BytesIO
import os
import stat
import pandas as pd
import numpy as np
from pandas_profiling import ProfileReport
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from datetime import datetime
import sweetviz as sv
import streamlit.components.v1 as components
from ydata_profiling import ProfileReport


class Data_Handling():
    def get_raw(self, file):
        try:
            raw_data = pd.read_csv(file)
        except Exception:
        # try:
            raw_data = pd.read_excel(file)
        except:
            print("Use .csv or .xlsx files only!")
            return None
        # raw_data['AccountName'] = raw_data['AccountName'].str.strip()
        return raw_data
    
    def create_rfm_dataframe(self, df, id_field):
        # Initialize the RFM DataFrame using the unique account IDs
        df_rfm = pd.DataFrame(df[id_field].unique())
        df_rfm.columns = [id_field]

        # Get today's date
        today = pd.to_datetime(datetime.today().date())

        # Convert 'Deal : Expected close date' to datetime
        df['Deal : Expected close date'] = pd.to_datetime(df['Deal : Expected close date'], dayfirst=True, errors='coerce')

        # Adjust 'Expected close date' greater than today
        df['Adjusted Close Date'] = df['Deal : Expected close date'].apply(lambda x: today if pd.notna(x) and x > today else x)


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
        #df['Deal : Total Deal Value'] = df['Deal : Total Deal Value'].astype(str).replace('[\$,]', '', regex=True).astype(float)
        #df['Deal : Total Deal Value'] = pd.to_numeric(df['Deal : Total Deal Value'].str.replace('[\$,]', '', regex=True), errors='coerce')

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
    
    # Validation for mandatory fields
    def validate_columns(self, df, mandatory_fields, file_type):
        missing_fields = [field for field in mandatory_fields if field not in df.columns]
        if missing_fields:
            st.error(f'The {file_type} data is missing the following mandatory columns: {", ".join(missing_fields)}')
            return False
        return True

    """  # Define function to extract revenue, cost, and other values for the selected product
    def get_product_values(self, df, product, product_values):
        # Loop through all product columns (Deal : Product 1 to Deal : Product 4)
        for i in range(1, 5):
            product_column = f'Deal : Product {i}'
            
            # Check if the product column exists
            if product_column not in df.columns:
                continue
            
            # Find rows where the selected product is found in the specific 'Deal : Product n' column
            product_rows = df[df[product_column] == product]
            
            if not product_rows.empty:
                # List of column names to check
                columns_to_check = {
                    'Deal Software revenue': f'Deal : Software revenue: Product {i}',
                    'Deal Software cost': f'Deal : Software cost: Product {i}',
                    'Deal ASM revenue': f'Deal : ASM revenue: Product {i}',
                    'Deal ASM cost': f'Deal : ASM cost: Product {i}',
                    'Deal Service revenue': f'Deal : Service revenue: Product {i}',
                    'Deal Service cost': f'Deal : Service cost: Product {i}',
                    'Deal Cons days': f'Deal : Cons days: Product {i}',
                    'Deal PM days': f'Deal : PM days: Product {i}',
                    'Deal PA days': f'Deal : PA days: Product {i}',
                    'Deal Technical days': f'Deal : Technical days: Product {i}',
                    'Deal Hosting revenue': f'Deal : Hosting revenue: Product {i}',
                    'Deal Hosting cost': f'Deal : Hosting cost: Product {i}',
                    'Deal Managed service revenue': f'Deal : Managed service revenue: Product {i}',
                    'Deal Managed service cost': f'Deal : Managed service cost: Product {i}',
                }
                
                # Sum values from columns if they exist
                for key, col in columns_to_check.items():
                    if col in df.columns:
                        product_values[key] += product_rows[col].sum()

        return product_values """
    
    # Step 2: Define the function to accumulate values for each product
    def get_product_values(self, df, selected_products):
        # Iterate through each product in the selected_products list
        for product in selected_products:
            # Loop through all product columns (Deal : Product 1 to Deal : Product 4)
            for i in range(1, 5):
                product_column = f'Deal : Product {i}'

                # Check if the product column exists in the dataframe
                if product_column not in df.columns:
                    continue

                # Iterate through each row in the dataframe
                for idx, row in df.iterrows():
                    # Check if the product matches in the current product column
                    if row[product_column] == product:
                        # List of columns to accumulate values from
                        columns_to_check = {
                            'Deal Software Revenue': f'Deal : Software revenue: Product {i}',
                            'Deal Software Cost': f'Deal : Software cost: Product {i}',
                            'Deal ASM Revenue': f'Deal : ASM revenue: Product {i}',
                            'Deal ASM Cost': f'Deal : ASM cost: Product {i}',
                            'Deal Service Revenue': f'Deal : Service revenue: Product {i}',
                            'Deal Service Cost': f'Deal : Service cost: Product {i}',
                            'Deal Cons Days': f'Deal : Cons days: Product {i}',
                            'Deal PM Days': f'Deal : PM days: Product {i}',
                            'Deal PA Days': f'Deal : PA days: Product {i}',
                            'Deal Technical Days': f'Deal : Technical days: Product {i}',
                            'Deal Hosting Revenue': f'Deal : Hosting revenue: Product {i}',
                            'Deal Hosting Cost': f'Deal : Hosting cost: Product {i}',
                            'Deal Managed Service Revenue': f'Deal : Managed service revenue: Product {i}',
                            'Deal Managed Service Cost': f'Deal : Managed service cost: Product {i}',
                        }

                        # Accumulate the values from the respective columns if they exist
                        for key, col in columns_to_check.items():
                            if col in df.columns:
                                df.at[idx, key] += row[col]  # Accumulate value in the corresponding new column

        return df


    def convert_mixed_columns_to_string(self, df):
        for col in df.columns:
            try:
                if df[col].apply(lambda x: isinstance(x, str)).any() and pd.api.types.infer_dtype(df[col]) == 'mixed':
                    df[col] = df[col].astype(str)
                    st.warning(f"Column '{col}' was converted to string.")
            except Exception as e:
                st.error(f"Error converting column '{col}' to string: {e}")
        return df


    def clean_and_convert_amount_columns(self, df):
        """
        This function cleans and converts the amount columns in the dataframe, creates a 'Deal : Product' column 
        by combining 'Deal : Product n' columns (1 to 4), and then drops the unnecessary columns.

        Parameters:
        df (pd.DataFrame): The DataFrame containing deal data to process.

        Returns:
        pd.DataFrame: The processed DataFrame with cleaned amount columns and combined 'Deal : Product' column.
        """
        # Define the columns to process
        columns_to_process = [
            'Deal : Total Deal Value', 'Deal : Deal value in Base Currency',
            'Deal : Expected deal value', 'Deal : Total Cost', 'Deal : Gross Margin (GM)',
            'Deal : Software revenue: Product 1', 'Deal : Software revenue: Product 2', 'Deal : Software revenue: Product 3', 'Deal : Software revenue: Product 4',
            'Deal : Software cost: Product 1', 'Deal : Software cost: Product 2', 'Deal : Software cost: Product 3', 'Deal : Software cost: Product 4',
            'Deal : ASM revenue: Product 1', 'Deal : ASM revenue: Product 2', 'Deal : ASM revenue: Product 3', 'Deal : ASM revenue: Product 4',
            'Deal : ASM cost: Product 1', 'Deal : ASM cost: Product 2', 'Deal : ASM cost: Product 3', 'Deal : ASM cost: Product 4',
            'Deal : Service revenue: Product 1', 'Deal : Service revenue: Product 2', 'Deal : Service revenue: Product 3', 'Deal : Service revenue: Product 4',
            'Deal : Service cost: Product 1', 'Deal : Service cost: Product 2', 'Deal : Service cost: Product 3', 'Deal : Service cost: Product 4',
            'Deal : Cons days: Product 1', 'Deal : Cons days: Product 2', 'Deal : Cons days: Product 3', 'Deal : Cons days: Product 4',
            'Deal : Technical days: Product 1', 'Deal : Technical days: Product 2', 'Deal : Technical days: Product 3', 'Deal : Technical days: Product 4',
            'Deal : PM days: Product 1', 'Deal : PM days: Product 2', 'Deal : PM days: Product 3', 'Deal : PM days: Product 4',
            'Deal : PA days: Product 1', 'Deal : PA days: Product 2', 'Deal : PA days: Product 3', 'Deal : PA days: Product 4',
            'Deal : Hosting revenue: Product 1', 'Deal : Hosting revenue: Product 2', 'Deal : Hosting revenue: Product 3', 'Deal : Hosting revenue: Product 4',
            'Deal : Hosting cost: Product 1', 'Deal : Hosting cost: Product 2', 'Deal : Hosting cost: Product 3', 'Deal : Hosting cost: Product 4',
            'Deal : Managed service revenue: Product 1', 'Deal : Managed service revenue: Product 2', 'Deal : Managed service revenue: Product 3', 'Deal : Managed service revenue: Product 4',
            'Deal : Managed service cost: Product 1', 'Deal : Managed service cost: Product 2', 'Deal : Managed service cost: Product 3', 'Deal : Managed service cost: Product 4'
        ]

        # Convert columns to numeric (if applicable)
        for col in columns_to_process:
            if col in df.columns:
                df[col] = df[col].replace('[\$,]', '', regex=True).astype(float)


        return df


        
    # Function to convert date columns to datetime format
    def convert_date_columns_to_date(self, df):
        date_columns = [
            'Deal : Closed date', 
            'Deal : Expected close date', 
            'Deal : Created at', 
            'Deal : Updated at', 
            'Deal : Last assigned at', 
            'Deal : First assigned at', 
            'Deal : Deal stage updated at', 
            'Deal : Last activity date', 
            'Deal : Expected go live date/MED', 
            'Deal : Tentative start date/MSD', 
            'Deal : Commitment Expiration Date'
        ]
        
        for col in date_columns:
            if col in df.columns:
                # Convert to datetime using the format YYYY-MM-DD
                df[col] = pd.to_datetime(df[col], dayfirst=True, format='mixed', errors='coerce')
        
        return df

    def filter_by_products(self, df, selected_products):
        # Initialize a DataFrame to store filtered rows
        filtered_df = pd.DataFrame()

        # Loop through each product in selected_products
        for product in selected_products:
            # Initialize a mask that is False by default
            product_mask = pd.Series([False] * len(df), index=df.index)

            # Check in 'Deal : Product 1' to 'Deal : Product 4'
            for i in range(1, 5):
                product_column = f'Deal : Product {i}'

                # Ensure the column exists in the DataFrame
                if product_column in df.columns:
                    # Update mask to True for rows where the product matches
                    product_mask |= df[product_column] == product

            # Append rows where the product matches to the filtered DataFrame
            filtered_df = pd.concat([filtered_df, df[product_mask]])

        # Drop duplicates in case the same row matches multiple products
        return filtered_df.drop_duplicates()


    def data_profiling(self, df, df_name):
        st.markdown(f'**{df_name} Data Profiling**')
        st.write(f"Basic Statistics for {df_name} data:")
        
        # Select only numeric columns for statistics
        numeric_df = df.select_dtypes(include=['number'])

        # Get the descriptive statistics using describe()
        desc = numeric_df.describe()

        # Calculate the sum for each numeric column and append it as a new row
        sum_row = pd.DataFrame(numeric_df.sum(), columns=['sum']).T

        # Concatenate the sum row with the describe() output
        desc_with_sum = pd.concat([desc, sum_row])

        # Display the statistics in Streamlit
        st.write(desc_with_sum)
        

    def display_column_sums_streamlit(self, df):
        """
        Display the sum of specified columns in the dataframe using Streamlit.

        Parameters:
        df (pd.DataFrame): The input dataframe containing the columns.
        columns (list): A list of columns for which to calculate the sum.

        Returns:
        pd.DataFrame: A DataFrame showing the column name and corresponding sum.
        """
        
        columns = [
            'Deal : Total Deal Value',
            'Deal : Total Cost',
            'Deal : Gross Margin (GM)',
            'Deal Software Revenue',
            'Deal Software Cost',
            'Deal ASM Revenue',
            'Deal ASM Cost',
            'Deal Service Revenue',
            'Deal Service Cost',
            'Deal Cons Days',
            'Deal PM Days',
            'Deal PA Days',
            'Deal Technical Days',
            'Deal Hosting Revenue',
            'Deal Hosting Cost',
            'Deal Managed Service Revenue',
            'Deal Managed Service Cost'
        ]
        # Initialize a list to hold the column names and their corresponding sums
        column_sums = []

        # Loop through each column in the provided list
        for column in columns:
            if column in df.columns:
                # Calculate the sum for the column
                column_sum = df[column].sum(skipna=True)
                # Append the column and its sum to the list
                column_sums.append({'Column': column, 'Sum': column_sum})

            else:
                # Append the column with a sum of 0 if it's not found in the dataframe
                column_sums.append({'Column': column, 'Sum': 0})
        
        # Convert the list of dictionaries into a DataFrame for display
        sums_df = pd.DataFrame(column_sums)

        # Display the DataFrame with formatted sums using Streamlit
        st.write("### Column Sums")
        st.dataframe(sums_df.style.format({'Sum': "{:,.2f}"}))

        return sums_df

    # Function to generate ydata_profiling report and save it
    def generate_ydata_profiling_report(self, df, title):
        report = ProfileReport(df, title=title)
        report_file = f"{title} Report.html"  # Specify the file name
        report.to_file(report_file)            # Save the report as an HTML file
        return report_file                     # Return the file path

    # Display existing profiling report function
    def display_ydata_profiling_report(self, report_file_path):
        try:
            with open(report_file_path, 'r', encoding='utf-8') as f:
                report_html = f.read()
            components.html(report_html, height=700, scrolling=True)

        except PermissionError:
            st.error(f"Permission denied when trying to access {report_file_path}. Please check file permissions.")
        except FileNotFoundError:
            st.error(f"The file {report_file_path} does not exist. Please generate the report first.")
        except OSError as e:
            st.error(f"OS error occurred: {e}")
        except UnicodeDecodeError:
            st.error("Error decoding the profiling report. The file might contain incompatible characters.")
            
    def set_file_permissions(self, file_path):
        try:
            os.chmod(file_path, stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH)
            st.write(f"Permissions set to 644 for file: {file_path}")
            # Check permissions after setting
            permissions = oct(os.stat(file_path).st_mode)[-3:]
            st.write(f"Current permissions: {permissions}")
        except FileNotFoundError:
            st.write(f"File not found: {file_path}")
        except PermissionError:
            st.write(f"Permission denied: {file_path}")
        except OSError as e:
            st.write(f"OS error occurred: {e}")



    # Function to generate and display Sweetviz report
    def generate_sweetviz_report(self, df, df_name):
        report = sv.analyze(df)
        report_name = f"{df_name}_report.html"
        report.show_html(filepath=report_name, open_browser=False)
        return report_name

    def display_sweetviz_report(self, report_name):
        try:
            with open(report_name, 'r', encoding='utf-8') as f:
                report_html = f.read()
            components.html(report_html, height=700, scrolling=True)
        except UnicodeDecodeError:
            st.error("Error decoding the Sweetviz report. The file might contain characters that are not compatible with the default encoding.")


    def create_excel(self, df):
        output = BytesIO()
        writer = pd.ExcelWriter(output, engine='xlsxwriter')
        df.to_excel(writer, index=False)
        
        writer.close()
        processed_data = output.getvalue()

        return processed_data

    def filter_data_by_ranking(self, download_data):
        unique_rankings = download_data['Ranking'].unique().tolist()
        
        # Ensure there are unique values to select
        if unique_rankings:
            selected_rankings = st.multiselect('Select Clusters to Filter:', unique_rankings)
            
            if selected_rankings:
                # Filter the data based on the selected rankings
                filtered_data = download_data[download_data['Ranking'].isin(selected_rankings)]
                
                # Count the number of records where 'TRG Customer' is 'Yes' and 'No'
                trg_customer_yes_count = filtered_data[filtered_data['Account : TRG Customer'] == 'Yes'].shape[0]
                trg_customer_no_count = filtered_data[filtered_data['Account : TRG Customer'] == 'No'].shape[0]
                
                # Display the counts
                st.markdown(f"**Total 'TRG Customer' Count:**")
                st.markdown(f"- **Yes:** {trg_customer_yes_count}")
                st.markdown(f"- **No:** {trg_customer_no_count}")
                
                st.markdown(f'**Filtered Data by Rankings: {", ".join(selected_rankings)}**')
                st.dataframe(filtered_data)
                
                return filtered_data
            else:
                st.warning("Please select at least one ranking value to filter.")
                return download_data
        else:
            st.warning("No unique 'Ranking' values found to filter.")
            return download_data
        
