import os
import pandas as pd
import functools as ft
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from ydata_profiling import ProfileReport
import sweetviz as sv
import streamlit.components.v1 as components
import stat
import streamlit as st

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
        avg_time_tracked_ticket_count = helpdesk_performance['Ticket_Count'].mean()


        # Calculate Agent_Needed based on the mean value
        helpdesk_performance['Agent_Needed'] = (avg_time_tracked * avg_time_tracked_ticket_count) / (helpdesk_performance['Working_Days'].mean() * 8)

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

    def load_and_process_companies(self,file):
        # Load the Excel file into a DataFrame
        df_companies = pd.read_excel(file)

        # Ensure the required columns exist
        required_columns = ['Company Name', 'Initial ASM Date', 'Renewal date']
        if not all(col in df_companies.columns for col in required_columns):
            raise ValueError(f"File must contain these columns: {', '.join(required_columns)}")

        # Convert 'Initial ASM Date' and 'Renewal Date' to datetime format
        df_companies['Initial ASM Date'] = pd.to_datetime(df_companies['Initial ASM Date'], errors='coerce')
        df_companies['Renewal date'] = pd.to_datetime(df_companies['Renewal date'], errors='coerce')

        # Create a DataFrame with months ranging from the earliest 'Initial ASM Date' to the latest 'Renewal Date'
        df_companies['Month'] = df_companies.apply(lambda row: pd.date_range(start=row['Initial ASM Date'], end=row['Renewal date'], freq='MS').strftime('%Y-%m'), axis=1)
        
        # Explode the 'Month' column to have a row for each month
        df_companies = df_companies.explode('Month').reset_index(drop=True)

        # Add the 'Valid Maintenance' column
        df_companies['Valid Maintenance'] = 'Yes'

        return df_companies[['Month', 'Company Name', 'Valid Maintenance']]

    # Function to convert date columns to datetime format
    def convert_date_columns_to_date(self, df):
        date_columns = [
            'Created time', 
            'Due by Time', 
            'Resolved time', 
            'Closed time', 
            'Last updated time', 
            'Initial response time', 
            'Initial ASM Date', 
            'Handover date'
        ]
        
        for col in date_columns:
            if col in df.columns:
                # Convert to datetime using the format YYYY-MM-DD, handling errors and missing values
                df[col] = pd.to_datetime(df[col], dayfirst=True, format='mixed', errors='coerce')
                
                # Remove timezone information if present
                if pd.api.types.is_datetime64_any_dtype(df[col]):
                    df[col] = df[col].dt.tz_localize(None)
        
        return df

    def convert_time_to_float(self, df, columns):
        def time_to_float(time_str):
            try:
                # Split the time into hours, minutes, and seconds (if available)
                parts = time_str.split(':')
                if len(parts) == 3:  # hh:mm:ss format
                    hh, mm, ss = parts
                elif len(parts) == 2:  # hh:mm format
                    hh, mm = parts
                    ss = '0'  # No seconds provided, default to 0
                else:
                    return None  # Invalid time format

                # Convert the time into decimal hours
                return float(hh) + float(mm) / 60 + float(ss) / 3600
            except (ValueError, AttributeError):
                return None  # Handle cases where time format is incorrect or missing

        for column in columns:
            if column in df.columns:
                # Apply the time_to_float conversion to each column
                df[column] = df[column].apply(time_to_float)

        return df
    

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
            print(f"Permissions set to 644 for file: {file_path}")
            # Check permissions after setting
            permissions = oct(os.stat(file_path).st_mode)[-3:]
            print(f"Current permissions: {permissions}")
        except FileNotFoundError:
            print(f"File not found: {file_path}")
        except PermissionError:
            print(f"Permission denied: {file_path}")
        except OSError as e:
            print(f"OS error occurred: {e}")

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

    def filter_data_by_ranking(self, download_data):
        unique_rankings = download_data['Ranking'].dropna().unique().tolist()
        
        # Ensure there are unique values to select
        if unique_rankings:
            selected_rankings = st.multiselect('Select Clusters to Filter:', unique_rankings)
            
            if selected_rankings:
                # Filter the data based on the selected rankings
                filtered_data = download_data[download_data['Ranking'].isin(selected_rankings)]
                
                # Count the number of records where 'Valid Maintenance' is 'Yes' and 'No'
                valid_maintenance_yes_count = filtered_data[filtered_data['Valid Maintenance'] == 'Yes'].shape[0]
                valid_maintenance_no_count = filtered_data[filtered_data['Valid Maintenance'] == 'No'].shape[0]
                
                # Display the counts
                st.markdown(f"**Total Valid Maintenance Count:**")
                st.markdown(f"- **Yes:** {valid_maintenance_yes_count}")
                st.markdown(f"- **No:** {valid_maintenance_no_count}")
                
                st.markdown(f'**Filtered Data by Rankings: {", ".join(selected_rankings)}**')
                st.dataframe(filtered_data)
                
                return filtered_data
            else:
                st.warning("Please select at least one ranking value to filter.")
                return download_data
        else:
            st.warning("No unique 'Ranking' values found to filter.")
            return download_data




