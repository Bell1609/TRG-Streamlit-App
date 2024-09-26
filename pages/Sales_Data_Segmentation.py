from __future__ import division
from io import BytesIO
import time
from matplotlib import pyplot as plt
import streamlit as st
import pandas as pd
import sys
import os
import sweetviz as sv
import streamlit.components.v1 as components
from ydata_profiling import ProfileReport
import stat


# Add the parent directory to the system path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fs.data_handling import Data_Handling
from fs.graph_drawing import Graph_Drawing

data_handling = Data_Handling()
graph_drawing = Graph_Drawing()

def convert_mixed_columns_to_string(df):
        for col in df.columns:
            if df[col].dtype == 'object' and pd.api.types.infer_dtype(df[col]) == 'mixed':
                try:
                    df[col] = df[col].astype(str)
                    st.warning(f"Column '{col}' contained mixed types. It has been converted to string.")
                except Exception as e:
                    st.error(f"Failed to convert column '{col}' to string: {e}")
        return df


def clean_and_convert_amount_columns(df):
    # Define the columns to process
    columns_to_process = [
        'Deal : Total Deal Value',
        'Deal : Deal value in Base Currency',
        'Deal : Expected deal value'
        ]
        
    for col in columns_to_process:
        if col in df.columns:
            # Remove '$' character and convert to numeric
            df[col] = df[col].replace('[\$,]', '', regex=True).astype(float)
        
    return df
    
# Function to convert date columns to datetime format
def convert_date_columns_to_date(df):
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

# Main preprocessing function
def preprocess_data(df):
    # Clean and convert amount columns
    df = clean_and_convert_amount_columns(df)
    
    # Define mixed columns to convert to strings (replace with actual columns if needed)
    df = convert_mixed_columns_to_string(df)
    
    # Convert date columns to datetime format
    df = convert_date_columns_to_date(df)
    
    return df

# Function to generate ydata_profiling report and save it
def generate_ydata_profiling_report(df, title):
    report = ProfileReport(df, title=title)
    report_file = f"{title} Report.html"  # Specify the file name
    report.to_file(report_file)            # Save the report as an HTML file
    return report_file                     # Return the file path

# Display existing profiling report function
def display_ydata_profiling_report(report_file_path):
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
        
def set_file_permissions(file_path):
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

def data_profiling(df, df_name):
    st.markdown(f'**{df_name} Data Profiling**')
    st.write(f"Data Types for {df_name} data:")
    st.write(df.dtypes)
    st.write(f"Missing Values in {df_name} data:")
    st.write(df.isnull().sum())
    st.write(f"Basic Statistics for {df_name} data:")
    st.write(df.describe())

# Function to generate and display Sweetviz report
def generate_sweetviz_report(df, df_name):
    report = sv.analyze(df)
    report_name = f"{df_name}_report.html"
    report.show_html(filepath=report_name, open_browser=False)
    return report_name

def display_sweetviz_report(report_name):
    try:
        with open(report_name, 'r', encoding='utf-8') as f:
            report_html = f.read()
        components.html(report_html, height=700, scrolling=True)
    except UnicodeDecodeError:
        st.error("Error decoding the Sweetviz report. The file might contain characters that are not compatible with the default encoding.")
        

st.sidebar.success('Select the ticket data or sales data')

st.header('Sales Data Segmenting')

st.subheader('Data Load')

# File uploaders for Deals data and Accounts data
deals_file = st.file_uploader('Upload your Deals data file:', ['csv', 'xlsx'])
accounts_file = st.file_uploader('Upload your Accounts data file:', ['csv', 'xlsx'])

if 'stage' not in st.session_state:
    st.session_state.stage = 0

def click_button(stage):
    st.session_state.stage = stage

def create_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=False)
    
    writer.close()
    processed_data = output.getvalue()

    return processed_data

def filter_data_by_ranking(download_data):
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



# Mandatory fields for deals data
mandatory_deals_fields = ['Deal : Account ID', 'Deal : Account name', 'Deal : Expected close date', 'Deal : Total Deal Value', 'Deal : Probability (%)']
mandatory_accounts_fields = ['SalesAccount : id','Account : Name', 'Account : TRG Customer']

# Validation for mandatory fields
def validate_columns(df, mandatory_fields, file_type):
    missing_fields = [field for field in mandatory_fields if field not in df.columns]
    if missing_fields:
        st.error(f'The {file_type} data is missing the following mandatory columns: {", ".join(missing_fields)}')
        return False
    return True

if deals_file and accounts_file:
    deals_data = data_handling.get_raw(deals_file)
    accounts_data = data_handling.get_raw(accounts_file)
    
    if not deals_data.empty and not accounts_data.empty:
        # Convert columns with mixed types to strings
        deals_data = preprocess_data(deals_data)
        accounts_data = preprocess_data(accounts_data)
        deal_output = create_excel(deals_data)
        accounts_output = create_excel(accounts_data)

        # Ensure the ID fields are treated as strings before merging
        accounts_data['SalesAccount : id'] = accounts_data['SalesAccount : id'].astype(str)

        # Add 'Deal : Account ID' column to Deals DataFrame
        deals_data = data_handling.add_account_id_column(deals_data, accounts_data)

        st.markdown('**Deals Data Frame**')
        st.dataframe(deals_data)
        st.markdown('**Accounts Data Frame**')
        st.dataframe(accounts_data)

        # Validate mandatory fields in Deals and Accounts data
        if not validate_columns(deals_data, mandatory_deals_fields, 'Deals'):
            st.stop()
            
        if not validate_columns(accounts_data, mandatory_accounts_fields, 'Accounts'):
            st.stop()

        st.subheader('Data Exploration')

        #Data profiling before segmentation
        data_profiling(deals_data, 'Deals')
        data_profiling(accounts_data, 'Accounts')
        # # Add buttons to generate Sweetviz reports
        # if st.button('Generate Profiling Reports'):
        #     # deals_report = generate_sweetviz_report(deals_data, 'Deals')
        #     # display_sweetviz_report(deals_report)
        #     # Generate the report and get the HTML file path
            
        #     deals_report_file_path = generate_ydata_profiling_report(deals_data, 'Deals Data Report')
        #     # Display the generated ydata_profiling report in Streamlit
        #     display_ydata_profiling_report(deals_report_file_path)
            

        #     st.markdown('**Accounts Data Profile Report**')
        #     # accounts_report = generate_sweetviz_report(accounts_data, 'Accounts')
        #     # display_sweetviz_report(accounts_report)
        #     accounts_report_file_path = generate_profiling_report(accounts_data, 'Accounts Data Report')
        #     # Display the generated ydata_profiling report in Streamlit
        #     display_ydata_profiling_report(accounts_report_file_path)
        
        # if st.button('Display Profiling Reports'):
        #     st.markdown('**Deals Data Profile Report**')
        #     # Display the generated ydata_profiling report in Streamlit
        #     display_ydata_profiling_report(deals_report_file_path)
            

        #     st.markdown('**Accounts Data Profile Report**')
        #     # Display the generated ydata_profiling report in Streamlit
        #     display_ydata_profiling_report(accounts_report_file_path)
        
        # Set default report file paths (in the same folder as the application)
        deals_report_file_path = 'Deals Data Report.html'
        accounts_report_file_path = 'Accounts Data Report.html'
        
        # Generate Profiling Report Button
        if st.button('Generate Profiling Reports'):
            # Generate the reports
            st.markdown('**Generating Deals Data Profile Report...**')
            deals_report_file_path = generate_ydata_profiling_report(deals_data, 'Deals Data')
                
            st.markdown('**Generating Accounts Data Profile Report...**')
            accounts_report_file_path = generate_ydata_profiling_report(accounts_data, 'Accounts Data')
                
            st.success('Reports generated successfully!')

        if st.button('Display Profiling Reports'):
            # Validate if the report files exist before displaying them
            st.markdown('**Deals Data Profile Report**')
            if os.path.exists(deals_report_file_path):
                set_file_permissions(deals_report_file_path)
                display_ydata_profiling_report(deals_report_file_path)
            else:
                st.error('Deals Data Report does not exist. Please generate the report first.')

            st.markdown('**Accounts Data Profile Report**')
            if os.path.exists(accounts_report_file_path):
                set_file_permissions(deals_report_file_path)
                display_ydata_profiling_report(accounts_report_file_path)
            else:
                st.error('Accounts Data Report does not exist. Please generate the report first.')


        st.subheader('Data Preprocessing')
        # List columns for both files
        deals_columns = deals_data.columns.tolist()
        accounts_columns = accounts_data.columns.tolist()

        # Choose columns for merging
        st.markdown('**Select Columns for Merging DataFrames**')

        # Ensure mandatory fields are selected
        selected_deals_columns = st.sidebar.multiselect('Select Deals Columns:', deals_columns, default=mandatory_deals_fields)
        selected_accounts_columns = st.sidebar.multiselect('Select Accounts Columns:', accounts_columns, default=mandatory_accounts_fields)

        if not all(field in selected_deals_columns for field in mandatory_deals_fields):
            st.error(f'You must select these mandatory fields from the Deals data: {mandatory_deals_fields}')
            st.stop()

        if not all(field in selected_accounts_columns for field in mandatory_accounts_fields):
            st.error(f'You must select these mandatory fields from the Accounts data: {mandatory_accounts_fields}')
            st.stop()

        # Select ID fields for merging
        st.markdown('**Select ID Fields for Merging**')
        # Set default values for ID fields
        default_deals_id_field = 'Deal : Account ID'
        default_accounts_id_field = 'SalesAccount : id'

        # Ensure that the default values are part of the selectable options
        if default_deals_id_field not in selected_deals_columns:
            st.warning(f"Default Deals ID field '{default_deals_id_field}' is not in the selected deals columns.")
            default_deals_id_field = None  # Remove the default value if it doesn't exist

        if default_accounts_id_field not in selected_accounts_columns:
            st.warning(f"Default Accounts ID field '{default_accounts_id_field}' is not in the selected accounts columns.")
            default_accounts_id_field = None  # Remove the default value if it doesn't exist

        # Create selectboxes with the default values
        deals_id_field = st.sidebar.selectbox('Select Deals ID Field:', selected_deals_columns, index=selected_deals_columns.index(default_deals_id_field) if default_deals_id_field else 0)
        accounts_id_field = st.sidebar.selectbox('Select Accounts ID Field:', selected_accounts_columns, index=selected_accounts_columns.index(default_accounts_id_field) if default_accounts_id_field else 0)

        # Filter deals data by 'Deal : Probability (%)'
        prob_min, prob_max = st.sidebar.slider('Select Probability (%) Range:', 0, 100, (0, 100))
        deals_data['Deal : Probability (%)'] = deals_data['Deal : Probability (%)'].astype(int)
        deals_data = deals_data[(deals_data['Deal : Probability (%)'] >= prob_min) & (deals_data['Deal : Probability (%)'] <= prob_max)]

        # Checkbox for filtering by 'TRG Customer'
        filter_trg_customer = st.sidebar.checkbox('Filter by TRG Customer')
        
        # Add a sidebar selectbox for Deal: Project type
        if 'Deal : Project type' in deals_data.columns:
            selected_project_type = st.sidebar.multiselect('Select Project Type:', deals_data['Deal : Project type'].unique())

        deals_data = deals_data[(deals_data['Deal : Project type'].isin(selected_project_type))]
      
        
        try:
            # Merge dataframes based on selected ID fields
            merged_data = deals_data[selected_deals_columns].merge(accounts_data[selected_accounts_columns], left_on=deals_id_field, right_on=accounts_id_field, how='left')
            
            # Check if the filter_trg_customer flag is set
            if filter_trg_customer:
                if 'Account : TRG Customer' in merged_data.columns:
                    merged_data = merged_data[merged_data['Account : TRG Customer'] == 'Yes']
                else:
                    st.warning('Column "Account : TRG Customer" not found in merged data.')
            st.success('DataFrames merged successfully.')
            
            st.dataframe(merged_data)
        except KeyError as ke:
            st.error(f'Error merging data: {ke}')
            st.stop()

        # Run RFM Segmentation
        if st.button('Run RFM Segmentation'):
            click_button(1)
        
        if st.session_state.stage >= 1:
            # Creates RFM dataframe for the segmentation
            rfm_data = data_handling.create_rfm_dataframe(merged_data, accounts_id_field)  # Use the new ID column for RFM segmentation
            # st.markdown('**RFM Data Frame**')
            # st.dataframe(rfm_data)
            
            # Measure the start time
            start_time = time.time()

            # Creates dataframe with clusters from kmeans
            kmeans_data, cluster_centers, silhouette_score, best_k, best_random_state = data_handling.create_kmeans_dataframe(rfm_data, accounts_id_field)
            # st.markdown("Cluster Center Dataframe")
            # st.dataframe(cluster_centers)
            # st.markdown("Kmeans Dataframe")
            # st.dataframe(kmeans_data)
            # Measure the end time
            end_time = time.time()

            # Calculate the elapsed time
            elapsed_time = end_time - start_time

            # Display the silhouette score
            st.markdown('**Result of Segmentation**')
            st.dataframe(cluster_centers)
            st.write('Silhouette Score: {:0.2f}'.format(silhouette_score))
            
            # Display the number of clusters and random state used
            st.write('Number of Clusters:', best_k)
            st.write('Random State:', best_random_state)

            # Display the time taken to execute the clustering
            st.write('Time taken to complete the clustering: {:.2f} seconds'.format(elapsed_time))

            
            # Creates graphs 
            st.markdown('**RFM Data Visualization**')
            for component, color in zip(['Recency', 'Frequency', 'Monetary'], ['blue', 'green', 'orange']):
                figure = graph_drawing.rfm_component_graph(rfm_data, component, color)
                st.pyplot(figure)
                plt.close()
                
            if st.button('Show treemap'):
                click_button(2)
            
            if st.session_state.stage >= 2:
                # Creates treemaps
                total_customers, tree_figure = graph_drawing.treemap_drawing(cluster_centers)
                st.write('Total Customers: ',total_customers)     
                st.pyplot(tree_figure)
            
            if st.button('Show scatterplot'):
                click_button(3)
            
            if st.session_state.stage >= 3:
                # Creates scatter plots for Recency, Frequency, and Monetary
                scatter_figures = graph_drawing.scatter_3d_drawing(kmeans_data)
                
                st.plotly_chart(scatter_figures)
                plt.close()

            # Prepare output deal data with cluster looked up by account ID to excel
            download_data = data_handling.create_dataframe_to_download(kmeans_data, merged_data, selected_accounts_columns, accounts_id_field)
            st.markdown('**Data Ready For Download**')

            # Call the new function to filter by ranking and display the data
            filtered_data = filter_data_by_ranking(download_data)
            
            # Generate the downloadable Excel files based on the filtered data        
            output = create_excel(download_data) # Initializes the Excel sheet
            
            # Allow users to download Deals data with assigned clusters
            st.download_button(
                label='Download Deals Data with Cluster',
                data=output,
                file_name='Accounts_segmented_data.xlsx',
                mime='application/vnd.ms-excel'
            )
            
            st.download_button(
                label='Download Deals Raw Data Excel',
                data=deal_output,
                file_name='FS Deals.xlsx',
                mime='application/vnd.ms-excel'
            )
            
            st.download_button(
                label='Download Accounts Raw Data Excel',
                data=accounts_output,
                file_name='FS Accounts.xlsx',
                mime='application/vnd.ms-excel'
            )
                
            # if st.button('Download Segmentation Data'):
            #     click_button(4)
            
            # if st.session_state.stage >= 4:
            #     st.success('Segmentation data is ready to download!')

            #     # Allow users to download Deals data with assigned clusters
            #     st.download_button(
            #         label='Download Deals Data with Cluster',
            #         data=output,
            #         file_name='Accounts_segmented_data.xlsx',
            #         mime='application/vnd.ms-excel'
            #     )
                
            #     st.download_button(
            #         label='Download Deals Raw Data Excel',
            #         data=output,
            #         file_name='FS Deals.xlsx',
            #         mime='application/vnd.ms-excel'
            #     )
                
            #     st.download_button(
            #         label='Download Accounts Raw Data Excel',
            #         data=output,
            #         file_name='FS Accounts.xlsx',
            #         mime='application/vnd.ms-excel'
            #     )
else:
    st.warning('Please upload both Deals and Accounts data files.')
