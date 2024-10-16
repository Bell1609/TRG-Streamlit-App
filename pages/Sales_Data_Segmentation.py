from __future__ import division
import time
from matplotlib import pyplot as plt
import streamlit as st
import pandas as pd
import sys
import os
from authentication import make_sidebar # authentication instance

make_sidebar() # authentication instance


# Add the parent directory to the system path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fs.data_handling import Data_Handling
from fs.graph_drawing import Graph_Drawing

data_handling = Data_Handling()
graph_drawing = Graph_Drawing()

# Main preprocessing function
def preprocess_data(df):
    
    # Check if 'Deal : id' column exists before converting to string
    if 'Deal : id' in df.columns:
        df['Deal : id'] = df['Deal : id'].astype(str)

    # Clean and convert amount columns
    df = data_handling.clean_and_convert_amount_columns(df)
    # Drop the original columns
    
    
    # Define mixed columns to convert to strings (replace with actual columns if needed)
    df = data_handling.convert_mixed_columns_to_string(df)
    
    # Convert date columns to datetime format
    df = data_handling.convert_date_columns_to_date(df)
    
    return df


st.header('Sales Data Segmenting')

st.subheader('Data Load')

# File uploaders for Deals data and Accounts data
deals_file = st.file_uploader('Upload your Deals data file:', ['csv', 'xlsx'])
accounts_file = st.file_uploader('Upload your Accounts data file:', ['csv', 'xlsx'])

if 'stage' not in st.session_state:
    st.session_state.stage = 0

def click_button(stage):
    st.session_state.stage = stage





# Mandatory fields for deals data
mandatory_deals_fields = ['Deal : Account ID', 'Deal : Account name', 'Deal : Expected close date', 'Deal : Total Deal Value', 'Deal : Deal stage']
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
        # deal_output = create_excel(deals_data)
        # accounts_output = create_excel(accounts_data)

        # Ensure the ID fields are treated as strings before merging
        accounts_data['SalesAccount : id'] = accounts_data['SalesAccount : id'].astype(str)

        # Add 'Deal : Account ID' column to Deals DataFrame
        deals_data = data_handling.add_account_id_column(deals_data, accounts_data)

        
        # Validate mandatory fields in Deals and Accounts data
        if not validate_columns(deals_data, mandatory_deals_fields, 'Deals'):
            st.stop()
            
        if not validate_columns(accounts_data, mandatory_accounts_fields, 'Accounts'):
            st.stop()

        st.success('Data files uploaded successfully')
        # Duong change
        # Checkbox for filtering by 'TRG Customer'
        filter_trg_customer = st.sidebar.checkbox('TRG Customer Only')
        
        
        # Ensure the 'Deal : Expected close date' column is in datetime format
        deals_data['Deal : Expected close date'] = pd.to_datetime(deals_data['Deal : Expected close date'], errors='coerce')

        # Extract the min and max date range from the 'Deal : Expected close date' column
        min_date = deals_data['Deal : Expected close date'].min()
        max_date = deals_data['Deal : Expected close date'].max()

        # Generate a list of month-year options between the min and max dates
        month_year_range = pd.date_range(min_date, max_date, freq='MS').strftime('%m-%Y').tolist()

        # Sidebar dropdown for selecting the 'From Month'
        st.sidebar.write('Select Expected Close Month')
        from_month = st.sidebar.selectbox(
            'From:',
            options=month_year_range,
            index=0  # Default to the first month
        )

        # Sidebar dropdown for selecting the 'To Month'
        to_month = st.sidebar.selectbox(
            'To:',
            options=month_year_range,
            index=len(month_year_range) - 1  # Default to the last month
        )

        # Validate that 'To Month' is greater than or equal to 'From Month'
        from_month_index = month_year_range.index(from_month)
        to_month_index = month_year_range.index(to_month)

        if to_month_index < from_month_index:
            st.sidebar.error("'To Month' must be greater than or equal to 'From Month'. Please select valid options.")

        # Report options for user selection
        report_options = ['Won', 'Pipeline']
        selected_reports = st.sidebar.multiselect('Select Report Type', options=report_options, default=['Won'])

        # Validate 'Report Type' selection
        if not selected_reports:
            st.sidebar.error("Please select at least one type of report.")
            st.stop()  # Stop execution if no report type is selected

        # Deal stage filtering logic based on report type
        if 'Won' in selected_reports and 'Pipeline' in selected_reports:
            # If both "Won" and "Pipeline" are selected, include "Won" and all other stages except "Lost"
            stage_options = [stage for stage in deals_data['Deal : Deal stage'].unique() if stage != 'Lost']
        elif 'Won' in selected_reports:
            # If only "Won" is selected, restrict stages to "Won"
            stage_options = ['Won']
        elif 'Pipeline' in selected_reports:
            # If only "Pipeline" is selected, include all stages except "Won" and "Lost"
            stage_options = [stage for stage in deals_data['Deal : Deal stage'].unique() if stage not in ['Won', 'Lost']]


        # Display filtered deal stages for selection
        selected_stages = st.sidebar.multiselect('Select Deal Stages', options=stage_options, default=stage_options)

        # Validate 'Deal Stage' selection
        if not selected_stages:
            st.sidebar.error("Please select at least one Deal Stage.")
            st.stop()  # Stop execution if no deal stage is selected

        # Filtering based on sidebar selections, including the new product filter
        deals_data_filtered = deals_data[
            (deals_data['Deal : Deal stage'].isin(selected_stages)) &
            (deals_data['Deal : Expected close date'] >= pd.to_datetime(from_month, format='%m-%Y')) &
            (deals_data['Deal : Expected close date'] <= pd.to_datetime(to_month, format='%m-%Y')) 
        ]
        
        # Project category options: Recurring and Non-Recurring
        project_categories = ['Recurring Projects', 'Non-Recurring Projects']
        selected_categories = st.sidebar.multiselect('Select Project Category', options=project_categories, default=project_categories)

        # Validate 'Project Category' selection
        if not selected_categories:
            st.sidebar.error("Please select at least one Project Category.")
            st.stop()  # Stop execution if no category is selected

        # Define the recurring and non-recurring project types
        recurring_project_types = ['ARR', 'Existing - Additional users (No services)']
        non_recurring_project_types = [ptype for ptype in deals_data_filtered['Deal : Project type'].unique() if ptype not in recurring_project_types]

        # Filter project types based on selected categories
        if 'Recurring Projects' in selected_categories and 'Non-Recurring Projects' in selected_categories:
            # If both categories are selected, show all project types
            type_options = deals_data_filtered['Deal : Project type'].unique()
        elif 'Recurring Projects' in selected_categories:
            # If only 'Recurring Projects' is selected, show only recurring project types
            type_options = recurring_project_types
        elif 'Non-Recurring Projects' in selected_categories:
            # If only 'Non-Recurring Projects' is selected, show only non-recurring project types
            type_options = non_recurring_project_types

        # Display filtered project types for selection
        selected_types = st.sidebar.multiselect('Select Project Type', options=type_options, default=type_options)

        # Validate 'Deal : Project Type' selection
        if not selected_types:
            st.sidebar.error("Please select at least one Project Type.")
            st.stop()  # Stop execution if no project type is selected

        # Filtering based on sidebar selections
        deals_data_filtered = deals_data_filtered[(deals_data_filtered['Deal : Project type'].isin(selected_types))]

        # Add a sidebar selectbox for 'Deal : Type of Renewal' if it exists in the dataset
        renewal_options = deals_data_filtered['Deal : Type of Renewal'].unique()
    
        selected_type_of_renewal = st.sidebar.multiselect(
            'Select Type of Renewal:',
            options=renewal_options,
            default=renewal_options if renewal_options.size > 0 else None
            )
        if not selected_type_of_renewal:
            st.sidebar.error("Please select at least one Type of Renewal.")
            st.stop()  # Stop execution if no project type is selected

        deals_data_filtered = deals_data_filtered[(deals_data_filtered['Deal : Type of Renewal'].isin(selected_type_of_renewal))]
        
        # Duong change
        st.subheader('Data Exploration')
        st.markdown('**Filtered Deals Data**')
        st.dataframe(deals_data_filtered)
        st.markdown('**Loaded Accounts Data**')
        st.dataframe(accounts_data)


        #Data profiling before segmentation
        data_handling.data_profiling(deals_data_filtered, 'Deals')
        data_handling.data_profiling(accounts_data, 'Accounts')
        
        # Set default report file paths (in the same folder as the application)
        deals_report_file_path = 'Deals Data Report.html'
        accounts_report_file_path = 'Accounts Data Report.html'
        
        # # Generate Profiling Report Button
        # if st.button('Generate Profiling Reports'):
        #     # Generate the reports
        #     st.markdown('**Generating Filtered Deals Data Profile Report...**')
        #     deals_report_file_path = generate_ydata_profiling_report(deals_data_filtered, 'Deals Data')
                
        #     st.markdown('**Generating Accounts Data Profile Report...**')
        #     accounts_report_file_path = generate_ydata_profiling_report(accounts_data, 'Accounts Data')
                
        #     st.success('Reports generated successfully!')

        # if st.button('Display Profiling Reports'):
        #     # Validate if the report files exist before displaying them
        #     st.markdown('**Deals Data Profile Report**')
        #     if os.path.exists(deals_report_file_path):
        #         set_file_permissions(deals_report_file_path)
        #         display_ydata_profiling_report(deals_report_file_path)
        #     else:
        #         st.error('Deals Data Report does not exist. Please generate the report first.')

        #     st.markdown('**Accounts Data Profile Report**')
        #     if os.path.exists(accounts_report_file_path):
        #         set_file_permissions(deals_report_file_path)
        #         display_ydata_profiling_report(accounts_report_file_path)
        #     else:
        #         st.error('Accounts Data Report does not exist. Please generate the report first.')


        # st.subheader('Data Preprocessing')
        # # List columns for both files
        # deals_columns = deals_data_filtered.columns.tolist()
        # accounts_columns = accounts_data.columns.tolist()

        # # Choose columns for merging
        # st.markdown('**Select Columns for Merging DataFrames**')

        # # Ensure mandatory fields are selected
        # selected_deals_columns = st.sidebar.multiselect('Select Deals Columns:', deals_columns, default=mandatory_deals_fields)
        # selected_accounts_columns = st.sidebar.multiselect('Select Accounts Columns:', accounts_columns, default=mandatory_accounts_fields)

        # if not all(field in selected_deals_columns for field in mandatory_deals_fields):
        #     st.error(f'You must select these mandatory fields from the Deals data: {mandatory_deals_fields}')
        #     st.stop()

        # if not all(field in selected_accounts_columns for field in mandatory_accounts_fields):
        #     st.error(f'You must select these mandatory fields from the Accounts data: {mandatory_accounts_fields}')
        #     st.stop()

        # # Select ID fields for merging
        # st.markdown('**Select ID Fields for Merging**')
        # # Set default values for ID fields
        # default_deals_id_field = 'Deal : Account ID'
        # default_accounts_id_field = 'SalesAccount : id'

        # # Ensure that the default values are part of the selectable options
        # if default_deals_id_field not in selected_deals_columns:
        #     st.warning(f"Default Deals ID field '{default_deals_id_field}' is not in the selected deals columns.")
        #     default_deals_id_field = None  # Remove the default value if it doesn't exist

        # if default_accounts_id_field not in selected_accounts_columns:
        #     st.warning(f"Default Accounts ID field '{default_accounts_id_field}' is not in the selected accounts columns.")
        #     default_accounts_id_field = None  # Remove the default value if it doesn't exist

        # # Create selectboxes with the default values
        # deals_id_field = st.sidebar.selectbox('Select Deals ID Field:', selected_deals_columns, index=selected_deals_columns.index(default_deals_id_field) if default_deals_id_field else 0)
        # accounts_id_field = st.sidebar.selectbox('Select Accounts ID Field:', selected_accounts_columns, index=selected_accounts_columns.index(default_accounts_id_field) if default_accounts_id_field else 0)

        # # Filter deals data by 'Deal : Probability (%)'
        # prob_min, prob_max = st.sidebar.slider('Select Probability (%) Range:', 0, 100, (0, 100))
        # deals_data['Deal : Probability (%)'] = deals_data['Deal : Probability (%)'].astype(int)
        # deals_data = deals_data[(deals_data['Deal : Probability (%)'] >= prob_min) & (deals_data['Deal : Probability (%)'] <= prob_max)]

        default_deals_id_field = 'Deal : Account ID'
        default_accounts_id_field = 'SalesAccount : id'
        
        try:
            # Merge dataframes based on selected ID fields
            merged_data = deals_data[mandatory_deals_fields].merge(accounts_data[mandatory_accounts_fields], left_on=default_deals_id_field, right_on=default_accounts_id_field, how='left')
            
            # Check if the filter_trg_customer flag is set
            if filter_trg_customer:
                if 'Account : TRG Customer' in merged_data.columns:
                    merged_data = merged_data[merged_data['Account : TRG Customer'] == 'Yes']
                else:
                    st.warning('Column "Account : TRG Customer" not found in merged data.')
            # st.success('DataFrames merged successfully.')
            
            # st.dataframe(merged_data)
        except KeyError as ke:
            st.error(f'Error merging data: {ke}')
            st.stop()

        # Run RFM Segmentation
        if st.button('Run RFM Segmentation'):
            click_button(1)
        
        if st.session_state.stage >= 1:
            # Creates RFM dataframe for the segmentation
            rfm_data = data_handling.create_rfm_dataframe(merged_data, default_accounts_id_field)  # Use the new ID column for RFM segmentation
            st.markdown('**RFM Data Frame**')
            st.dataframe(rfm_data)
            
            # Measure the start time
            start_time = time.time()

            # Creates dataframe with clusters from kmeans
            kmeans_data, cluster_centers, silhouette_score, best_k, best_random_state = data_handling.create_kmeans_dataframe(rfm_data, default_accounts_id_field)
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
            download_data = data_handling.create_dataframe_to_download(kmeans_data, merged_data, mandatory_accounts_fields, default_accounts_id_field)
            st.markdown('**Data Ready For Download**')

            # Call the new function to filter by ranking and display the data
            filtered_data = data_handling.filter_data_by_ranking(download_data)
            
            # Generate the downloadable Excel files based on the filtered data        
            output = data_handling.create_excel(download_data) # Initializes the Excel sheet
            #deal_ouput = create_excel(deals_data_filtered)
            
            # Allow users to download Deals data with assigned clusters
            st.download_button(
                label='Download Deals Data with Cluster',
                data=output,
                file_name='Accounts_segmented_data.xlsx',
                mime='application/vnd.ms-excel'
            )
            
            # st.download_button(
            #     label='Download Deals Raw Data Excel',
            #     data=deal_ouput,
            #     file_name='FS Deals.xlsx',
            #     mime='application/vnd.ms-excel'
            # )
            
            # st.download_button(
            #     label='Download Accounts Raw Data Excel',
            #     data=accounts_output,
            #     file_name='FS Accounts.xlsx',
            #     mime='application/vnd.ms-excel'
            # )
                
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
