from __future__ import division
from io import BytesIO
import time
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import sys
import os
import sweetviz as sv
import streamlit.components.v1 as components
import stat


# Add the parent directory to the system path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fd.ticket_data_graph_drawing import Ticket_Graph_Drawing
from fd.ticket_data_handling import Ticket_Data

from ydata_profiling import ProfileReport

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

def data_profiling(df, df_name):
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

    # # Optionally display data types and missing values for the entire DataFrame
    # st.write(f"Data Types for {df_name} data:")
    # st.write(df.dtypes)

    # st.write(f"Missing Values in {df_name} data:")
    # st.write(df.isnull().sum())

    
# Function to convert date columns to datetime format
def convert_date_columns_to_date(df):
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

def convert_time_to_float(df, columns):
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



# Main preprocessing function
# def preprocess_data(df):
#     # Convert date columns to datetime format
#     df = convert_date_columns_to_date(df)
    
#     # Convert time columns to float
#     columns_to_convert = ['First response time (in hrs)', 'Resolution time (in hrs)', 'Time tracked']
#     df = convert_time_to_float(df, columns_to_convert)
#     return df

# def preprocess_data(df, type_options=None, group_options=None, year_options=None, use_ams=False, use_cms=False, use_valid_maintenance=False):
#     # Remove rows where 'Type' column has NaN values
#     df = df.dropna(subset=['Type'])
    
#     # Convert date columns to datetime format
#     df = convert_date_columns_to_date(df)
    
#     # Convert time columns to float
#     columns_to_convert = ['First response time (in hrs)', 'Resolution time (in hrs)', 'Time tracked']
#     df = convert_time_to_float(df, columns_to_convert)
    
#     # Set values < 0 to 0 in specified columns
#     for column in columns_to_convert:
#         if column in df.columns:
#             df[column] = df[column].clip(lower=0)
            
#     # Filter data by type_options, use_ams, and use_cms, year_options, 
#     if type_options:
#         df = df[df['Type'].isin(type_options)]
        
#     if group_options:
#         df = df[df['Group'].isin(group_options)]
            
        
#     if year_options:
#         # Ensure 'Created time' is in datetime format
#         df['Created time'] = pd.to_datetime(df['Created time'])
            
#         # Create a new column 'Created Year' based on 'Created time'
#         df['Created Year'] = df['Created time'].dt.year
            
#         # Keep rows where 'Created Year' is in year_options
#         df = df[df['Created Year'].isin(year_options)]
    
            
#         # Optionally, drop the 'Created Year' column if it's no longer needed
#         df = df.drop(columns=['Created Year'])

        
#     if use_ams:
#         df = df[df['AMS'] == True]

#     if use_cms:
#         df = df[df['CMS'] == True]
            
#     if use_valid_maintenance:
#         df = df[df['Valid Maintenance'] == 'Yes']
    
#     return df

def preprocess_data(df):
    # Remove rows where 'Type' column has NaN values
    df = df.dropna(subset=['Type'])

    # Convert date columns to datetime format
    df = convert_date_columns_to_date(df)

    # Convert time columns to float
    columns_to_convert = ['First response time (in hrs)', 'Resolution time (in hrs)', 'Time tracked']
    df = convert_time_to_float(df, columns_to_convert)

    # Set values < 0 to 0 in specified columns
    for column in columns_to_convert:
        df[column] = df[column].clip(lower=0)


    return df

def filter_data_by_ranking(download_data):
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
    
st.set_page_config(page_title='Home Page')

st.header('Ticket Data Segmenting')

ticket_data = Ticket_Data()
ticket_graph_drawing = Ticket_Graph_Drawing()

st.sidebar.success('Select the ticket data or sales data')

st.subheader('Data Load')

file = st.file_uploader('Upload your ticket data file:', ['csv', 'xlsx', 'xls'])

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

if file:
    raw_data = ticket_data.get_raw(file)
    if not raw_data.empty:
        st.subheader('Data Preprocessing')
        try:
            st.markdown('**Processed Data Frame**')
            processed_data = preprocess_data(raw_data)
            st.dataframe(processed_data)
            #df = ticket_data.create_ticket_dataframe(processed_data)
            st.success('Dataframe created successfully.')
            
        except KeyError as ke:
            st.error("""You need columns with such names: Contact ID, Client code, Company Name, Ticket ID,
             Brand, Systems Environment, Valid Maintenance, AMS, CMS, FS TRG Customer, Country, Industry, License Qty, Created time, Type, Group, Agent, Time tracked, First response time (in hrs),
             Resolution time (in hrs), Agent interactions, Customer interactions, Tags, Survey results, Product, Module, Ticket Level
            """)

            st.stop()
        except Exception as e:
            st.error(f'Error creating dataframe: {e}')
            st.stop()
        
        # User options for AMS, CMS, and ID field
        st.sidebar.header("Filter Options")
        id_field = st.sidebar.selectbox("Select Customer ID Field", options=["Contact ID", "Company Name"])
            
        use_ams = st.sidebar.checkbox('Filtered by AMS', value=False)
        use_cms = st.sidebar.checkbox('Filtered by CMS', value=False)
        use_valid_maintenance = st.sidebar.checkbox('Valid Maintenance', value=False)
            
        # Extract unique years from the 'Created Time' column
        processed_data['Created time'] = pd.to_datetime(raw_data['Created time'])
        unique_years = processed_data['Created time'].dt.year.unique()
        unique_years.sort()  # Sort the years for better user experience

        # Add sidebar dropdown for selecting years from 'Created Time'
        year_options = st.sidebar.multiselect(
            'Select Years',
            options=unique_years,
            default=unique_years  # Default to all unique years
        )

        # Ensure at least one year is selected
        if not year_options:
            st.warning("Please select at least one year.")
            year_options = unique_years  # Revert to default

        # Add sidebar dropdown for 'Type' column selection
        
        type_options = processed_data['Type'].unique()
        default_type_options = [type for type in type_options if type not in ['nan', 'Internal IT', 'Knowledge Base', 'CMS Time Log', 'Feature Request', 'Service Task', 'Presale']]
        type_options = st.sidebar.multiselect(
            'Select Ticket Types',
            options=type_options,
            default=default_type_options
        )

        # Ensure at least one ticket type is selected
        if not type_options:
            st.warning("Please select at least one ticket type.")
            type_options = processed_data['Type'].unique()  # Revert to default

        # Extract unique values from the 'Group' column
        group_options = processed_data['Group'].unique()

        # Define default values, excluding 'ERP' and 'Presales'
        default_group_options = [group for group in group_options if group not in ['ERP', 'Presales']]

        # Add sidebar dropdown for 'Group' column selection
        selected_groups = st.sidebar.multiselect(
            'Select Groups',
            options=group_options,
            default=default_group_options
        )

        # Extract unique values from the 'Group' column
        agent_options = processed_data['Agent'].unique()

        # Add sidebar dropdown for 'Group' column selection
        selected_agents = st.sidebar.multiselect(
            'Select Agents',
            options=agent_options,
            default=agent_options
        )
        
        # Ensure at least one agent is selected
        if not selected_agents:
            st.warning("Please select at least one agent.")
            selected_agents = agent_options  # Revert to default

        # Add sidebar dropdown for "Ticket Volume Columns"
        ticket_volume_columns = ['Ticket ID', 'Time tracked', 'First response time (in hrs)', 'Resolution time (in hrs)', 'Agent interactions', 'Customer interactions']
        selected_volume_columns = st.sidebar.multiselect(
            'Ticket Volume Columns',
            options=ticket_volume_columns,
            default=ticket_volume_columns
        )

        # Ensure at least one volume column is selected
        if not selected_volume_columns:
            st.warning("Please select at least one ticket volume column.")
            selected_volume_columns = ticket_volume_columns  # Revert to default
            
        # Filter data by type_options, group_options, and year_options
        if type_options is not None:
            processed_data = processed_data[processed_data['Type'].isin(type_options)]

        if group_options is not None:
            processed_data = processed_data[processed_data['Group'].isin(group_options)]
            
        if agent_options is not None:
            processed_data = processed_data[processed_data['Agent'].isin(agent_options)]

        if year_options is not None:
            processed_data['Created time'] = pd.to_datetime(processed_data['Created time'], errors='coerce')

            # Create a new column 'Created Year' based on 'Created time'
            processed_data['Created Year'] = processed_data['Created time'].dt.year

            # Keep rows where 'Created Year' is in year_options
            processed_data = processed_data[processed_data['Created Year'].isin(year_options)]

            # Optionally, drop the 'Created Year' column if it's no longer needed
            processed_data = processed_data.drop(columns=['Created Year'], errors='ignore')

        # Handle use_ams, use_cms, and use_valid_maintenance
        if use_ams:
            processed_data = processed_data[processed_data['AMS'] == True]

        if use_cms:
            df = processed_data[processed_data['CMS'] == True]

        if use_valid_maintenance:
            processed_data = processed_data[processed_data['Valid Maintenance'] == 'Yes']
        
        
        st.subheader('Data Exploration')
        
        #Data profiling before segmentation
        data_profiling(processed_data, 'Tickets')
        
        # Set default report file paths (in the same folder as the application)
        ticket_report_file_path = 'Tickets Data Report.html'
        # Generate Profiling Report Button
        if st.button('Generate Profiling Reports'):
            # Generate the reports
            st.markdown('**Generating Ticket Data Profile Report...**')
            ticket_report_file_path = generate_ydata_profiling_report(processed_data, 'Tickets Data')   
            st.success('Reports generated successfully!')

        if st.button('Display Profiling Reports'):
            # Validate if the report files exist before displaying them
            st.markdown('**Tickets Data Profile Report**')
            if os.path.exists(ticket_report_file_path):
                set_file_permissions(ticket_report_file_path)
                display_ydata_profiling_report(ticket_report_file_path)
            else:
                st.error('Tickets Data Report does not exist. Please generate the report first.')

        st.subheader('Data Segmentation')
        if st.button('Run FD Segmentation'):
            click_button(1)
        
        if st.session_state.stage >= 1:
            # Creates RFM dataframe for the segmentation
            #df_ticket = ticket_data.create_df_rfm_grouped_by_id(df, id_field, use_ams, use_cms)
            # Run RFM grouping
            df_rfm_grouped = ticket_data.create_df_rfm_grouped_by_id(processed_data, id_field=id_field, selected_volume_columns=selected_volume_columns)
           
            # Display the chosen values on the screen
            # st.markdown('**Selected Features**')
            # st.write(f"AMS Included: {use_ams}")
            # st.write(f"CMS Included: {use_cms}")
            # st.write(f"Selected ID Field: {id_field}")
            # st.write("Selected Ticket Type: {type_options}")
            # st.write("Selected Ticket Volume Columns: {selected_ticket_columns}")
            st.markdown('**Dataframe will be used in segmentation**')
            st.dataframe(df_rfm_grouped)
            
            # Creates dataframe with clusters from kmeans
            # Measure the start time
            start_time = time.time()

            # Execute the function
            # kmeans_data, cluster_centers, silhouette_score, best_k, best_random_state = ticket_data.create_kmeans_dataframe(
            #     df_ticket, df, use_ams, use_cms, id_field)
            kmeans_data, cluster_centers, silhouette_score, best_k, best_random_state = ticket_data.create_kmeans_dataframe(df_rfm_grouped, processed_data, selected_volume_columns=selected_volume_columns, id_field=id_field)
            st.markdown('**Result of Segmentation**')
            st.write("Cluster Center Dataframe")
            st.dataframe(cluster_centers)
            st.write("Kmeans Dataframe")
            st.dataframe(kmeans_data)
            data_profiling(kmeans_data,"KMeans")
            # Measure the end time
            end_time = time.time()

            # Calculate the elapsed time
            elapsed_time = end_time - start_time

            # Display the silhouette score
            #st.markdown('**Result of Segmentation**')
            #st.dataframe(cluster_centers)
            st.write('Silhouette Score: {:0.2f}'.format(silhouette_score))
            
            # Display the number of clusters and random state used
            st.write('Number of Clusters:', best_k)
            st.write('Random State:', best_random_state)

            # Display the time taken to execute the function
            st.write('Time taken to complete the clustering: {:.2f} seconds'.format(elapsed_time))
            
            st.markdown('**Segmentation Result Data Visualization**')

            # Creates graphs 
            figure = ticket_graph_drawing.recency_graph(id_field, df_rfm_grouped)
            st.pyplot(figure)
            plt.close()
            
            # Conditionally draw ticket_graph if 'Ticket ID' is in selected_volume_columns
            if 'Ticket ID' in selected_volume_columns:
                figure = ticket_graph_drawing.tickets_graph(id_field, df_rfm_grouped)
                st.pyplot(figure)
                plt.close()
            
            # Conditionally draw time_tracked_graph if 'Time tracked' is in selected_volume_columns
            if 'Time tracked' in selected_volume_columns:
                figure = ticket_graph_drawing.time_tracked_graph(df_rfm_grouped)
                st.pyplot(figure)
                plt.close()

            # Conditionally draw first_response_time_graph if 'First response time (in hrs)' is in selected_volume_columns
            if 'First response time (in hrs)' in selected_volume_columns:
                figure = ticket_graph_drawing.first_response_time_graph(df_rfm_grouped)
                st.pyplot(figure)
                plt.close()

            # Conditionally draw resolution_time_graph if 'Resolution time (in hrs)' is in selected_volume_columns
            if 'Resolution time (in hrs)' in selected_volume_columns:
                figure = ticket_graph_drawing.resolution_time_graph(df_rfm_grouped)
                st.pyplot(figure)
                plt.close()

            # Conditionally draw interactions_graph if both 'Customer interaction' and 'Agent interactions' are in selected_volume_columns
            if 'Customer interactions' in selected_volume_columns or 'Agent interactions' in selected_volume_columns:
                figure = ticket_graph_drawing.interactions_graph(df_rfm_grouped, selected_volume_columns=selected_volume_columns)
                st.pyplot(figure)
                plt.close()

                
            if st.button('Show treemap'):
                if st.session_state.stage < 2:
                    click_button(2)
            
            if st.session_state.stage >= 2:
                start_time = time.time()
                # Creates treemaps
                total_customers, tree_figure = ticket_graph_drawing.treemap_drawing(cluster_centers, selected_volume_columns)
                st.write('Total Customers: ',total_customers)                
                st.pyplot(tree_figure)
                plt.close()
                
                end_time = time.time()

                # Calculate the elapsed time
                elapsed_time = end_time - start_time
                st.write('Time taken to complete TreeMap: {:.2f} seconds'.format(elapsed_time))
        
                
            st.markdown('**Data Ready For Download**')
            download_data = ticket_data.create_ticket_dataframe_to_download(kmeans_data, processed_data, id_field=id_field)
            #st.dataframe(download_data)

            # Call the new function to filter by ranking and display the data
            filtered_data = filter_data_by_ranking(download_data)
            
            # Generate the downloadable Excel files based on the filtered data        
            
            
            output = create_excel(download_data) # Initializes the Excel sheet
            kmeans_output = create_excel(kmeans_data)
            
            # Download Excel for download_data
            if st.download_button('Download Ticket Cluster Excel', data=output, file_name='tickets-cluster.xlsx', mime='application/vnd.ms-excel'):
                st.write('Tickets-cluster Excel file downloaded!')

            # Download Excel file for kmeans_data
            if st.download_button('Download Segmentation Result Excel', data=kmeans_output, file_name='kmeans-data.xlsx', mime='application/vnd.ms-excel'):
                st.write('Kmeans-data Excel file downloaded!')

