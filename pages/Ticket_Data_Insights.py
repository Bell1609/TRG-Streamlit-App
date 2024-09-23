from __future__ import division
from io import BytesIO
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import sys
import os
import matplotlib.dates as mdates

# Add the parent directory to the system path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fd.ticket_data_graph_drawing import Ticket_Graph_Drawing
from fd.ticket_data_handling import Ticket_Data

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
            df[col] = pd.to_datetime(df[col], dayfirst=True, format='mixed', errors='coerce')
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                df[col] = df[col].dt.tz_localize(None)
    
    return df

def convert_time_to_float(df, columns):
    def time_to_float(time_str):
        try:
            parts = time_str.split(':')
            if len(parts) == 3:
                hh, mm, ss = parts
            elif len(parts) == 2:
                hh, mm = parts
                ss = '0'
            else:
                return None
            return float(hh) + float(mm) / 60 + float(ss) / 3600
        except (ValueError, AttributeError):
            return None

    for column in columns:
        if column in df.columns:
            df[column] = df[column].apply(time_to_float)

    return df

def preprocess_data(df):
    df = df.dropna(subset=['Type'])
    df = convert_date_columns_to_date(df)
    columns_to_convert = ['First response time (in hrs)', 'Resolution time (in hrs)', 'Time tracked']
    df = convert_time_to_float(df, columns_to_convert)
    return df

def create_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=False)
    writer.close()
    return output.getvalue()

st.set_page_config(page_title='Home Page')

st.header('Helpdesk Ticket Data Insights')

ticket_data = Ticket_Data()
ticket_graph_drawing = Ticket_Graph_Drawing()

st.sidebar.success('Select the ticket data or sales data')

st.subheader('Data Load')

# File uploaders
ticket_file = st.file_uploader('Upload your ticket data file:', ['csv', 'xlsx'])
#employee_file = st.file_uploader('Upload your employee data file:', ['csv', 'xlsx'])

# Input box for support utilization percentage
support_percentage = st.text_input('Enter Support Utilization Percentage:', value='65')

# Validate numeric input for support utilization percentage
if support_percentage.isdigit():
    support_percentage = float(support_percentage)
else:
    st.error("Please enter a valid numeric value for support utilization percentage.")
    support_percentage = 65.0  # Default to 65 if input is invalid

if 'stage' not in st.session_state:
    st.session_state.stage = 0

def click_button(stage):
    st.session_state.stage = stage

#if ticket_file and employee_file:
if ticket_file:
    ticket_raw_data = ticket_data.get_raw(ticket_file)
    #employees_transformed = ticket_data.load_and_transform_employees(employee_file)
    
    #if not ticket_raw_data.empty and not employees_transformed.empty:
    if not ticket_raw_data.empty:
        st.subheader('Data Preprocessing')
        try:
            st.markdown('**Processed Data Frame**')
            processed_data = preprocess_data(ticket_raw_data)
            st.dataframe(processed_data)
            # st.markdown('**Employees Data**')
            # st.dataframe(employees_transformed)
        except KeyError as ke:
            st.error("""You need columns with such names: Contact ID, Company Name, Ticket ID,
             Brand, Systems Environment, Valid Maintenance, AMS, CMS, FS TRG Customer, Country, Industry, License Qty, Created time, Type, Group, Agent, Time tracked, First response time (in hrs),
             Resolution time (in hrs), Agent interactions, Customer interactions, Tags, Survey results, Product, Module, Ticket Level
            """)
        except Exception as e:
            st.error(f"Error creating dataframe: {e}")
            st.stop()
        
        # Sidebar section for filters
        st.sidebar.header("Filter Options")
        
        # Extract unique years from 'Created time' column
        processed_data['Created time'] = pd.to_datetime(ticket_raw_data['Created time'])
        unique_years = processed_data['Created time'].dt.year.unique()
        unique_years.sort()

        # Sidebar dropdown for selecting years
        year_options = st.sidebar.multiselect(
            'Select Years',
            options=unique_years,
            default=unique_years
        )

        if not year_options:
            st.warning("Please select at least one year.")
            year_options = unique_years
            
        # Sidebar for selecting department
        st.sidebar.header("Select Department")
        selected_departments = st.sidebar.multiselect(
            "Choose Departments", 
            options=['SUP', 'TEC', 'CS'],
            default=['SUP', 'TEC']  # Default selected department
        )

        # # Filter employees_transformed based on the selected departments
        # if selected_departments:
        #     filtered_employees = employees_transformed[employees_transformed['Dept'].isin(selected_departments)]
        # else:
        #     st.error("Please select at least one department.")
        #     filtered_employees = pd.DataFrame()  # Reset to an empty DataFrame if no department is selected
        #     st.stop()
            
        # Selected group options of ticket data based on the selected departments
        default_group_options = []
        if 'SUP' in selected_departments:
            default_group_options += ['Left', 'Support', 'Premium', 'GMT+7', 'Manager', 'Part Time']
        if 'TEC' in selected_departments:
            default_group_options += ['Technical & Cloud', 'Technical Left']
        if 'CS' in selected_departments:
            default_group_options += ['CS']

        # Ensure unique options and sort them
        default_group_options = list(set(default_group_options))
        default_group_options.sort()

        group_options = processed_data['Group'].unique()
        selected_groups = st.sidebar.multiselect('Select Groups', options=group_options, default=default_group_options)

        # Service type filter options
        service_type_options = ['ASM', 'AMS', 'CMS']
        selected_service_types = st.sidebar.multiselect('Select Service Type(s)', options=service_type_options, default=service_type_options)

        # Initialize an empty list for selected types
        selected_types = []
        # Mapping service type to 'Type' field dynamically based on multi-select choices
        if 'AMS' in selected_service_types:
            selected_types.extend(['Application Manage Service'])
        if 'CMS' in selected_service_types:
            selected_types.extend(['Cloud Manage Service'])
        if 'ASM' in selected_service_types:
            # Only include 'ASM' related types (excluding 'AMS' and 'CMS')
            type_options = processed_data['Type'].unique()
            asm_types = [type for type in type_options if type not in ['Application Manage Service', 'Cloud Manage Service']]
            selected_types.extend(asm_types)

        # Remove duplicates from selected_types
        selected_types = list(set(selected_types))

        # Display the filtered 'Type' options as a multi-select
        selected_type_options = st.sidebar.multiselect('Select Type(s)', options=selected_types, default=selected_types)

        # Filtering based on sidebar selections
        processed_data_filtered = processed_data[
            (processed_data['Group'].isin(selected_groups)) &
            (processed_data['Created time'].dt.year.isin(year_options))
        ]
             
        
        # further filter by selected type accordingly with service type
        processed_data_filtered = processed_data_filtered[(processed_data_filtered['Type'].isin(selected_type_options))]
        
        # Additional checkbox filter options
        use_valid_maintenance = st.sidebar.checkbox('Valid Maintenance', value=False)
        
        if use_valid_maintenance:
            processed_data_filtered = processed_data_filtered[processed_data_filtered['Valid Maintenance'] == 'Yes']
        
        #Get the average handling time per ticket
        AHT = processed_data_filtered['Time tracked'].quantile(0.75)
        
        st.subheader('Data Exploration')
        st.markdown('**HelpDesk Performance Data**')

        # Update helpdesk performance based on filtered data
        helpdesk_performance = ticket_data.create_helpdesk_performance(processed_data_filtered, support_percentage, AHT)
        st.dataframe(helpdesk_performance)
        ticket_n_contact_by_company = ticket_data.create_ticket_and_contact_grouped_by_company(processed_data_filtered)
        

        # Check if 'Part Time' is not in selected_groups
        # if 'Part Time' not in selected_groups:
        #     # Filter out Staff Name whose 'Type' is 'Part time'
        #     filtered_employees = filtered_employees[filtered_employees['Type'] != 'Part time']

        # # Check if filtered_employees is not empty before accessing 'Status'
        # if not filtered_employees.empty and 'Status' in filtered_employees.columns:
        #     # Calculate employed staff per month and merge
        #     employed_per_month = filtered_employees[filtered_employees['Status'] == 'Employed'] \
        #         .groupby('Month').size().reset_index(name='Employed Count')

        #     # Merge helpdesk performance with employed staff count
        #     sup_performance_merged = pd.merge(helpdesk_performance, employed_per_month, on='Month', how='left')
        #     st.dataframe(sup_performance_merged)
        # else:
        #     st.error("No employed staff data available to display.")
        #     st.stop()

        st.markdown('**Tickets and Contacts Grouped by Company**')
        st.dataframe(ticket_n_contact_by_company)
        
        # Data profiling before segmentation
        st.subheader('Data Profiling')
        st.write(f"**Data Profiling for Tickets**")
        st.write(processed_data_filtered.describe())
        
        st.write(f"**Data Profiling for HelpDesk Performance**")
        st.write(helpdesk_performance.describe())
        
        st.write(f"**Data Profiling for Ticket Data Grouped by Company**")
        st.write(ticket_n_contact_by_company.describe())
    
        # Display the dropdown for selecting a column to plot        
        st.subheader('Helpdesk Tickets Data Insights')
        selected_column1 = st.selectbox(
            "Select a Metric to visualize",
            helpdesk_performance.columns.drop('Month')  # Exclude 'Month' from the options
        )
         
        if selected_column1:
            # Plot the selected column
            fig1, ax1 = ticket_graph_drawing.visualize_helpdesk_performance_column(helpdesk_performance, selected_column1)

            # Set the x-axis ticks manually to display months every 3rd month
            months = helpdesk_performance['Month'].unique()
            ax1.set_xticks(range(0, len(months), 3))  # Set tick positions every 3 months
            ax1.set_xticklabels(months[::3], rotation=45, ha='right')  # Set labels to display every 3rd month

            # Display the plot
            st.pyplot(fig1)


        # Generate downloadable Excel files        
        # helpdesk_output = create_excel(helpdesk_performance)
        # raw_output = create_excel(processed_data)
        #processed_data_output = create_excel(processed_data_filtered)

        # Download buttons
        # st.download_button('Download HelpDesk Performance Excel', data=helpdesk_output, file_name='helpdesk_performance.xlsx', mime='application/vnd.ms-excel')
        # st.download_button('Download Ticket Raw Excel', data=raw_output, file_name='raw_tickets.xlsx', mime='application/vnd.ms-excel')
        #st.download_button('Download Ticket Processed Excel', data=processed_data_output, file_name='processed_tickets.xlsx', mime='application/vnd.ms-excel')

    
