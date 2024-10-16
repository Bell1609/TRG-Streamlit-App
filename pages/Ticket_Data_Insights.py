from __future__ import division
from io import BytesIO
import pandas as pd
import streamlit as st
import sys
import os
from authentication import make_sidebar # authentication instance

make_sidebar() # authentication instance

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

st.title('Helpdesk Ticket Data Insights')

ticket_data = Ticket_Data()
ticket_graph_drawing = Ticket_Graph_Drawing()

st.sidebar.success('Select the ticket data or sales data')

st.subheader('Data Load')

# File uploaders
ticket_file = st.file_uploader('Upload your ticket data file:', ['csv', 'xlsx'])
#employee_file = st.file_uploader('Upload your employee data file:', ['csv', 'xlsx'])
# Add a file uploader for the Companies data
companies_file = st.file_uploader("Upload FD Companies.xlsx", type=['csv','xlsx'])

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

#if ticket_file and companies_file:
if ticket_file and companies_file:
    ticket_raw_data = ticket_data.get_raw(ticket_file)
    #employees_transformed = ticket_data.load_and_transform_employees(employee_file)
    # Call the function to load and process the data
    companies_df = ticket_data.load_and_process_companies(companies_file)
    # Display the first few rows of the processed data
    st.markdown("**Processed Companies Data**")
    st.dataframe(companies_df)
    
    if not ticket_raw_data.empty and not companies_df.empty:
    #if not ticket_raw_data.empty:
        #st.subheader('Data Preprocessing')
        try:
            #st.markdown('**Processed Data Frame**')
            processed_data = preprocess_data(ticket_raw_data)
            #st.dataframe(processed_data)
            # st.markdown('**Employees Data**')
            # st.dataframe(employees_transformed)
            # Call the function to load and process the data
            companies_df = ticket_data.load_and_process_companies(companies_file)
            st.markdown('**Companies Data Transposed**')
            st.dataframe(companies_df)

            
            
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
        
        #Get the average handling time per ticket and other information from tickets data
        ticket_count = processed_data_filtered['Ticket ID'].count()
        avg_time_tracked = processed_data_filtered['Time tracked'].mean()
        avg_agent_interactions = processed_data_filtered['Agent interactions'].mean()
        avg_customer_interactions = processed_data_filtered['Customer interactions'].mean()
        
        
        st.subheader('Data Exploration')
        st.markdown('**HelpDesk Performance Data**')

        # Update helpdesk performance based on filtered data
        helpdesk_performance = ticket_data.create_helpdesk_performance(processed_data_filtered, support_percentage, avg_time_tracked)
        st.dataframe(helpdesk_performance)
        ticket_n_contact_by_company = ticket_data.create_ticket_and_contact_grouped_by_company(processed_data_filtered)
        
        # Calculate the count of valid companies grouped by Month
        valid_companies_by_month = companies_df.groupby('Month').agg(Valid_Company_Count=('Company Name', 'count')).reset_index()

        # Right join valid companies data with the helpdesk performance data on 'Month'
        combined_df = pd.merge(valid_companies_by_month, helpdesk_performance[['Month', 'Company_Count']], on='Month', how='right')

        # Fill NaN values in 'Valid_Company_Count' with 0
        combined_df['Valid_Company_Count'].fillna(0, inplace=True)
        
        # Calculate the average percentage of Company_Count / Valid_Company_Count
        combined_df['Percentage'] = combined_df['Company_Count'] / combined_df['Valid_Company_Count']
        avg_percentage = combined_df['Percentage'].mean() * 100

        # Display the combined DataFrame
        st.markdown("**Combined Data with Valid and Company Count**")
        st.dataframe(combined_df)
        

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
        # Display Data Profiling section
        st.subheader('Data Profiling')

        # Profiling for Tickets
        st.write(f"**Data Profiling for Tickets**")
        st.write(processed_data_filtered.describe())

        # Write additional important information for Tickets
        
        st.write(f"- Total Ticket Count: {ticket_count}")
        st.write(f"- Average Time Tracked per Ticket: {avg_time_tracked:.2f} hrs")
        st.write(f"- Average Agent Interactions per Ticket: {avg_agent_interactions:.2f}")
        st.write(f"- Average Customer Interactions per Ticket: {avg_customer_interactions:.2f}")

        # Profiling for HelpDesk Performance
        st.write(f"**Data Profiling for HelpDesk Performance**")
        st.write(helpdesk_performance.describe())

        # Write additional important information for HelpDesk Performance
        avg_tickets_per_month = helpdesk_performance['Ticket_Count'].mean()
        avg_agents_needed_per_month = helpdesk_performance['Agent_Needed'].mean()
        avg_capacity_needed_per_month = helpdesk_performance['Capacity_Needed'].mean()
        st.write(f"- Average Tickets per Month: {avg_tickets_per_month:.2f}")
        st.write(f"- Average Agents Needed per Month: {avg_agents_needed_per_month:.2f}")
        st.write(f"- Average Capacity Needed per Month: {avg_capacity_needed_per_month:.2f}")
        
        # Profiling for Ticket Data Grouped by Company
        st.write(f"**Data Profiling for Ticket Data Grouped by Company**")
        st.write(ticket_n_contact_by_company.describe())
        
        # Write additional important information for Ticket Data Grouped by Company
        unique_companies = ticket_n_contact_by_company['Company Name'].nunique()
        avg_tickets_per_company = ticket_n_contact_by_company['Ticket_Count'].mean()
        avg_contacts_per_company = ticket_n_contact_by_company['Contact_Count'].mean()
        st.write(f"- Unique Companies: {unique_companies}")
        st.write(f"- Average Tickets per Company per Month: {avg_tickets_per_company:.2f}")
        st.write(f"- Average Contacts per Company per Month: {avg_contacts_per_company:.2f}")
        
        # Profiling for Company Data Grouped by Month
        st.write(f"**Data Profiling for Company Data Grouped by Month**")
        st.write(combined_df.describe())
        
        # Write additional important information for valid company Data Grouped by month
        combined_df['Percentage'] = combined_df['Company_Count'] / combined_df['Valid_Company_Count']
        avg_percentage = combined_df['Percentage'].mean() * 100
        avg_valid_company = combined_df['Valid_Company_Count'].mean()
        avg_company_count = combined_df['Company_Count'].mean()
        st.write(f"- Average Valid Maintenance Companies per Month: {avg_valid_company:.2f}")
        st.write(f"- Average Companies Having Tickets per Month: {avg_company_count:.2f}")
        st.write(f"- Average Percentage of Companies Having Tickets / Valid Maintenance Companies: {avg_percentage:.2f}%")    
    
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
            ax1.set_xticklabels(months[::3], rotation=90, ha='right')  # Set labels to display every 3rd month

            # Display the plot
            st.pyplot(fig1)
            
        # Display a subheader for the section
        st.subheader("Valid Maintenance Companies Count vs Companies Having Tickets Over Time")

        # Call the function and get the figure
        fig = ticket_graph_drawing.plot_valid_vs_total_company_count(combined_df)

        # Display the figure in Streamlit
        st.pyplot(fig)


        # Generate downloadable Excel files        
        # helpdesk_output = create_excel(helpdesk_performance)
        # raw_output = create_excel(processed_data)
        #processed_data_output = create_excel(processed_data_filtered)

        # Download buttons
        # st.download_button('Download HelpDesk Performance Excel', data=helpdesk_output, file_name='helpdesk_performance.xlsx', mime='application/vnd.ms-excel')
        # st.download_button('Download Ticket Raw Excel', data=raw_output, file_name='raw_tickets.xlsx', mime='application/vnd.ms-excel')
        #st.download_button('Download Ticket Processed Excel', data=processed_data_output, file_name='processed_tickets.xlsx', mime='application/vnd.ms-excel')

    
