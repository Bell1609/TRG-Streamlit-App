import numpy as np
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Function to load Excel file and display dataframe
def load_data(file):
    try:
        df = pd.read_excel(file)
        return df
    except Exception as e:
        st.error(f"Error loading the file: {e}")
        return None

# Function for data profiling and sum statistics
def data_profiling(df):
    st.subheader('Data Profiling: Basic Statistics')

    # Get the descriptive statistics
    desc_stats = df.describe(include='all')

    # Select numeric columns for sum calculation
    numeric_df = df.select_dtypes(include=['number'])

    # Calculate sum for numeric columns
    sum_stats = numeric_df.sum()

    # Convert sum stats to a DataFrame and transpose to match the shape of `desc_stats`
    sum_stats_df = pd.DataFrame(sum_stats, columns=['sum']).transpose()

    # Append the sum row to the describe DataFrame
    desc_stats_with_sum = pd.concat([desc_stats, sum_stats_df])

    # Display the combined DataFrame with sum row
    st.write(desc_stats_with_sum)


# Function to filter data based on Name and Month
def filter_data(df):
    names = df['Name'].unique().tolist()
    months = df['Month'].unique().tolist()

    # Move the multiselect options to the sidebar
    selected_names = st.sidebar.multiselect('Select Name(s)', names)
    selected_months = st.sidebar.multiselect('Select Month(s)', months)

    # You can then use these selected values in the rest of your code
    df = df[(df['Name'].isin(selected_names)) & (df['Month'].isin(selected_months))]


    st.subheader('Filtered Data')

    return df



#Function to plot pie chart for task percentages

# def plot_pie_chart(df):
#     task_columns = ['Billable in contract', 'CMS','CR FOC', 'Under Estimation', 'Implementation Issue', 'Presales', 'Shadow','Support Task']
#     available_task_columns = [col for col in task_columns if col in df.columns]

#     if available_task_columns:
#         # Replace NaN values with 0 before summing the columns
#         task_sums = df[available_task_columns].sum().replace(np.nan, 0)

#         # Ensure task sums are greater than 0 to avoid plotting empty charts
#         task_sums = task_sums[task_sums > 0]

#         if task_sums.empty:
#             st.error("No data available for selected task categories.")
#             return

#         # Create the pie chart
#         fig, ax = plt.subplots()
#         wedges, texts, autotexts = ax.pie(task_sums, labels=task_sums.index, autopct='%1.1f%%', startangle=90)

#         # Add legend to the right of the pie chart
#         ax.legend(wedges, task_sums.index, title="Task Categories", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))

#         # Equal aspect ratio ensures that pie is drawn as a circle
#         ax.axis('equal')  

#         # Display the pie chart in Streamlit
#         st.pyplot(fig)
#     else:
#         st.error("The required task columns are not found in the dataset.")


def plot_pie_chart(df):
    task_columns = ['Billable in contract', 'CMS', 'CR FOC', 'Under Estimation', 'Implementation Issue', 'Presales', 'Shadow', 'Support Task']
    available_task_columns = [col for col in task_columns if col in df.columns]

    if available_task_columns:
        # Replace NaN values with 0 before summing the columns
        task_sums = df[available_task_columns].sum().replace(np.nan, 0)

        # Ensure task sums are greater than 0 to avoid plotting empty charts
        task_sums = task_sums[task_sums > 0]

        if task_sums.empty:
            st.error("No data available for selected task categories.")
            return

        # Calculate percentages
        total_sum = task_sums.sum()
        task_percentages = (task_sums / total_sum) * 100

        # Group tasks with percentage < 5% into "Others"
        threshold = 5
        small_tasks = task_percentages[task_percentages < threshold]
        large_tasks = task_percentages[task_percentages >= threshold]

        if not small_tasks.empty:
            others_sum = small_tasks.sum()
            task_percentages = pd.concat([large_tasks, pd.Series([others_sum], index=['Others'])])

        # Create a custom autopct function to hide percentages < 5%
        def autopct_format(pct):
            return ('%1.1f%%' % pct) if pct >= threshold else ''

        # Create the pie chart
        fig, ax = plt.subplots()
        wedges, texts, autotexts = ax.pie(
            task_percentages, 
            labels=task_percentages.index, 
            autopct=autopct_format, 
            startangle=90, 
            pctdistance=0.85,  # This pushes the percentages inside the slices
            labeldistance=1.05  # This adjusts the label distance to avoid overlap
        )

        # Adjust text sizes to avoid overlap
        for text in autotexts:
            text.set_fontsize(10)
        
        for text in texts:
            text.set_fontsize(10)

        # Add legend to the right of the pie chart
        ax.legend(wedges, task_percentages.index, title="Task Categories", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))

        # Equal aspect ratio ensures that pie is drawn as a circle
        ax.axis('equal')

        # Use tight layout to prevent overlap
        plt.tight_layout()

        # Display the pie chart in Streamlit
        st.pyplot(fig)

        # Create the full task breakdown for the table
        st.subheader("Task Details with Percentages")
        
        # Combine large tasks and small tasks (Others details)
        task_details_large = pd.DataFrame({
            'Task': large_tasks.index,
            'Percentage': large_tasks.values
        })

        task_details_others = pd.DataFrame({
            'Task': small_tasks.index,
            'Percentage': small_tasks.values
        })

        # Append the breakdown of small tasks (Others)
        task_details_full = pd.concat([task_details_large, task_details_others])

        st.table(task_details_full)

    else:
        st.error("The required task columns are not found in the dataset.")


# Streamlit app setup
st.title('Task Data Analysis App')

# Upload Excel file
file = st.file_uploader('Upload your Excel file', type=['xlsx'])

if file:
    df = load_data(file)
    
    if df is not None:
        # Display the DataFrame
        st.subheader('DataFrame')
        st.dataframe(df)

        # Perform data profiling
        data_profiling(df)

       
        # Get unique months from the dataframe
        months = df['Month'].unique()

        # Move the multiselect options to the sidebar, with the default being all months
        selected_months = st.sidebar.multiselect('Select Month(s)', months, default=months)

        # You can also apply the same logic to 'Name' if needed
        names = df['Name'].unique()
        selected_names = st.sidebar.multiselect('Select Name(s)', names)

        # Filter the dataframe based on the selected names and months
        filtered_df = df[(df['Name'].isin(selected_names)) & (df['Month'].isin(selected_months))]



        st.subheader('Filtered Data')

        # Plot pie chart for task percentages
        plot_pie_chart(filtered_df)
