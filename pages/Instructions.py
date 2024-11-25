from __future__ import division
import streamlit as st
from authentication import make_sidebar

make_sidebar()

st.title("Instructions")
st.sidebar.success('Select the ticket data or sales data')

# HTML content as a multi-line string
html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Instruction</title>
</head>
<body>
    <h3>Relevant resources **<strong>MUST KNOW</strong>**</h3>
        <ul>
            <li>Sales and tickets data segmentation uses RFM (Recency, Frequency, Monetary) analysis. Therefore,
                anyone who wants to learn more
                about RFM analysis can visit <a href="https://www.putler.com/rfm-analysis/?ref=200lab.io">this
                    website</a></li>
            <li>The clustering method used for this app is K-Means clustering method. For more information, visit <a
                    href="https://neptune.ai/blog/k-means-clustering">this website</a></li>
        </ul>
    <h3>What is this application doing </h3>
    <p>This application is built for data insights and data segmentation purpose. It will give you underline filtered data rows, visualization of filtered data for some important key metrics, basic statistics information, such as:</p>
    <ul>
        <li><strong>Count:</strong> Number of non-null entries </li>
        <li><strong>Mean:</strong> Average of the column.</li>
        <li><strong>Standard Deviation (std):</strong> Measure of spread of the data.</li>
        <li><strong>Min/Max:</strong> Minimum and maximum values.</li>
        <li><strong>Percentiles (25%, 50%, 75%):</strong> Quartiles, which describe the distribution.</li>
    </ul>
    <h3>How to Use</h3>
    <p>The application is user-friendly and simple to navigate. Follow these steps to get started:</p>
    <ul>
        <li><strong>Select the report page:</strong> Select the report page you need on the sidebar, e.g Sales Data Insights, Tickets Data Insights,...</li>
        <li><strong>Load the required data:</strong> Upload your data files from Freshsales or Freshdesk, making sure it includes all necessary fields.</li>
        <li><strong>Select filters:</strong> Use the filter options in the sidebar to customize your report or chart based on specific criteria. You can easily drill down into the data or narrow it down to meet your specific needs.</li>
        <li><strong>Generate reports and charts:</strong> Click the relevant buttons to create the reports or charts you need, which will be displayed in the main screen area.</li>
    </ul>
    <h3>Data Source</h3>
    <p>The data source requirements vary based on the type of report you are generating:</p>
    <ul>
        <li><strong>Sales Reports:</strong> Requires Deals and/or Accounts data from Freshsales, including all fields available in Freshsales.</li>
        <li><strong>Ticket Reports:</strong> Requires Ticket data from Freshdesk, with all relevant fields included.</li>
        <li><strong>Capacity Reports:</strong> Requires booking data, which should be in Excel format with separate columns for "Month", "Name", and "Task Type".</li>
    </ul>
</body>
</html>
"""

# Display the HTML content
st.markdown(html_content, unsafe_allow_html=True)
