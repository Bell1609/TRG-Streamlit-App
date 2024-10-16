from __future__ import division
import streamlit as st
from authentication import make_sidebar

make_sidebar()

st.title("Instructions")
st.sidebar.success('Select the ticket data or sales data')

# Read the HTML file content
with open("instructions.html", 'r') as file:
    html_content = file.read()

# Display the HTML content
st.markdown(html_content, unsafe_allow_html=True)