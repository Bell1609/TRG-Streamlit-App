import os
import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx
from streamlit.source_util import get_pages

def get_current_page_name():
    ctx = get_script_run_ctx()
    if ctx is None:
        raise RuntimeError("Couldn't get script context")

    pages = get_pages("")

    return pages[ctx.page_script_hash]["page_name"]

def make_sidebar():
    with st.sidebar:
        st.title("Functions:")
        st.write("")
        st.write("")

        if st.session_state.get("logged_in", False):
            # Get the list of all Python files in the 'pages' directory
            pages_folder = 'pages'
            page_files = [f for f in os.listdir(pages_folder) if f.endswith('.py')]

            # Loop through each file and add it to the sidebar
            for page_file in page_files:
                # Create human-readable page names
                page_name = page_file.replace('.py', '').replace('_', ' ').capitalize()
                page_path = f"{pages_folder}/{page_file}"

                # Add the page to the sidebar
                st.page_link(page_path, label=page_name, icon="📄")

            st.write("")
            st.write("")

            if st.button("Log out"):
                logout()

        else:
            st.write("Please log in")
            # Redirect to the login page only if the current page is not the main app
            if get_current_page_name() != "landing":
                st.session_state.logged_in = False  # Ensure logged_in is False
                st.switch_page("landing.py")  # Redirect to login

def logout():
    st.session_state.logged_in = False
    st.info("Logged out successfully!")
    st.rerun()  # Use rerun to reset the app after logout
