import os
import re
import time
import streamlit as st
import yaml
from pathlib import Path
import streamlit_authenticator as stauth
from streamlit.runtime.scriptrunner import get_script_run_ctx
from streamlit.source_util import get_pages

def get_current_page_name():
    ctx = get_script_run_ctx()
    if ctx is None:
        raise RuntimeError("Couldn't get script context")

    pages = get_pages("")

    return pages[ctx.page_script_hash]["page_name"]

# Helper functions----
# Load user credentials from a YAML file
def load_credentials():
    config_file = Path(__file__).parent / "config/config.yaml"
    with open(config_file) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    return config

# Save updated credentials
def save_credentials(config):
    config_file = Path(__file__).parent / "config/config.yaml"
    with open(config_file, 'w') as file:
        yaml.dump(config, file)

# Validate email
def is_valid_email(email):
    # Simple regex pattern for validating email format
    email_pattern = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
    return re.match(email_pattern, email)

# validate username
def is_valid_username(username):
    # Username must be alphanumeric and between 3-15 characters long
    username_pattern = r"^[a-zA-Z0-9]{3,15}$"
    return re.match(username_pattern, username)

def make_sidebar():
    with st.sidebar:
        st.write("")

        if st.session_state.get("authentication_status"):
            # Get the list of all Python files in the 'pages' directory
            pages_folder = 'pages'
            page_files = [f for f in os.listdir(pages_folder) if f.endswith('.py')]

            # Always show 'Instructions' page first
            st.page_link("pages/Instructions.py", label="Instructions", icon="📖")

            # Loop through each file and add it to the sidebar
            for page_file in page_files:
                if page_file != "Instructions.py":  # Skip the Instructions page
                    # populate page names
                    page_name = page_file.replace('.py', '').replace('_', ' ').capitalize()
                    page_path = f"{pages_folder}/{page_file}"

                    # Add the page to the sidebar
                    st.page_link(page_path, label=page_name, icon="📄")

            st.write("")
            # Logout button
            if st.button("Log out", key="logout_button"):
                logout()

        else:
            st.write("Please log in")
            # Redirect to the login page only if the current page is not the landing
            if get_current_page_name() != "landing":
                st.session_state.logged_in = False  # Ensure logged_in is False
                st.switch_page("landing.py")  # Redirect to login

def logout():
    st.session_state['authentication_status'] = None  # Set authentication status to False
    st.session_state.logged_in = False  # Set logged in state to False
    st.info("Logged out successfully!")
    st.rerun()  # Refresh the app to apply changes

# Register a new user with validation
# Register a new user with validation and st.form
def register_user():
    config = load_credentials()

    st.write("Register a new account")
    
    # Group inputs into a form
    with st.form("register_form", clear_on_submit=True):
        new_username = st.text_input("Enter your username")
        new_email = st.text_input("Enter your email")
        new_password = st.text_input("Enter your password", type="password")
        confirm_password = st.text_input("Confirm your password", type="password")

        # Submit button for the form
        submit_button = st.form_submit_button("Submit")

        if submit_button:
            # Flag to control submission
            is_form_valid = True

            # Validate username
            if new_username:
                if not is_valid_username(new_username):
                    st.error("Username must be alphanumeric and between 3-15 characters.")
                    is_form_valid = False
            else:
                is_form_valid = False
                st.error("Username is required")

            # Validate email
            if new_email:
                if not is_valid_email(new_email):
                    st.error("Please enter a valid email address.")
                    is_form_valid = False
                else:
                    authorized_emails = config.get("authorized_emails", [])
                    allowed_domains = config.get("allowed_domains", [])
                    email_domain = new_email.split('@')[-1]

                    if new_email not in authorized_emails and email_domain not in allowed_domains:
                        st.error("This email is not authorized for registration.")
                        is_form_valid = False
            else:
                is_form_valid = False
                st.error("Email is required")

            # Validate password
            if new_password:
                if new_password != confirm_password:
                    st.error("Passwords do not match.")
                    is_form_valid = False
            else:
                is_form_valid = False
                st.error("Password is required")

            # Only submit when form is valid
            if is_form_valid:
                config['credentials']['usernames'][new_username] = {
                    'name': new_username,  # Store the username as the name
                    'email': new_email,    # Store email separately
                    'password': new_password  # Store the hashed password
                }

                # Now hash the new credentials using stauth.Hasher
                stauth.Hasher.hash_passwords(config['credentials'])

                # Save the updated credentials with hashed passwords
                save_credentials(config)

                st.success("Registration successful! You can now log in.")

                # Redirect to login page
                time.sleep(1)

                st.session_state['form'] = 'login'  # Switch to login form
                st.rerun()  # Force app to rerun and redirect to login

def reset_password():
    config = load_credentials()

    st.write("Reset your password")

    with st.form("reset_password_form", clear_on_submit=True):
        username = st.text_input("Enter your username")
        email = st.text_input("Enter your email")

        new_password = st.text_input("Enter your new password", type="password")
        confirm_password = st.text_input("Confirm your new password", type="password")

        submit_button = st.form_submit_button("Submit")

        if submit_button:
            # Check if the username exists in the credentials
            if username not in config['credentials']['usernames']:
                st.error("Username not found. Please check and try again.")
            else:
                # Validate that the email matches the username
                stored_email = config['credentials']['usernames'][username].get('email')
                if stored_email != email:
                    st.error("The email you entered does not match the one on file for this username.")
                elif new_password != confirm_password:
                    st.error("Passwords do not match.")
                else:
                    # Create a credentials dictionary for hashing the new password
                    credentials_to_hash = {
                        'usernames': {
                            username: {
                                'name': config['credentials']['usernames'][username]['name'],
                                'password': new_password  # Store the plain-text password temporarily
                            }
                        }
                    }

                    # Hash the new password
                    hashed_credentials = stauth.Hasher.hash_passwords(credentials_to_hash)

                    # Update the user's password in the config
                    config['credentials']['usernames'][username]['password'] = hashed_credentials['usernames'][username]['password']

                    # Save the updated credentials to the YAML file
                    save_credentials(config)

                    st.success("Password reset successfully! You can now log in with your new password.")
                    # Redirect to login after a short pause
                    st.session_state['form'] = 'login'
                    st.rerun()