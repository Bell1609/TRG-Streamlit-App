import os
import re
import time
import random
import string
from datetime import datetime, timedelta
import streamlit as st
import yaml
from pathlib import Path
import streamlit_authenticator as stauth
from streamlit.runtime.scriptrunner import get_script_run_ctx
from streamlit.source_util import get_pages
from email_utils import send_verification_email, send_password_reset_email

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

def generate_reset_token(length=6):
    """Generates a random alphanumeric token for password reset."""
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

def is_valid_email_domain(email, allowed_domains):
    """Check if the email's domain is in the list of allowed domains."""
    domain = email.split("@")[-1]
    return domain in allowed_domains

# Dictionary to store activation codes temporarily
activation_codes = {}

def generate_activation_code(length=6):
    """Generates a random alphanumeric activation code."""
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

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

# Register a new user with validation and st.form
def register_user():
    config = load_credentials()
    allowed_domains = config.get("allowed_domains", [])  # Get allowed domains from config

    st.write("Register a new account")

    with st.form("register_form", clear_on_submit=False):
        new_username = st.text_input("Enter your username")
        new_email = st.text_input("Enter your email")
        new_password = st.text_input("Enter your password", type="password")
        confirm_password = st.text_input("Confirm your password", type="password")
        
        submit_button = st.form_submit_button("Submit")
        
        if submit_button:
            is_form_valid = True

            # Input validation
            if new_username and not is_valid_username(new_username):
                st.error("Username must be alphanumeric and between 3-15 characters.")
                is_form_valid = False
            if new_email and not is_valid_email(new_email):
                st.error("Please enter a valid email address.")
                is_form_valid = False
            elif not is_valid_email_domain(new_email, allowed_domains):  # Check allowed domain
                st.error(f"Email domain not allowed. Please use an email from these domains: {', '.join(allowed_domains)}")
                is_form_valid = False
            if new_password and new_password != confirm_password:
                st.error("Passwords do not match.")
                is_form_valid = False
            
            if is_form_valid:
                # Generate activation code and expiration time
                code = generate_activation_code()
                expiration_time = (datetime.now() + timedelta(hours=1)).isoformat()

                # Attempt to send the verification email
                if send_verification_email(new_email, code):
                    st.success("Registration successful! Please check your email for a verification link.")

                    # Save credentials only if email sent successfully
                    config['credentials']['usernames'][new_username] = {
                        'name': new_username,
                        'email': new_email,
                        'password': new_password,  # Hashed password should be used here
                        'verified': False,
                        'activation_code': code,
                        'code_expiry': expiration_time
                    }
                    stauth.Hasher.hash_passwords(config['credentials'])
                    save_credentials(config)

                    # Redirect to login page
                    st.session_state['form'] = 'login'
                    st.rerun()
                else:
                    st.error("Failed to send verification email. Please try again.")

    # Retry button outside the form
    if st.session_state.get('form') == 'register':
        st.button("Retry", on_click=lambda: st.session_state.update({'form': 'register'}))

def verify_user(email, code):
    config = load_credentials()
    username = None

    # Locate the username associated with the email
    for user, details in config['credentials']['usernames'].items():
        if details.get('email') == email:
            username = user
            break

    if username:
        user_data = config['credentials']['usernames'][username]
        
        # Debugging output to verify values
        st.write("Stored activation code:", user_data.get("activation_code"))
        st.write("Stored code expiry:", user_data.get("code_expiry"))

        # Verify the activation code and expiration
        stored_code = user_data.get("activation_code")
        code_expiry = user_data.get("code_expiry")

        # Check activation code validity
        if stored_code == code:
            # Ensure code_expiry is a string
            if isinstance(code_expiry, str):
                try:
                    # Attempt to parse `code_expiry`
                    expiry_datetime = datetime.fromisoformat(code_expiry)
                    st.write("Parsed expiry datetime:", expiry_datetime)

                    # Check if code has expired
                    if expiry_datetime >= datetime.now():
                        # Mark user as verified
                        user_data['verified'] = True
                        # Remove activation code and expiry
                        user_data.pop('activation_code', None)
                        user_data.pop('code_expiry', None)
                        save_credentials(config)
                        return True
                    else:
                        st.error("The activation code has expired.")
                except ValueError:
                    st.error("Activation code expiry format is invalid.")
                    st.write("Value of code_expiry that caused error:", code_expiry)
            else:
                st.error("Activation code expiry is missing or not in the correct format.")
        else:
            st.error("Invalid activation code.")
    else:
        st.error("Activation request not found.")
    return False

def reset_password():
    config = load_credentials()

    st.write("Reset your password")

    if "reset_step" not in st.session_state:
        st.session_state["reset_step"] = 1

    if st.session_state["reset_step"] == 1:
        with st.form("email_verification_form", clear_on_submit=True):
            username = st.text_input("Enter your username")
            email = st.text_input("Enter your email")
            submit_button = st.form_submit_button("Send Reset Code")

            if submit_button:
                if username not in config['credentials']['usernames']:
                    st.error("Username not found.")
                else:
                    user_data = config['credentials']['usernames'][username]
                    if not user_data.get('verified', False):
                        st.error("This account is not verified. Please verify your account before resetting the password.")
                    elif user_data.get('email') != email:
                        st.error("The email you entered does not match our records.")
                    else:
                        # Generate and store reset token
                        token = generate_reset_token()
                        expiration_time = (datetime.now() + timedelta(minutes=10)).isoformat()
                        user_data["reset_token"] = token
                        user_data["token_expiry"] = expiration_time
                        save_credentials(config)  # Update the config with the token

                        # Send token to email
                        send_password_reset_email(email, token)
                        st.success("A reset code has been sent to your email.")
                        st.session_state["reset_step"] = 2  # Move to the next step