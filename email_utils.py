import os
import smtplib
from email.mime.text import MIMEText
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Load SMTP configuration from environment variables
SMTP_SERVER = os.getenv("SMTP_SERVER")
SMTP_PORT = int(os.getenv("SMTP_PORT", 465))  # Ensure port is an integer
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")

def send_email(to_email, subject, body):
    """Sends an email with the given subject and body to the specified email address."""
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = SMTP_USER
    msg["To"] = to_email

    try:
        st.info("Connecting to the SMTP server...")
        # Use SMTP_SSL for direct SSL connection on port 465
        with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT) as server:
            st.info("Logging into the SMTP server...")
            server.login(SMTP_USER, SMTP_PASSWORD)
            
            st.info("Sending the email...")
            server.sendmail(SMTP_USER, to_email, msg.as_string())
            
            st.success(f"Email sent to {to_email}.")  # Display success message
            return True
    except smtplib.SMTPException as e:
        st.error(f"SMTP error: {e}")
    except Exception as e:
        st.error(f"Failed to send email: {e}")
    
    return False  # Return False if email was not sent

def send_verification_email(to_email, code):
    """Sends a verification email with the activation code."""
    subject = "CS Data Analysis Application - Activate Your Account"
    body = f"Thank you for registering! Please use the following code to activate your account: {code}"
    return send_email(to_email, subject, body)

def send_password_reset_email(to_email, token):
    """Sends a password reset email with a reset token."""
    subject = "CS Data Analysis Application - Your Password Reset Token"
    body = f"Use this code to reset your password: {token}"
    return send_email(to_email, subject, body)
