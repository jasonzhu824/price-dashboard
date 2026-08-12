"""Streamlit Community Cloud entry point for the IQVIA Price Dashboard.

Community Cloud uses streamlit_app.py in the repository root as the
default main file, so deploying a new app requires no extra configuration.
"""

from src.generate_iqvia_price_dashboard import build_dashboard

if __name__ == "__main__":
    build_dashboard()
