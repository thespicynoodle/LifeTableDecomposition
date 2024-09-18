import streamlit as st
from datetime import datetime
import time

def main():
  st.set_page_config(
    page_title="LifeTable Exploration Tool",
    page_icon=":world_map:",  # You can change this to any emoji or image URL
    layout="wide"  # Optional: makes the app use the full width of the browser window
)  
  
  # Create space in the sidebar to push the time to the bottom
  st.sidebar.write("\n" * 10)  # Adjust the number of newlines to push the content down
    # Main content of the app
  st.title("LifeTable Exploration Tool")
  st.write("Use the navigation on the left to browse different pages.")
if __name__ == "__main__":
    main()
