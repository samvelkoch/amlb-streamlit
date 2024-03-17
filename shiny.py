import requests
import streamlit as st


def display_shiny_app():
    """Add components to communicate with and display the results of the Shiny app"""
    frameworks = st.session_state.raw_data.framework.unique()
    with st.sidebar:
        selected_frameworks = st.multiselect(
            label="Choose Frameworks",
            options=frameworks,
            default=frameworks
        )
        col1, col2 = st.columns(2)
        with col1:
            input_min = st.slider(
                label="Minimum",
                min_value=0,
                max_value=100,
                value=0
            )
        with col2:
            input_max = st.slider(
                label="Maximum",
                min_value=0,
                max_value=100,
                value=100
            )

    # Define the REST API URL
    api_url = "http://backend:8080/api/bt_tree"  # Use the service name as the hostname
    test_url = "http://backend:8080/api/test"  # Use the service name as the hostname

    # payload
    payload = {
        "text": "Hello World!",
        "min": input_min,
        "max": input_max
    }

    # Make a request to the REST API
    response = requests.post(test_url, data=payload)

    # Check if the request was successful
    if response.status_code == 200:
        # Display the PNG image from the API response
        st.image(response.content)
    else:
        st.error("Failed to fetch image from the REST API")
