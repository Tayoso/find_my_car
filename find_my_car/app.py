"""Streamlit web interface for the car recommendation system."""

import os
from typing import Optional
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from find_my_car.recommender import load_classifier, get_car_recommendation

# Load environment variables
load_dotenv()

def load_csv(file) -> Optional[pd.DataFrame]:
    """Load and validate a CSV file containing car data."""
    try:
        df = pd.read_csv(file)
        required_columns = [
            'make', 'model', 'age', 'body_type', 'fuel_type',
            'transmission_type', 'mileage', 'cost'
        ]
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            st.error(f"Missing required columns: {', '.join(missing_columns)}")
            return None
            
        return df
    except Exception as e:
        st.error(f"Error loading CSV file: {str(e)}")
        return None

def format_car_features(df: pd.DataFrame) -> str:
    """Format car features for display."""
    features = {
        'body_types': df['body_type'].unique().tolist(),
        'fuel_types': df['fuel_type'].unique().tolist(),
        'transmission_types': df['transmission_type'].unique().tolist(),
        'price_range': (float(df['cost'].min()), float(df['cost'].max())),
        'age_range': (int(df['age'].min()), int(df['age'].max())),
        'mileage_range': (float(df['mileage'].min()), float(df['mileage'].max()))
    }
    
    return (
        f"- Body types: {', '.join(features['body_types'])}\n"
        f"- Fuel types: {', '.join(features['fuel_types'])}\n"
        f"- Transmission types: {', '.join(features['transmission_types'])}\n"
        f"- Price range: ${features['price_range'][0]:,.2f} - ${features['price_range'][1]:,.2f}\n"
        f"- Age range: {features['age_range'][0]} - {features['age_range'][1]} years\n"
        f"- Mileage range: {features['mileage_range'][0]:,.0f} - {features['mileage_range'][1]:,.0f} miles"
    )

def main():
    """Main Streamlit application."""
    st.title("🚗 Find My Car Assistant")
    st.markdown("""
    Welcome to the Car Recommendation Assistant! This system uses natural language processing
    to provide intelligent car recommendations based on your requirements.
    """)

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "df" not in st.session_state:
        st.session_state.df = None
    if "classifier" not in st.session_state:
        with st.spinner("Loading model... this may take a few minutes"):
            try:
                st.session_state.classifier = load_classifier()
            except Exception as e:
                st.error(f"Error loading model: {str(e)}")
                st.stop()

    # File upload section
    st.subheader("Upload Car Database")
    uploaded_file = st.file_uploader(
        "Upload your CSV file",
        type="csv",
        help="Upload a CSV file containing car information"
    )

    if uploaded_file is not None:
        df = load_csv(uploaded_file)
        if df is not None:
            st.session_state.df = df
            st.success("CSV file loaded successfully!")
            
            # Display the dataframe
            st.subheader("Current Car Database")
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True
            )

    # Chat interface
    st.subheader("Chat with Car Assistant")

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask about your ideal car..."):
        if st.session_state.df is None:
            st.error("Please upload a car database first!")
        else:
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
                
            # Get and display assistant response
            with st.chat_message("assistant"):
                with st.spinner("Finding the best matches..."):
                    response = get_car_recommendation(
                        st.session_state.classifier,
                        prompt,
                        st.session_state.df
                    )
                    st.markdown(response)
                
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main() 