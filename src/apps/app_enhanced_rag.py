#!/usr/bin/env python3
"""
Enhanced Vehicle Recommendation App with Semantic Search and Reasoning Agents
Powered by Llama 3 with advanced RAG capabilities
"""

import streamlit as st
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from core.vehicle_rag_enhanced import EnhancedVehicleRAG
import time
import json

# Page configuration
st.set_page_config(
    page_title="Enhanced Vehicle Finder - Llama 3 RAG",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .feature-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 4px solid #1f77b4;
    }
    .metric-box {
        background-color: #e8f4fd;
        padding: 0.5rem;
        border-radius: 0.3rem;
        margin: 0.5rem 0;
        text-align: center;
    }
    .recommendation-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border: 1px solid #dee2e6;
    }
    .status-success {
        color: #28a745;
        font-weight: bold;
    }
    .status-warning {
        color: #ffc107;
        font-weight: bold;
    }
    .status-error {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

def initialize_rag_system():
    """Initialize the enhanced RAG system."""
    if 'enhanced_rag' not in st.session_state:
        with st.spinner("🔄 Initializing Enhanced RAG System..."):
            try:
                st.session_state.enhanced_rag = EnhancedVehicleRAG("src/data/stocked_cars.csv")
                st.success("✅ Enhanced RAG System Ready!")
                return True
            except Exception as e:
                st.error(f"❌ Failed to initialize RAG system: {e}")
                return False
    return True

def display_system_status():
    """Display the status of enhanced features."""
    rag = st.session_state.enhanced_rag
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        status_class = "status-success" if rag.semantic_search_ready else "status-warning"
        st.markdown(f'<div class="metric-box"><span class="{status_class}">🔍 Semantic Search</span></div>', unsafe_allow_html=True)
    
    with col2:
        status_class = "status-success" if rag.reasoning_ready else "status-warning"
        st.markdown(f'<div class="metric-box"><span class="{status_class}">🤖 Reasoning Agent</span></div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown(f'<div class="metric-box">📊 {len(rag.documents)} Vehicles</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown(f'<div class="metric-box">⚡ Enhanced RAG</div>', unsafe_allow_html=True)

def display_conversation_summary():
    """Display current conversation summary."""
    if 'conversation_history' in st.session_state and st.session_state.conversation_history:
        st.markdown("### 📋 Conversation Summary")
        
        summary = st.session_state.enhanced_rag._generate_conversation_summary()
        if summary != "No preferences set yet":
            st.info(f"**Current Preferences:** {summary}")
        else:
            st.info("No preferences set yet. Start by telling us what you're looking for!")

def display_metrics(result):
    """Display performance and search metrics."""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Response Time", f"{result['response_time']:.3f}s")
    
    with col2:
        st.metric("Semantic Matches", result['semantic_matches'])
    
    with col3:
        st.metric("Pattern Matches", result['pattern_matches'])
    
    with col4:
        st.metric("Combined Results", result['combined_matches'])

def display_recommendations(result):
    """Display vehicle recommendations."""
    st.markdown("### 🚗 Recommendations")
    
    with st.container():
        st.markdown(f"""
        <div class="recommendation-box">
            {result['recommendations']}
        </div>
        """, unsafe_allow_html=True)

def display_next_question(result):
    """Display the next question if available."""
    if result['next_question']:
        st.markdown("### ❓ Next Question")
        
        question = result['next_question']
        
        if question['type'] == 'radio':
            selected = st.radio(
                question['question'],
                options=question['options'],
                key=f"question_{len(st.session_state.conversation_history)}"
            )
            if st.button("Submit Answer", key=f"submit_{len(st.session_state.conversation_history)}"):
                process_user_input(selected)
        
        elif question['type'] == 'multiselect':
            selected = st.multiselect(
                question['question'],
                options=question['options'],
                key=f"question_{len(st.session_state.conversation_history)}"
            )
            if st.button("Submit Answer", key=f"submit_{len(st.session_state.conversation_history)}"):
                if selected:
                    process_user_input(", ".join(selected))
        
        elif question['type'] == 'text':
            user_input = st.text_input(
                question['question'],
                key=f"question_{len(st.session_state.conversation_history)}"
            )
            if st.button("Submit Answer", key=f"submit_{len(st.session_state.conversation_history)}"):
                if user_input:
                    process_user_input(user_input)

def process_user_input(user_input):
    """Process user input with enhanced RAG."""
    if not user_input.strip():
        return
    
    # Add to conversation history
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    
    st.session_state.conversation_history.append({
        "role": "user",
        "content": user_input,
        "timestamp": time.time()
    })
    
    # Process with enhanced RAG
    with st.spinner("🔍 Searching with Enhanced RAG..."):
        result = st.session_state.enhanced_rag.process_with_enhanced_rag(user_input)
    
    # Add assistant response to history
    st.session_state.conversation_history.append({
        "role": "assistant",
        "content": result['recommendations'],
        "metrics": {
            "response_time": result['response_time'],
            "semantic_matches": result['semantic_matches'],
            "pattern_matches": result['pattern_matches'],
            "combined_matches": result['combined_matches'],
            "reasoning_used": result['reasoning_used']
        },
        "timestamp": time.time()
    })
    
    # Rerun to update the display
    st.rerun()

def display_conversation_history():
    """Display the conversation history."""
    if 'conversation_history' in st.session_state and st.session_state.conversation_history:
        st.markdown("### 💬 Conversation History")
        
        for i, message in enumerate(st.session_state.conversation_history):
            if message['role'] == 'user':
                st.markdown(f"**You:** {message['content']}")
            else:
                st.markdown(f"**Assistant:** {message['content']}")
                
                # Show metrics for assistant messages
                if 'metrics' in message:
                    metrics = message['metrics']
                    with st.expander(f"📊 Response Metrics (Response #{i//2 + 1})"):
                        col1, col2, col3, col4, col5 = st.columns(5)
                        with col1:
                            st.metric("Time", f"{metrics['response_time']:.3f}s")
                        with col2:
                            st.metric("Semantic", metrics['semantic_matches'])
                        with col3:
                            st.metric("Pattern", metrics['pattern_matches'])
                        with col4:
                            st.metric("Combined", metrics['combined_matches'])
                        with col5:
                            status = "✅" if metrics['reasoning_used'] else "❌"
                            st.metric("Reasoning", status)
            
            st.divider()

def main():
    """Main application function."""
    st.markdown('<h1 class="main-header">🚗 Enhanced Vehicle Finder</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Powered by Llama 3 with Semantic Search & Reasoning Agents</p>', unsafe_allow_html=True)
    
    # Initialize the enhanced RAG system
    if not initialize_rag_system():
        st.error("Failed to initialize the system. Please check your setup.")
        return
    
    # Display system status
    display_system_status()
    
    # Sidebar for features
    with st.sidebar:
        st.markdown("### 🎯 Features")
        st.markdown("""
        - **🔍 Semantic Search**: Understands natural language queries
        - **🤖 Reasoning Agents**: Llama 3 powered analysis
        - **📊 Pattern Matching**: Fast rule-based filtering
        - **🔗 Combined Results**: Intelligent result merging
        - **💬 Conversation Memory**: Remembers your preferences
        """)
        
        st.markdown("### 📝 Example Queries")
        st.markdown("""
        - "I want a BMW below £20000"
        - "Show me electric vehicles for my family"
        - "i need a new diesel hatchback less than 17000"
        - "family car with good safety features"
        - "luxury SUV for business use"
        """)
        
        # Clear conversation button
        if st.button("🗑️ Clear Conversation"):
            if 'conversation_history' in st.session_state:
                del st.session_state.conversation_history
            st.session_state.enhanced_rag.conversation_state = {
                "passenger_count": None,
                "body_type": None,
                "price_range": None,
                "make_pref": None,
                "fuel_pref": None,
                "answered_questions": set(),
                "conversation_history": [],
                "reasoning_context": {},
                "semantic_matches": []
            }
            st.rerun()
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # User input
        st.markdown("### 💬 Tell us what you're looking for")
        
        user_input = st.text_area(
            "Describe your ideal vehicle:",
            placeholder="e.g., I want a BMW SUV under £25000 for my family",
            height=100
        )
        
        col1_1, col1_2 = st.columns([1, 1])
        with col1_1:
            if st.button("🚗 Find Vehicles", type="primary"):
                if user_input.strip():
                    process_user_input(user_input)
        
        with col1_2:
            if st.button("🎲 Try Example"):
                examples = [
                    "I want a BMW below £20000",
                    "Show me electric vehicles for my family",
                    "i need a new diesel hatchback less than 17000"
                ]
                import random
                st.session_state.example_query = random.choice(examples)
                st.rerun()
        
        # Display example query if set
        if 'example_query' in st.session_state:
            st.info(f"💡 Try this example: {st.session_state.example_query}")
            if st.button("Use This Example"):
                process_user_input(st.session_state.example_query)
                del st.session_state.example_query
                st.rerun()
        
        # Display conversation summary
        display_conversation_summary()
        
        # Display conversation history
        display_conversation_history()
    
    with col2:
        # Display current system status
        st.markdown("### 🔧 System Status")
        
        rag = st.session_state.enhanced_rag
        
        # Enhanced features status
        st.markdown("#### Enhanced Features")
        
        semantic_status = "✅ Ready" if rag.semantic_search_ready else "⚠️ Unavailable"
        reasoning_status = "✅ Ready" if rag.reasoning_ready else "⚠️ Unavailable"
        
        st.markdown(f"""
        - **Semantic Search**: {semantic_status}
        - **Reasoning Agent**: {reasoning_status}
        - **Pattern Matching**: ✅ Always Available
        - **Combined Results**: ✅ Active
        """)
        
        # Show recent metrics if available
        if 'conversation_history' in st.session_state and st.session_state.conversation_history:
            latest_response = None
            for message in reversed(st.session_state.conversation_history):
                if message['role'] == 'assistant' and 'metrics' in message:
                    latest_response = message['metrics']
                    break
            
            if latest_response:
                st.markdown("#### 📊 Latest Response Metrics")
                st.metric("Response Time", f"{latest_response['response_time']:.3f}s")
                st.metric("Semantic Matches", latest_response['semantic_matches'])
                st.metric("Pattern Matches", latest_response['pattern_matches'])
                st.metric("Combined Results", latest_response['combined_matches'])
                st.metric("Reasoning Used", "✅" if latest_response['reasoning_used'] else "❌")
    
    # Display results if available
    if 'conversation_history' in st.session_state and st.session_state.conversation_history:
        latest_message = st.session_state.conversation_history[-1]
        if latest_message['role'] == 'assistant':
            st.markdown("---")
            st.markdown("### 🎯 Latest Results")
            
            # Display metrics
            if 'metrics' in latest_message:
                display_metrics(latest_message['metrics'])
            
            # Display recommendations
            display_recommendations({
                'recommendations': latest_message['content'],
                'next_question': None  # We'll handle this separately
            })
            
            # Display next question if available
            # (This would need to be stored in the conversation history)

if __name__ == "__main__":
    main() 