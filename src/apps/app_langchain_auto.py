#!/usr/bin/env python3
"""
Streamlit App for True LangChain Auto-Managed Vehicle System
LangChain handles everything - conversation, memory, state, flow
"""

import streamlit as st
import pandas as pd
import os
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'core'))
from vehicle_rag_langchain_auto import LangChainVehicleRAG

# Page configuration
st.set_page_config(
    page_title="LangChain Auto Vehicle Recommendation",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    text-align: center;
    color: #1f77b4;
    margin-bottom: 2rem;
}

.chat-message {
    padding: 1rem;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
    border-left: 4px solid;
}

.user-message {
    background-color: #e3f2fd;
    border-left-color: #2196f3;
}

.bot-message {
    background-color: #f3e5f5;
    border-left-color: #9c27b0;
}

.form-container {
    background-color: #f8f9fa;
    padding: 1.5rem;
    border-radius: 0.5rem;
    border: 1px solid #dee2e6;
    margin-bottom: 1rem;
}

.question-form {
    background-color: #fff3cd;
    border: 1px solid #ffeaa7;
    border-radius: 0.5rem;
    padding: 1.5rem;
    margin: 1rem 0;
}

.badge {
    display: inline-block;
    padding: 0.25rem 0.5rem;
    font-size: 0.75rem;
    font-weight: bold;
    border-radius: 0.25rem;
    margin: 0.25rem;
}

.badge-langchain {
    background-color: #ff6b6b;
    color: white;
}

.badge-auto {
    background-color: #4ecdc4;
    color: white;
}

.badge-openai {
    background-color: #45b7d1;
    color: white;
}

.langchain-info {
    background-color: #fff3cd;
    border: 1px solid #ffeaa7;
    border-radius: 0.5rem;
    padding: 1rem;
    margin: 1rem 0;
}

.auto-managed {
    background-color: #d4edda;
    border: 1px solid #c3e6cb;
    border-radius: 0.5rem;
    padding: 1rem;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state for LangChain auto-management."""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'vehicle_rag' not in st.session_state:
        st.session_state.vehicle_rag = None
    if 'conversation_state' not in st.session_state:
        st.session_state.conversation_state = {}
    if 'current_question' not in st.session_state:
        st.session_state.current_question = None
    if 'waiting_for_answer' not in st.session_state:
        st.session_state.waiting_for_answer = False

def check_openai_key():
    """Check if OpenAI API key is available."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.warning("⚠️ No OpenAI API key found. LangChain will use fallback mode.")
    return api_key

def load_vehicle_data(csv_path: str = "src/data/stocked_cars.csv"):
    """Load vehicle data."""
    try:
        df = pd.read_csv(csv_path)
        return df
    except Exception as e:
        st.error(f"Error loading vehicle data: {e}")
        return None

def initialize_langchain_system(csv_path: str = "src/data/stocked_cars.csv"):
    """Initialize the LangChain auto-managed system."""
    if st.session_state.vehicle_rag is None:
        try:
            with st.spinner("🔄 LangChain is setting up auto-management..."):
                st.session_state.vehicle_rag = LangChainVehicleRAG(csv_path=csv_path)
            st.success("✅ LangChain auto-management ready!")
        except Exception as e:
            st.error(f"❌ Error initializing LangChain system: {e}")

def render_question_form(question_data):
    """Render the next question as an interactive form."""
    if not question_data or question_data.get("next_question") == "none":
        return None
    
    question = question_data.get("next_question", "")
    question_type = question_data.get("question_type", "text")
    options = question_data.get("options", [])
    
    st.markdown('<div class="question-form">', unsafe_allow_html=True)
    st.markdown(f"**🤖 LangChain's Next Question:** {question}")
    
    if question_type == "radio" and options:
        selected_option = st.radio(
            "Select your answer:",
            options,
            key="radio_answer"
        )
        return selected_option
    
    elif question_type == "multiselect" and options:
        selected_options = st.multiselect(
            "Select your preferences:",
            options,
            key="multiselect_answer"
        )
        return selected_options
    
    elif question_type == "text":
        text_answer = st.text_input(
            "Your answer:",
            placeholder="Enter your response...",
            key="text_answer"
        )
        return text_answer
    
    st.markdown('</div>', unsafe_allow_html=True)
    return None

def main():
    """Main application function."""
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">🚗 LangChain Auto-Managed Vehicle Recommendation</h1>', unsafe_allow_html=True)
    
    # System info badges
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<div style="text-align: center; margin-bottom: 2rem;">', unsafe_allow_html=True)
        st.markdown('<span class="badge badge-langchain">LangChain Auto-Managed</span>', unsafe_allow_html=True)
        st.markdown('<span class="badge badge-auto">Zero Manual Work</span>', unsafe_allow_html=True)
        if os.getenv("OPENAI_API_KEY"):
            st.markdown('<span class="badge badge-openai">OpenAI GPT-3.5</span>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # LangChain auto-management info
    st.markdown("""
    <div class="auto-managed">
        <strong>🤖 LangChain Auto-Management:</strong>
        <ul>
            <li>✅ Conversation Memory - LangChain remembers everything</li>
            <li>✅ State Management - LangChain tracks conversation state</li>
            <li>✅ Question Flow - LangChain determines next questions</li>
            <li>✅ Skip Logic - LangChain automatically skips answered questions</li>
            <li>✅ Context Building - LangChain maintains conversation context</li>
            <li>✅ Structured Output - LangChain parses responses automatically</li>
        </ul>
        <strong>🎯 Zero Manual Work Required!</strong>
    </div>
    """, unsafe_allow_html=True)
    
    # Vehicle data info
    df = load_vehicle_data("src/data/stocked_cars.csv")
    if df is not None:
        col1, col2, col3, col4 = st.columns(4)
        with col1: st.metric("Total Vehicles", f"{len(df):,}")
        with col2: st.metric("Price Range", f"£{df['PRICE'].min():,} - £{df['PRICE'].max():,}")
        with col3: st.metric("Makes", len(df['MAKE'].unique()))
        with col4: st.metric("Body Types", len(df['BODY_TYPE'].unique()))
    
    # Initialize LangChain system button
    if st.button("🔄 Initialize LangChain Auto-System", type="primary", use_container_width=True):
        initialize_langchain_system("src/data/stocked_cars.csv")
    
    # Main chat interface
    st.markdown("### 💬 Chat with LangChain Auto-Managed Assistant")
    
    # Display chat history
    for message in st.session_state.messages:
        with st.container():
            if message["role"] == "user":
                st.markdown(f"""<div class="chat-message user-message"><strong>You:</strong> {message["content"]}</div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""<div class="chat-message bot-message"><strong>Assistant:</strong> {message["content"]}</div>""", unsafe_allow_html=True)
    
    # Handle current question form
    if st.session_state.current_question and st.session_state.waiting_for_answer:
        st.markdown("### 📝 Answer LangChain's Question")
        answer = render_question_form(st.session_state.current_question)
        
        if answer:
            col1, col2 = st.columns([3, 1])
            with col2:
                if st.button("✅ Submit Answer", type="primary"):
                    # Add user's answer to conversation
                    st.session_state.messages.append({"role": "user", "content": f"Answer: {answer}"})
                    
                    # Process with LangChain
                    with st.spinner("🤖 LangChain is processing your answer..."):
                        result = st.session_state.vehicle_rag.process_with_langchain(f"User answered: {answer}")
                        
                        # Add recommendations
                        st.session_state.messages.append({"role": "assistant", "content": result["recommendations"]})
                        
                        # Update current question
                        if result["next_question"] and result["next_question"] != "none":
                            st.session_state.current_question = result
                            st.session_state.waiting_for_answer = True
                        else:
                            st.session_state.current_question = None
                            st.session_state.waiting_for_answer = False
                            st.session_state.messages.append({"role": "assistant", "content": "✅ All questions answered! Here are your final recommendations."})
                    
                    st.rerun()
    
    # Initial chat input - LangChain handles everything
    if st.session_state.vehicle_rag is not None and not st.session_state.waiting_for_answer:
        with st.form("chat_form", clear_on_submit=True):
            st.markdown('<div class="form-container">', unsafe_allow_html=True)
            user_input = st.text_area(
                "💬 What are you looking for?",
                placeholder="e.g., I need a BMW hatchback under £20,000, Show me electric vehicles for my family...",
                height=100,
                key="user_input"
            )
            submitted = st.form_submit_button("🚗 Ask LangChain", type="primary", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            if submitted and user_input.strip():
                # Add user message
                st.session_state.messages.append({"role": "user", "content": user_input.strip()})
                
                # Let LangChain handle everything automatically
                with st.spinner("🤖 LangChain is processing automatically..."):
                    result = st.session_state.vehicle_rag.process_with_langchain(user_input.strip())
                    
                    # Add recommendations
                    st.session_state.messages.append({"role": "assistant", "content": result["recommendations"]})
                    
                    # Handle next question
                    if result["next_question"] and result["next_question"] != "none":
                        st.session_state.current_question = result
                        st.session_state.waiting_for_answer = True
                    else:
                        st.session_state.messages.append({"role": "assistant", "content": "✅ All questions answered! Here are your final recommendations."})
                    
                    # Show LangChain auto-management info
                    if result["langchain_managed"]:
                        st.success("✅ LangChain auto-managed this conversation!")
                        st.info(f"💬 Summary: {result['conversation_summary']}")
                        if result["skipped_questions"]:
                            st.info(f"⏭️ LangChain automatically skipped: {', '.join(result['skipped_questions'])}")
                    else:
                        st.warning("⚠️ Using fallback mode (LangChain unavailable)")
                
                st.rerun()
    elif st.session_state.vehicle_rag is None:
        st.info("🔄 Please initialize the LangChain auto-system above to start chatting!")
    
    # Clear chat button
    if st.session_state.messages:
        if st.button("🗑️ Start Over", use_container_width=True):
            st.session_state.messages = []
            st.session_state.conversation_state = {}
            st.session_state.current_question = None
            st.session_state.waiting_for_answer = False
            st.rerun()
    
    # LangChain auto-management details
    with st.expander("🤖 How LangChain Auto-Management Works"):
        st.markdown("""
        **LangChain handles everything automatically:**
        
        1. **Conversation Memory**: LangChain remembers all previous interactions
        2. **State Management**: LangChain tracks what questions have been answered
        3. **Question Flow**: LangChain determines the next logical question
        4. **Skip Logic**: LangChain automatically skips already-answered questions
        5. **Context Building**: LangChain maintains conversation context
        6. **Structured Output**: LangChain parses responses into structured data
        
        **Benefits of LangChain Auto-Management:**
        - ✅ Zero manual state tracking
        - ✅ Zero manual skip logic
        - ✅ Zero manual conversation management
        - ✅ Zero manual memory management
        - ✅ Zero manual flow control
        - ✅ Everything handled by LangChain automatically
        
        **What you see is what LangChain manages:**
        - Conversation history
        - Question flow
        - Skip logic
        - Memory
        - State
        - Context
        """)
    
    # Conversation memory info
    if st.session_state.vehicle_rag and hasattr(st.session_state.vehicle_rag, 'conversation_state'):
        with st.expander("🧠 Conversation Memory Status"):
            conversation_state = st.session_state.vehicle_rag.conversation_state
            st.write(f"**Memory Entries:** {len(conversation_state.get('conversation_history', []))}")
            if conversation_state.get('conversation_history'):
                st.write("**Recent Memory:**")
                for i, entry in enumerate(conversation_state['conversation_history'][-3:], 1):
                    st.write(f"{i}. {entry['input'][:100]}...")
            
            st.write("**Current State:**")
            for key, value in conversation_state.items():
                if key != 'conversation_history':
                    st.write(f"- {key}: {value}")

if __name__ == "__main__":
    main() 