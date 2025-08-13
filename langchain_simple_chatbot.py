import os
import streamlit as st
from dotenv import load_dotenv
import pandas as pd
from typing import List, Dict, Any

# Load environment variables
load_dotenv()

# LangChain imports (simpler version)
from langchain_anthropic import ChatAnthropic
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate

# Import existing utilities
from src.utils import load_car_data, get_car_context, is_car_query, select_diverse_cars
from src.prompts import get_car_sales_system_prompt

# ----------------------
# LangChain Setup
# ----------------------

def setup_langchain_components():
    """Setup LangChain components for enhanced functionality"""
    
    # Initialize the LLM
    llm = ChatAnthropic(
        model="claude-3-5-sonnet-20241022",
        temperature=0.1,
        max_tokens=512,
        api_key=os.getenv("ANTHROPIC_API_KEY")
    )
    
    # Setup conversation memory (remembers last 10 exchanges)
    memory = ConversationBufferWindowMemory(
        k=10,
        return_messages=True,
        memory_key="history"
    )
    
    # Create conversation chain
    conversation = ConversationChain(
        llm=llm,
        memory=memory,
        verbose=False
    )
    
    return llm, memory, conversation

def create_enhanced_car_context(query: str, car_df, memory):
    """Create enhanced context using conversation history"""
    
    # Get basic car context
    car_context = get_car_context(query, car_df)
    
    # Get conversation history for context
    history_context = ""
    if memory and memory.chat_memory.messages:
        history_context = "\n\nCONVERSATION HISTORY:\n"
        for i, msg in enumerate(memory.chat_memory.messages[-6:]):  # Last 6 messages
            role = "User" if isinstance(msg, HumanMessage) else "Assistant"
            history_context += f"{role}: {msg.content}\n"
    
    return car_context, history_context

def enhanced_car_query_with_memory(query: str, car_df, llm, memory):
    """Enhanced car query using LangChain memory and conversation history"""
    
    # Get enhanced context
    car_context, history_context = create_enhanced_car_context(query, car_df, memory)
    
    # Create enhanced system prompt
    base_system = get_car_sales_system_prompt("")  # We'll add car data separately
    
    enhanced_system = f"""
    {base_system}
    
    {history_context}
    
    CAR INVENTORY CONTEXT:
    {car_context if car_context else "No specific cars found matching criteria."}
    
    INSTRUCTIONS:
    1. Consider the conversation history when making recommendations
    2. If the user has mentioned preferences before, factor them in
    3. Provide personalized recommendations based on their history
    4. If they've asked about similar cars before, suggest alternatives
    5. Always maintain the 3-recommendation format when possible
    """
    
    # Get response from LLM
    response = llm.invoke([
        SystemMessage(content=enhanced_system),
        HumanMessage(content=query)
    ])
    
    return response.content

def create_conversation_chain_with_car_data(car_df, llm, memory):
    """Create a conversation chain specifically for car sales"""
    
    # Create a custom prompt template
    template = """
    You are an intelligent car sales assistant with access to our car inventory and conversation history.
    
    Previous conversation:
    {history}
    
    Human: {input}
    
    Available car data summary:
    - Total cars: {total_cars}
    - Price range: £{min_price:,} - £{max_price:,}
    - Makes available: {makes}
    - Body types: {body_types}
    
    Instructions:
    1. Use conversation history to provide personalized recommendations
    2. Consider previous preferences and constraints mentioned
    3. Provide exactly 3 diverse recommendations when possible
    4. Include VRM, make, model, price, and desirability score
    5. If no cars match, suggest alternatives or ask for clarification
    6. Be conversational and helpful
    
    Assistant:"""
    
    # Get car data stats
    if car_df is not None:
        total_cars = len(car_df)
        min_price = car_df['PRICE'].min()
        max_price = car_df['PRICE'].max()
        makes = ', '.join(car_df['MAKE'].unique()[:10])  # First 10 makes
        body_types = ', '.join(car_df['BODY_TYPE'].unique())
    else:
        total_cars = 0
        min_price = max_price = 0
        makes = body_types = "None"
    
    prompt = PromptTemplate(
        input_variables=["history", "input", "total_cars", "min_price", "max_price", "makes", "body_types"],
        template=template
    )
    
    # Create conversation chain
    conversation = ConversationChain(
        llm=llm,
        memory=memory,
        prompt=prompt,
        verbose=False
    )
    
    return conversation

# ----------------------
# Streamlit App
# ----------------------

def main():
    st.set_page_config(
        page_title="LangChain Car Sales Assistant", 
        page_icon="🚗", 
        layout="centered"
    )
    
    st.title("🚗 LangChain Car Sales Assistant")
    st.caption("Enhanced with Memory, Conversation History & Smart Context")
    
    # Load car data
    car_data_summary, car_df = load_car_data()
    
    # Initialize LangChain components
    if 'langchain_initialized' not in st.session_state:
        with st.spinner("Initializing LangChain components..."):
            try:
                llm, memory, conversation = setup_langchain_components()
                enhanced_conversation = create_conversation_chain_with_car_data(car_df, llm, memory)
                
                st.session_state.llm = llm
                st.session_state.memory = memory
                st.session_state.conversation = conversation
                st.session_state.enhanced_conversation = enhanced_conversation
                st.session_state.langchain_initialized = True
                st.success("✅ LangChain components initialized!")
            except Exception as e:
                st.error(f"Failed to initialize LangChain: {e}")
                return
    
    # Sidebar
    with st.sidebar:
        st.subheader("LangChain Features")
        
        # Feature toggles
        use_memory = st.checkbox("Use Conversation Memory", value=True, help="Remember previous conversations")
        use_enhanced_context = st.checkbox("Use Enhanced Context", value=True, help="Include conversation history in responses")
        use_conversation_chain = st.checkbox("Use Conversation Chain", value=False, help="Use LangChain's conversation chain")
        
        st.markdown("---")
        st.subheader("Memory Status")
        if use_memory and st.session_state.memory:
            memory_length = len(st.session_state.memory.chat_memory.messages)
            st.info(f"Remembering {memory_length} messages")
            
            # Show recent conversation
            if memory_length > 0:
                st.subheader("Recent Conversation")
                for i, msg in enumerate(st.session_state.memory.chat_memory.messages[-4:]):
                    role = "👤" if isinstance(msg, HumanMessage) else "🤖"
                    st.text(f"{role} {msg.content[:50]}...")
        
        st.markdown("---")
        st.subheader("Car Data")
        if car_df is not None:
            st.success(f"✅ {len(car_df)} cars loaded")
            st.info(f"Price range: £{car_df['PRICE'].min():,} - £{car_df['PRICE'].max():,}")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I'm your LangChain-enhanced car sales assistant. I can remember our conversations and provide personalized recommendations! 🚗"}
        ]
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask me about cars..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking with LangChain..."):
                try:
                    if use_conversation_chain and st.session_state.enhanced_conversation:
                        # Use LangChain conversation chain
                        response = st.session_state.enhanced_conversation.predict(input=prompt)
                    elif use_enhanced_context and st.session_state.memory:
                        # Use enhanced context with memory
                        response = enhanced_car_query_with_memory(
                            prompt, 
                            car_df, 
                            st.session_state.llm, 
                            st.session_state.memory
                        )
                    else:
                        # Use original method
                        if is_car_query(prompt) and car_df is not None:
                            car_context = get_car_context(prompt, car_df)
                            if car_context:
                                enhanced_system = get_car_sales_system_prompt(car_data_summary) + f"\n\nRELEVANT CARS:\n{car_context}"
                            else:
                                enhanced_system = get_car_sales_system_prompt(car_data_summary)
                        else:
                            enhanced_system = get_car_sales_system_prompt(car_data_summary)
                        
                        response = st.session_state.llm.invoke([
                            SystemMessage(content=enhanced_system),
                            HumanMessage(content=prompt)
                        ]).content
                    
                    # Update memory if enabled
                    if use_memory:
                        st.session_state.memory.chat_memory.add_user_message(prompt)
                        st.session_state.memory.chat_memory.add_ai_message(response)
                    
                    st.write(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    
                except Exception as e:
                    st.error(f"Error generating response: {e}")
                    st.session_state.messages.append({"role": "assistant", "content": "Sorry, I encountered an error. Please try again."})
    
    # Footer
    st.markdown(
        "<div style='text-align:center;color:gray;font-size:0.9em;margin-top:1rem;'>"
        "Enhanced with LangChain: Memory & Conversation History</div>",
        unsafe_allow_html=True,
    )

if __name__ == "__main__":
    main()
