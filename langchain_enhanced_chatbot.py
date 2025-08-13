import os
import streamlit as st
from dotenv import load_dotenv
import pandas as pd
from typing import List, Dict, Any

# Load environment variables
load_dotenv()

# LangChain imports
from langchain_anthropic import ChatAnthropic
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from langchain.retrievers import DataFrameRetriever
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DataFrameLoader

# Import existing utilities
from src.utils import load_car_data, get_car_context, is_car_query
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

def create_car_knowledge_base(car_df):
    """Create a vector database for car information retrieval"""
    if car_df is None:
        return None
    
    # Create text descriptions for each car
    car_descriptions = []
    for _, car in car_df.iterrows():
        description = f"""
        VRM: {car['VRM']}
        Make: {car['MAKE']}
        Model: {car['MODEL']}
        Age Group: {car['AGE_GROUP']}
        Body Type: {car['BODY_TYPE']}
        Fuel Type: {car['FUEL_TYPE']}
        Price: £{car['PRICE']:,}
        Mileage: {car['MILEAGE']:,} miles
        Desirability Score: {car['RETAIL_DESIRABILITY_SCORE']:.2f}
        """
        car_descriptions.append(description)
    
    # Create documents for vector store
    documents = []
    for i, desc in enumerate(car_descriptions):
        documents.append({
            "page_content": desc,
            "metadata": {
                "vrm": car_df.iloc[i]['VRM'],
                "make": car_df.iloc[i]['MAKE'],
                "model": car_df.iloc[i]['MODEL'],
                "price": car_df.iloc[i]['PRICE'],
                "body_type": car_df.iloc[i]['BODY_TYPE'],
                "fuel_type": car_df.iloc[i]['FUEL_TYPE'],
                "desirability": car_df.iloc[i]['RETAIL_DESIRABILITY_SCORE']
            }
        })
    
    # Create embeddings and vector store
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    
    # Create FAISS vector store
    texts = [doc["page_content"] for doc in documents]
    metadatas = [doc["metadata"] for doc in documents]
    
    vectorstore = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
    
    return vectorstore

def enhanced_car_query(query: str, vectorstore, car_df, llm):
    """Enhanced car query using LangChain retrieval"""
    
    # Get relevant cars from vector store
    docs = vectorstore.similarity_search(query, k=5)
    
    # Extract car information
    relevant_cars = []
    for doc in docs:
        metadata = doc.metadata
        relevant_cars.append({
            'VRM': metadata['vrm'],
            'MAKE': metadata['make'],
            'MODEL': metadata['model'],
            'PRICE': metadata['price'],
            'BODY_TYPE': metadata['body_type'],
            'FUEL_TYPE': metadata['fuel_type'],
            'DESIRABILITY': metadata['desirability']
        })
    
    # Create context from retrieved cars
    context = "RELEVANT CARS FROM KNOWLEDGE BASE:\n"
    for car in relevant_cars:
        context += f"- VRM: {car['VRM']} | {car['MAKE']} {car['MODEL']} (£{car['PRICE']:,}, {car['BODY_TYPE']}, {car['FUEL_TYPE']}, Desirability: {car['DESIRABILITY']:.2f})\n"
    
    # Create enhanced prompt
    enhanced_prompt = f"""
    User Query: {query}
    
    {context}
    
    Please provide 3 recommendations based on the user's query and the relevant cars above.
    Focus on variety and match the user's criteria as closely as possible.
    """
    
    # Get response from LLM
    response = llm.invoke([HumanMessage(content=enhanced_prompt)])
    
    return response.content

def create_advanced_prompt_template():
    """Create an advanced prompt template with LangChain"""
    
    template = """
    You are an intelligent car sales assistant with access to our car inventory and conversation history.
    
    Previous conversation:
    {history}
    
    Current user query: {input}
    
    Available car data:
    {car_context}
    
    Instructions:
    1. Use conversation history to provide personalized recommendations
    2. Consider previous preferences and constraints mentioned
    3. Provide exactly 3 diverse recommendations when possible
    4. Include VRM, make, model, price, and desirability score
    5. If no cars match, suggest alternatives or ask for clarification
    
    Response:
    """
    
    return PromptTemplate(
        input_variables=["history", "input", "car_context"],
        template=template
    )

# ----------------------
# Streamlit App
# ----------------------

def main():
    st.set_page_config(
        page_title="Enhanced Car Sales Assistant", 
        page_icon="🚗", 
        layout="centered"
    )
    
    st.title("🚗 Enhanced Car Sales Assistant")
    st.caption("Powered by LangChain + Claude 3.5 Sonnet with Memory & Knowledge Retrieval")
    
    # Load car data
    car_data_summary, car_df = load_car_data()
    
    # Initialize LangChain components
    if 'langchain_initialized' not in st.session_state:
        with st.spinner("Initializing LangChain components..."):
            try:
                llm, memory, conversation = setup_langchain_components()
                vectorstore = create_car_knowledge_base(car_df)
                
                st.session_state.llm = llm
                st.session_state.memory = memory
                st.session_state.conversation = conversation
                st.session_state.vectorstore = vectorstore
                st.session_state.langchain_initialized = True
                st.success("✅ LangChain components initialized!")
            except Exception as e:
                st.error(f"Failed to initialize LangChain: {e}")
                return
    
    # Sidebar
    with st.sidebar:
        st.subheader("Enhanced Features")
        
        # Feature toggles
        use_memory = st.checkbox("Use Conversation Memory", value=True)
        use_retrieval = st.checkbox("Use Knowledge Retrieval", value=True)
        use_advanced_prompting = st.checkbox("Use Advanced Prompting", value=True)
        
        st.markdown("---")
        st.subheader("Memory Status")
        if use_memory and st.session_state.memory:
            memory_length = len(st.session_state.memory.chat_memory.messages)
            st.info(f"Remembering {memory_length} messages")
        
        st.markdown("---")
        st.subheader("Knowledge Base")
        if use_retrieval and st.session_state.vectorstore:
            st.success("✅ Vector database ready")
            st.info(f"Indexed {len(car_df)} cars")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I'm your enhanced car sales assistant with memory and knowledge retrieval. Ask me about our inventory! 🚗"}
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
            with st.spinner("Thinking..."):
                try:
                    if use_retrieval and st.session_state.vectorstore:
                        # Use enhanced retrieval
                        response = enhanced_car_query(
                            prompt, 
                            st.session_state.vectorstore, 
                            car_df, 
                            st.session_state.llm
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
        "Enhanced with LangChain: Memory, Retrieval, & Advanced Prompting</div>",
        unsafe_allow_html=True,
    )

if __name__ == "__main__":
    main()
