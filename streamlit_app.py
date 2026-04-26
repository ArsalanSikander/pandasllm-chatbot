import streamlit as st
import pandas as pd
from pandas_llm import PandasLLM
import openai

st.set_page_config(page_title="PandasLLM EDA", layout='wide')
st.title("Arsalan's PandasLLM Chatbot app")

try:
    openai_api_key = st.secrets["OPENAI_API_KEY"]
except Exception as e:
    st.error(f"No api key found: {e}")
    st.info("Add your OpenAI API key")
    st.stop()

try:
    client = openai.OpenAI(api_key=openai_api_key)
    client.models.list() 
except Exception as e:
    st.error(f"Invalid or expired OpenAI API key: {e}")
    st.stop()

csv_file = st.sidebar.file_uploader("Upload data (CSV)", type=["csv"])

if csv_file:
    df = pd.read_csv(csv_file)
    st.write("### Dataset Preview", df.head(3))

    user_query = st.text_input("Query your data in natural language!")

    if st.button("Analyze"):
        if user_query:
            with st.spinner("Processing request..."):
                try:

                    llm = PandasLLM(data=df, llm_api_key=openai_api_key)

                    result = llm.prompt(user_query)
                    
                    python_code = getattr(llm, 'code_block', 'Code not captured')
                    
                    st.success("Result:")
                    st.write(result)
                    
                    with st.expander("Generated Code"):
                        st.code(python_code, language="python")
                        
                except Exception as e:
                    # This will show the real error
                    st.error(f"Analysis failed: {type(e).__name__} – {str(e)}")
                    st.info("Possible reasons: invalid API key, no credits left, or the LLM couldn't generate valid pandas code.")
        else:
            st.warning("Please enter a query")

