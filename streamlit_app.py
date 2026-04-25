import streamlit as st
import pandas as pd
from pandas_llm import PandasLLM

st.set_page_config(page_title="PandasLLM EDA", layout='wide')
st.title("Arsalan's PandasLLM Chatbot app")

# Get OpenAI API key from secrets
try:
    openai_api_key = st.secrets["OPENAI_API_KEY"]
except Exception as e:
    st.error(f"Error accessing OpenAI API Key: {e}")
    st.stop()

csv_file = st.sidebar.file_uploader("Upload data (CSV)", type=["csv"])

if csv_file:
    df = pd.read_csv(csv_file)
    st.write("### Dataset Preview", df.head(3))

    user_query = st.text_input("Query your data in natural language!")

    if st.button("Analyze"):
        if user_query:
            with st.spinner("Processing request"):
                # Initialize PandasLLM with your dataframe and OpenAI key
                llm = PandasLLM(data=df, llm_api_key=openai_api_key)

                try:
                    # Directly use the prompt method (generates + executes code)
                    result = llm.prompt(user_query)

                    # Optionally retrieve the generated code (if needed)
                    python_code = getattr(llm, 'code_block', 'Code not available')

                    st.success("Result:")
                    st.write(result)

                    with st.expander("Show Generated Code"):
                        st.code(python_code, language="python")

                except Exception as e:
                    st.error(f"Execution Error: {e}")
                    st.info("The LLM generated code that could not be run!")
        else:
            st.warning("Please enter a query")