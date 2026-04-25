import streamlit as st
import pandas as pd
from pandas_llm import PandasLLM
import google.generativeai as genai

st.set_page_config(page_title="PandasLLM EDA", layout='wide')
st.title("Arsalan's PandasLLM Chatbot app")

try:
    gem_key = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=gem_key)
    model = genai.GenerativeModel('gemini-flash')
except Exception as e:
    st.error(f"Error accessing the Gemini Key: {e}")
    st.stop()

csv_file = st.sidebar.file_uploader("Upload data (CSV)", type=["csv"])

if csv_file:
    df = pd.read_csv(csv_file)
    st.write("### Dataset Preview", df.head(3))

    conv_df = PandasLLM(data=df)

    user_query = st.text_input("Query your data in natural langauge!")

    if st.button("Analyze"):
        if user_query:
            with st.spinner("Processing request"):
                prompt = conv_df.create_prompt(user_query)

                try:
                    gem_response = model.generate_content(prompt)
                    python_code = gem_response.text

                    st.code(python_code, language='python')

                    result = conv_df._execInSandbox(python_code)
                    st.success("Result:")
                    st.write(result)

                    with st.expander("Show Generated Code"):
                        st.code(python_code, language="python")

                except Exception as e:
                    st.error("Execution Error: {e}")
                    st.info("The LLM generated code that could not be run!")
        else:
            st.warning("Please enter a query")