import streamlit as st
import pandas as pd
import pandasai as pai
from pandasai_litellm import LiteLLM

st.set_page_config(page_title="PandasLLM EDA", layout='wide')
st.title("EDA using Natural Language")

try:
    ai_api_key = st.secrets["ai_api_key"]
except Exception as e:
    ai_api_key  = st.text_input("Enter OpenAI api key: ", type='password')
    if not ai_api_key:
        st.info("You must add your API key to continue!")
        st.stop()

csv_data = st.sidebar.file_uploader("Upload CSV file", type='csv')

if csv_data:
    df = pd.read_csv(csv_data)
    st.write("### Preview of data", df.head(3))

    user_query = st.text_input("Query your data in natural langauge")

    if st.button("Analyze"):
        if not user_query.strip():
            st.warning("Please enter a query!")
        else:
            with st.spinner("Processing your request"):
                try:
                    llm = LiteLLM(
                        model = 'gpt-4o-mini',
                        temperature = 0
                    )

                    pai.config.set({"llm" : llm})

                    smart_df = pai.SmartDataframe(df)

                    answer = smart_df.chat(user_query)

                    st.success("Result: ")
                    
                    if isinstance(answer, pd.DataFrame):
                        st.dataframe(answer)
                    else:
                        st.write(answer)  # fallback to just showing whatever answer

                except Exception as e:
                    st.error(f"Something went wrong: {e}")
                    st.info("Check API key, or the answer was non-executable")
