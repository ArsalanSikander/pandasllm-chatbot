import streamlit as st
import pandas as pd
from pandas_llm import PandasLLM
import openai

st.set_page_config(page_title="PandasLLM EDA", layout='wide')
st.title("Arsalan's PandasLLM Chatbot app")

# Get OpenAI API key from Streamlit secrets
try:
    openai_api_key = st.secrets["OPENAI_API_KEY"]
except Exception as e:
    st.error(f"❌ OpenAI API key not found in secrets: {e}")
    st.info("Please add your OpenAI API key to Streamlit secrets.")
    st.stop()

# Optional: Validate API key with v1.0.0+ syntax
try:
    client = openai.OpenAI(api_key=openai_api_key)
    client.models.list()  # This is the correct syntax for v1.0.0+
except Exception as e:
    st.error(f"❌ Invalid or expired OpenAI API key: {e}")
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
                    # Initialize PandasLLM with your dataframe and OpenAI key
                    llm = PandasLLM(data=df, llm_api_key=openai_api_key)
                    
                    # Run the query – this generates, executes, and returns the result
                    result = llm.prompt(user_query)
                    
                    # Try to retrieve the generated Python code (if available)
                    python_code = getattr(llm, 'code_block', 'Code not captured')
                    
                    st.success("✅ Result:")
                    st.write(result)
                    
                    with st.expander("🔍 Show Generated Code"):
                        st.code(python_code, language="python")
                        
                except Exception as e:
                    # This will show the real error instead of "Please try later"
                    st.error(f"❌ Analysis failed: {type(e).__name__} – {str(e)}")
                    st.info("Possible reasons: invalid API key, no credits left, or the LLM couldn't generate valid pandas code.")
        else:
            st.warning("Please enter a query")

# Optional: Add a note on the sidebar about requirements
st.sidebar.markdown("---")
st.sidebar.info(
    "**Requirements**\n\n"
    "- OpenAI API key with credits\n"
    "- Python packages: `pandas-llm`, `openai`, `streamlit`, `pandas`\n\n"
    "Install with: `pip install pandas-llm openai streamlit pandas`"
)