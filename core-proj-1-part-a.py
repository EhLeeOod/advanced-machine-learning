import streamlit as st
import pandas as pd

# title
st.title('Sales Price Analysis')

# display df
st.header('Product Sales Data')
df = pd.read_csv('Data/df_sales_new.csv')
st.dataframe(df)

# A button to trigger the display of a dataframe of Descriptive Statistics
st.subheader('Descriptive Statistics')
df_describe = df.describe()
if st.button('Show Descriptive Statistics'):
    st.dataframe(df_describe)

# A button to trigger the display of the summary information (the output of .info)

# use IO buffer to capture output of df.info()
from io import StringIO
buffer = StringIO()
# write info to buffer
df.info(buf=buffer)
# retrieve content from buffer
summary_info = buffer.getvalue()

st.subheader('Summary Info')
if st.button('Show Summary Info'):
    st.text(summary_info)

# A button to trigger the display of the Null values
null_values = df.isna().sum()

st.subheader('Null Values')
if st.button('Show Null Values'):
    st.dataframe(null_values)



