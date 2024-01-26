############## PART A #################
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
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

############# PART B #######################

#############################################################################

# functions

def explore_categorical(df, x, fillna = True, placeholder = 'MISSING',
                        figsize = (6,4), order = None):
 
  # Make a copy of the dataframe and fillna 
  temp_df = df.copy()
  # Before filling nulls, save null value counts and percent for printing 
  null_count = temp_df[x].isna().sum()
  null_perc = null_count/len(temp_df)* 100
  # fillna with placeholder
  if fillna == True:
    temp_df[x] = temp_df[x].fillna(placeholder)
  # Create figure with desired figsize
  fig, ax = plt.subplots(figsize=figsize)
  # Plotting a count plot 
  sns.countplot(data=temp_df, x=x, ax=ax, order=order)
  # Rotate Tick Labels for long names
  ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
  # Add a title with the feature name included
  ax.set_title(f"Column: {x}")
  
  # Fix layout and show plot (before print statements)
  fig.tight_layout()
  plt.show()
    
  return fig, ax

def explore_numeric(df, x, figsize=(6,5) ):
  """Source: https://login.codingdojo.com/m/606/13765/117605"""
  # Making our figure with gridspec for subplots
  gridspec = {'height_ratios':[0.7,0.3]}
  fig, axes = plt.subplots(nrows=2, figsize=figsize,
                           sharex=True, gridspec_kw=gridspec)
  # Histogram on Top
  sns.histplot(data=df, x=x, ax=axes[0])
  # Boxplot on Bottom
  sns.boxplot(data=df, x=x, ax=axes[1])
  ## Adding a title
  axes[0].set_title(f"Column: {x}", fontweight='bold')
  ## Adjusting subplots to best fill Figure
  fig.tight_layout()
  # Ensure plot is shown before message
  plt.show()
  return fig


######################################################################

# Explore Column Plots

# Add the selection of a column to explore
columns_to_use = ['Item_Weight', 'Item_Fat_Content', 'Item_Visibility', 'Item_Type', 'Item_MSRP', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type', 'Item_Outlet_Sales']

# Conditional statement to determine which function to use
column = st.selectbox(label="Select a column", options=columns_to_use)
if df[column].dtype == 'object':
    fig, ax  = explore_categorical(df, column)
else:
    fig = explore_numeric(df, column)
    
st.markdown("#### Displaying appropriate plot based on selected column")

# Display the appropriate exploration plots depending on the type of feature selected
st.pyplot(fig)

#######################################################################

# Feature vs Target Plots

# Add the selection of a feature to explore (exclude the target in your selection list)
columns_to_use_no_target = ['Item_Weight', 'Item_Fat_Content', 'Item_Visibility', 'Item_Type', 'Item_MRP', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type']

feature = st.selectbox(label='Select a feature to compare with Item Outlet Sales', options=columns_to_use_no_target)
# fill null values to avoid error
df['Outlet_Size'] = df['Outlet_Size'].fillna('MISSING')
# Display the appropriate plot of the feature versus the target depending on the type of feature selected

# numeric function
def plotly_numeric_vs_target(df, x, y='Item_Outlet_Sales', trendline='ols',add_hoverdata=True):
    if add_hoverdata == True:
        hover_data = list(df.columns)
    else: 
        hover_data = None
        
    pfig = px.scatter(df, x=x, y=y,width=800, height=600,
                     hover_data=hover_data,
                      trendline=trendline,
                      trendline_color_override='red',
                     title=f"{x} vs. {y}")
    
    pfig.update_traces(marker=dict(size=3),
                      line=dict(dash='dash'))
    return pfig

# categorical function
def plotly_categorical_vs_target(df, x, y='Item_Outlet_Sales', histfunc='avg', width=800,height=500):
    fig = px.histogram(df, x=x,y=y, color=x, width=width, height=height,
                       histfunc=histfunc, title=f'Compare {histfunc.title()} {y} by {x}')
    fig.update_layout(showlegend=False)
    return fig
    
# Conditional statement to determine which function to use
if df[feature].dtype == 'object':
    fig_vs  = plotly_categorical_vs_target(df, x = feature)
else:
    fig_vs  = plotly_numeric_vs_target(df, x = feature)

st.plotly_chart(fig_vs)






