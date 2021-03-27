import streamlit as st
import pandas as pd
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
import os
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

def main():
    uploaded_file = st.file_uploader("Choose a file")    
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        numcols = df.select_dtypes(include=numerics).columns.to_list()
        categcols = df.select_dtypes(exclude=numerics).columns.to_list()

        # a selector for the graph type
        st.sidebar.info('Your Catalog is *ready!* :sunglasses:')
        graph_type = st.sidebar.radio("Choose the analytics type",
            ["Profiling", "Boxplot", "Histogram", "Scatter Plot", "Bar Plot", "Scatter Matrix", "Crosstab", "Correlation"])
        # Profiling
        if graph_type == "Profiling":
            pr = ProfileReport(df, explorative=True)
            st.title("Pandas Profiling in Streamlit")
            st_profile_report(pr)

        # boxplots  
        if graph_type == "Boxplot":
            st.subheader('Boxplots')
            boxplotcol = st.multiselect(label='What columns you want to display', options=numcols)
            boxplotfig = px.box(df, y=boxplotcol)
            st.plotly_chart(boxplotfig, use_container_width=True)
        # Histogram
        if graph_type == "Histogram":
            st.subheader('Histograms')
            histcol = st.selectbox('What column you want to display', df.columns)
            bins = st.number_input('Specify number of bins', value=20)
            histfig = go.Figure(px.histogram(df, x=histcol,nbins=bins))
            st.plotly_chart(histfig, use_container_width=True)
        # scatter plot
        if graph_type == "Scatter Plot":
            st.subheader('Scatter Plot')
            x_scatter_plot = st.selectbox('select the x-axis', numcols, key='x_scatter_plot')
            y_scatter_plot = st.selectbox('select the y-axis', numcols, key='y_scatter_plot')
            scatterplotfig = px.scatter(df, x=x_scatter_plot, y=y_scatter_plot)
            st.plotly_chart(scatterplotfig, use_container_width=True)
        # bar plot
        if graph_type == "Bar Plot":
            st.subheader('Bar Plot')
            x_bar = st.selectbox('select the x-axis', categcols, key='x_bar')
            y_bar = st.selectbox('select the y-axis', numcols, key='y_bar')
            aggs = ["count","sum","mean","median","mode","rms","stddev","min","max","first","last"]
            aggs_bar = st.selectbox('Use dropdown to change aggregation', aggs, key='aggs_bar')
            df_bar_agg = df[[x_bar,y_bar]].groupby(x_bar).agg(aggs_bar)            
            bar_chart_fig = px.bar(df_bar_agg)
            st.plotly_chart(bar_chart_fig, use_container_width=True)
        #scatter_matrix
        if graph_type == "Scatter Matrix":
            st.subheader('Scatter Matrix')
            dimensions = st.multiselect(label='What columns you want to display', options=numcols,key='dimensions')
            scatter_matrix_color = st.selectbox('Color by', categcols, key='scatter_matrix_color')
            scatter_matrix_fig = px.scatter_matrix(df, dimensions=dimensions, color=scatter_matrix_color)
            st.plotly_chart(scatter_matrix_fig, use_container_width=True)
        # crosstab
        if graph_type == "Crosstab":
            st.subheader('crosstab')
            x_crosstab = st.selectbox('select the x-axis', categcols, key='x_crosstab')
            y_crosstab = st.selectbox('select the y-axis', categcols, key='y_crosstab')
            crosstab = df.pivot_table(index= x_crosstab, columns= y_crosstab, aggfunc=lambda x: len(x), margins=True)
            st.write(crosstab)
        # Correlation
        if graph_type == "Correlation":
            st.header("Correlation Dynamic Dropdown")
            x_corr = st.selectbox("x", numcols, key = 'x_corr')
            y_options, y_formats = get_y_vars(df, x_corr, numcols)
            y_corr = st.selectbox(f"y (sorted by correlation with {x_corr})", y_options, format_func=y_formats.get, key = 'y_corr')
            plot = alt.Chart(df).mark_circle().encode(
                alt.X(x_corr,scale=alt.Scale(zero=False)),
                alt.Y(y_corr,scale=alt.Scale(zero=False)))
            st.altair_chart(plot)

@st.cache(allow_output_mutation=True)
def load_data(file):
    df = pd.read_excel(file)
    cols = [col.strip().lower().replace(" ", "_") for col in df.columns]
    df.columns = cols
    return df

@st.cache(allow_output_mutation=True)
def get_y_vars(dataset, x, variables):
    corrs = dataset.corr()[x]
    remaining_variables = [v for v in variables if v != x]
    sorted_remaining_variables = sorted(
        remaining_variables, key=lambda v: corrs[v], reverse=True
    )
    format_dict = {v: f"{v} ({corrs[v]:.2f})" for v in sorted_remaining_variables}
    return sorted_remaining_variables, format_dict 

if __name__ == "__main__":
    main()
