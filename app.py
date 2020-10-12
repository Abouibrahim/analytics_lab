import streamlit as st
import pandas as pd
import pandas_profiling as pp
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
import os
import plotly.express as px
import plotly.graph_objects as go


def main():
    FILE_TYPES = ["csv"]
    file = st.sidebar.file_uploader("Upload your CSV file", type=FILE_TYPES)

    # Render the readme as markdown using st.markdown.
    with open('instructions.md') as finstruct:
        readme_text = st.markdown(finstruct.read())

    st.sidebar.markdown("**Analytics Engine** ⚙️")
    app_mode = st.sidebar.selectbox("Choose the analytics category",
        ["Show instructions", "Profile", "Exploratory Data Analysis", "Correlation Analysis", "Analytics 4", "Analytics 5"])

    # Create a text element and let the reader know the data is loading.
    #data_load_state = st.text('Loading data...')

    # Load data
    if file is not None:
        df, numcols, categcols = load_data(file)

    # Notify the reader that the data was successfully loaded.
    #data_load_state.text('Loading data...done!')

        # Once we have the file uploaded, select a category on the sidebar.

        if app_mode == "Show instructions":
            st.sidebar.success('To continue select a category.')
        elif app_mode == "Profile":
            readme_text.empty()
            profile(df)
        elif app_mode == "Exploratory Data Analysis":
            readme_text.empty()
            exploratory(df, numcols, categcols)

        elif app_mode == "Correlation Analysis":
            readme_text.empty()
            correlate(df, numcols)

@st.cache(allow_output_mutation=True)
def load_data(file):
    df = pd.read_csv(file)
    cols = [col.strip().lower().replace(" ", "_") for col in df.columns]
    df.columns = cols
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    numcols = df.select_dtypes(include=numerics).columns.to_list()
    categcols = df.select_dtypes(exclude=numerics).columns.to_list()
    return df, numcols, categcols

def profile(df):

    st.title('Profiling Report')
    # show head
    st.subheader('Sample ')
    if st.checkbox("Show First Rows", False):
        st.subheader("The first 5 rows")
        st.write(df.head())
    # show tail
    if st.checkbox("Show Last Rows", False):
        st.subheader("The first 5 rows")
        st.write(df.tail())
    # Dataset statistics
    st.subheader('Dataset statistics ')
    dfstats = {'Number of variables': df.shape[1], 'Number of observations': len(df)}
    dfstats = pd.DataFrame.from_dict(dfstats,orient='index', columns=[' '])
    st.table(dfstats)


    # Get summary statistics for the object (string) columns:
    st.subheader('Categorical variables')
    st.table(df.describe(include=[np.object]).T)

    # Types of Data
    st.subheader('Types of Data ')
    st.markdown('The types of descriptive methods that you use depend largely on the types of data that you have and the number of variables you are analyzing. ')
    st.table(pd.DataFrame(df.dtypes, columns =['Type ']))

    # find missing numbers
    st.subheader('Number of missing values ')
    st.write(df.isna().sum())

    # drops rows when specified columns have missing values
    non_null_cols = st.multiselect(label='Drop rows when specified columns have missing values',
                                    options=df.columns.tolist(), key='non_null_cols')

    if st.button('Drop Null Values', key = 'dropna'):
        df.dropna(subset=non_null_cols, inplace=True)

    st.markdown(f"{len(df)} records remaining")

    # fixing data types
    #df[["a", "b"]] = df[["a", "b"]].apply(pd.to_numeric)

def exploratory(df, numcols, categcols):
    with open('exploratory.md') as fexplore:
        st.sidebar.markdown(fexplore.read())
    st.title('Describing Data')

    # Descriptive statistics
    st.subheader('Descriptive statistics')
    st.markdown('count, mean, std, min, max as well as 25, 50 and 75 percentiles')
    st.write(df.describe(include=[np.number]).T)

    # boxplots
    st.subheader('Boxplots')
    boxplotcol = st.multiselect(label='What columns you want to display', options=numcols)
    boxplotfig = px.box(df, y=boxplotcol)
    st.plotly_chart(boxplotfig, use_container_width=True)

    # Histogram
    st.subheader('Histogram')
    with open('histogram.md') as fhist:
        st.markdown(fhist.read())

    histcol = st.selectbox('What column you want to display', df.columns)
    bins = st.number_input('Specify number of bins', value=20)
    histfig = go.Figure(px.histogram(df, x=histcol,nbins=bins))
    st.plotly_chart(histfig, use_container_width=True)

    # scatter plot
    st.subheader('Scatter Plot')
    x_scatter_plot = st.selectbox('select the x-axis', numcols, key='x_scatter_plot')
    y_scatter_plot = st.selectbox('select the y-axis', numcols, key='y_scatter_plot')
    scatterplotfig = px.scatter(df, x=x_scatter_plot, y=y_scatter_plot)
    st.plotly_chart(scatterplotfig, use_container_width=True)

    # bar plot
    st.subheader('Bar Plot')
    x_bar = st.selectbox('select the x-axis', categcols, key='x_bar')
    y_bar = st.selectbox('select the y-axis', categcols, key='y_bar')
    barfig = px.bar(df, x=x_bar, y=y_bar)
    st.plotly_chart(barfig, use_container_width=True)

    #scatter_matrix
    st.subheader('Scatter Matrix')
    dimensions = st.multiselect(label='What columns you want to display', options=numcols,key='dimensions')
    scatter_matrix_color = st.selectbox('Color by', categcols, key='scatter_matrix_color')
    scatter_matrix_fig = px.scatter_matrix(df, dimensions=dimensions, color=scatter_matrix_color)
    st.plotly_chart(scatter_matrix_fig, use_container_width=True)



    # crosstab
    st.subheader('crosstab')
    st.markdown('crosstab')
    x_crosstab = st.selectbox('select the x-axis', categcols, key='x_crosstab')
    y_crosstab = st.selectbox('select the y-axis', categcols, key='y_crosstab')
    crosstab = df.pivot_table(index= x_crosstab, columns= y_crosstab,
                                aggfunc=lambda x: len(x), margins=True)
    st.write(crosstab)

@st.cache
def get_y_vars(dataset, x, variables):
    corrs = dataset.corr()[x]
    remaining_variables = [v for v in variables if v != x]
    sorted_remaining_variables = sorted(
        remaining_variables, key=lambda v: corrs[v], reverse=True
    )
    format_dict = {v: f"{v} ({corrs[v]:.2f})" for v in sorted_remaining_variables}
    return sorted_remaining_variables, format_dict

def correlate(df, numcols):
    st.header("Correlation Dynamic Dropdown")
    x_corr = st.selectbox("x", numcols, key = 'x_corr')
    y_options, y_formats = get_y_vars(df, x_corr, numcols)
    y_corr = st.selectbox(f"y (sorted by correlation with {x_corr})", y_options, format_func=y_formats.get, key = 'y_corr')
    plot = alt.Chart(df).mark_circle().encode(
        alt.X(x_corr,scale=alt.Scale(zero=False)),
        alt.Y(y_corr,scale=alt.Scale(zero=False)))
    st.altair_chart(plot)



if __name__ == "__main__":
    main()
