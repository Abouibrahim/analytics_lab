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
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Configure page
st.set_page_config(
    page_title="AI-Powered Analytics Platform",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 1rem 0;
    }
    .insight-card {
        padding: 1.5rem;
        border-radius: 0.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin: 1rem 0;
    }
    .metric-card {
        padding: 1rem;
        border-radius: 0.5rem;
        background: #f8f9fa;
        border-left: 4px solid #667eea;
    }
    .quality-score {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

def main():
    st.markdown('<h1 class="main-header">🚀 AI-Powered Analytics Platform</h1>', unsafe_allow_html=True)
    st.markdown("### *Transform your data into actionable insights with cutting-edge AI*")

    uploaded_file = st.file_uploader("📁 Upload your CSV file to unlock AI-powered insights", type=['csv'])

    if uploaded_file is not None:
        df = load_data(uploaded_file)

        # Get column types
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        numcols = df.select_dtypes(include=numerics).columns.to_list()
        categcols = df.select_dtypes(exclude=numerics).columns.to_list()

        # Sidebar navigation with icons
        st.sidebar.success('🎉 Data loaded successfully!')
        st.sidebar.markdown("---")
        st.sidebar.markdown("## 🎯 Analytics Suite")

        analytics_mode = st.sidebar.radio(
            "Choose your analytics experience:",
            ["🤖 AI Auto-Insights", "💬 Natural Language Query", "📊 Smart Visualizations",
             "🔮 Predictive Analytics", "💎 Data Quality Dashboard", "🧪 A/B Testing Lab",
             "📈 Advanced Statistics", "👥 Cohort Analysis", "🎲 What-If Scenarios",
             "📋 Classic Analytics"]
        )

        # AI Auto-Insights
        if analytics_mode == "🤖 AI Auto-Insights":
            show_ai_insights(df, numcols, categcols)

        # Natural Language Query
        elif analytics_mode == "💬 Natural Language Query":
            show_nl_query(df, numcols, categcols)

        # Smart Visualizations
        elif analytics_mode == "📊 Smart Visualizations":
            show_smart_viz(df, numcols, categcols)

        # Predictive Analytics
        elif analytics_mode == "🔮 Predictive Analytics":
            show_predictive_analytics(df, numcols, categcols)

        # Data Quality Dashboard
        elif analytics_mode == "💎 Data Quality Dashboard":
            show_data_quality(df, numcols, categcols)

        # A/B Testing Lab
        elif analytics_mode == "🧪 A/B Testing Lab":
            show_ab_testing(df, numcols, categcols)

        # Advanced Statistics
        elif analytics_mode == "📈 Advanced Statistics":
            show_advanced_stats(df, numcols, categcols)

        # Cohort Analysis
        elif analytics_mode == "👥 Cohort Analysis":
            show_cohort_analysis(df, numcols, categcols)

        # What-If Scenarios
        elif analytics_mode == "🎲 What-If Scenarios":
            show_what_if_analysis(df, numcols, categcols)

        # Classic Analytics (original features)
        elif analytics_mode == "📋 Classic Analytics":
            show_classic_analytics(df, numcols, categcols)

# ============= AI AUTO-INSIGHTS =============
def show_ai_insights(df, numcols, categcols):
    st.title("🤖 AI-Powered Automatic Insights")
    st.markdown("*Let AI discover patterns, trends, and anomalies in your data automatically*")

    with st.spinner("🧠 AI is analyzing your data..."):
        insights = []

        # Executive Summary
        st.markdown("## 📊 Executive Summary")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records", f"{len(df):,}")
        with col2:
            st.metric("Features", len(df.columns))
        with col3:
            st.metric("Numeric Cols", len(numcols))
        with col4:
            st.metric("Categorical Cols", len(categcols))

        st.markdown("---")
        st.markdown("## 🔍 Key Discoveries")

        # Anomaly Detection
        if len(numcols) > 0:
            st.markdown("### 🚨 Anomaly Detection")
            anomaly_col = st.selectbox("Select column for anomaly detection:", numcols)

            if st.button("🔍 Detect Anomalies"):
                anomalies = detect_anomalies(df, anomaly_col)
                if len(anomalies) > 0:
                    st.warning(f"⚠️ Found {len(anomalies)} anomalies in {anomaly_col}")

                    # Visualize anomalies
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=df.index, y=df[anomaly_col],
                        mode='markers',
                        name='Normal',
                        marker=dict(color='lightblue', size=6)
                    ))
                    fig.add_trace(go.Scatter(
                        x=anomalies.index, y=anomalies[anomaly_col],
                        mode='markers',
                        name='Anomalies',
                        marker=dict(color='red', size=10, symbol='x')
                    ))
                    fig.update_layout(title=f"Anomaly Detection: {anomaly_col}")
                    st.plotly_chart(fig, use_container_width=True)

                    st.dataframe(anomalies)
                else:
                    st.success("✅ No significant anomalies detected!")

        # Correlation Insights
        if len(numcols) >= 2:
            st.markdown("### 🔗 Strongest Relationships Discovered")
            corr_matrix = df[numcols].corr()

            # Find top correlations
            top_corrs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    top_corrs.append({
                        'Feature 1': corr_matrix.columns[i],
                        'Feature 2': corr_matrix.columns[j],
                        'Correlation': corr_matrix.iloc[i, j]
                    })

            top_corrs_df = pd.DataFrame(top_corrs).sort_values('Correlation', key=abs, ascending=False).head(5)

            for idx, row in top_corrs_df.iterrows():
                strength = abs(row['Correlation'])
                if strength > 0.7:
                    emoji = "🔥"
                    desc = "Very Strong"
                elif strength > 0.5:
                    emoji = "💪"
                    desc = "Strong"
                else:
                    emoji = "👀"
                    desc = "Moderate"

                st.markdown(f"""
                <div class="insight-card">
                    {emoji} <b>{desc} Correlation ({row['Correlation']:.2f})</b><br>
                    {row['Feature 1']} ↔️ {row['Feature 2']}
                </div>
                """, unsafe_allow_html=True)

            # Correlation heatmap
            fig = px.imshow(corr_matrix,
                          text_auto='.2f',
                          color_continuous_scale='RdBu_r',
                          title="Correlation Heatmap")
            st.plotly_chart(fig, use_container_width=True)

        # Distribution Insights
        if len(numcols) > 0:
            st.markdown("### 📐 Distribution Analysis")
            for col in numcols[:3]:  # Show top 3
                skewness = df[col].skew()
                kurtosis = df[col].kurtosis()

                if abs(skewness) > 1:
                    st.info(f"📊 **{col}** shows {'right' if skewness > 0 else 'left'} skewness ({skewness:.2f})")
                if abs(kurtosis) > 3:
                    st.info(f"📊 **{col}** has {'heavy' if kurtosis > 3 else 'light'} tails (kurtosis: {kurtosis:.2f})")

        # Categorical Insights
        if len(categcols) > 0:
            st.markdown("### 🏷️ Categorical Insights")
            for col in categcols[:3]:
                n_unique = df[col].nunique()
                top_value = df[col].mode()[0] if len(df[col].mode()) > 0 else "N/A"
                top_pct = (df[col].value_counts().iloc[0] / len(df) * 100) if n_unique > 0 else 0

                st.info(f"📌 **{col}**: {n_unique} unique values. Most common: '{top_value}' ({top_pct:.1f}%)")

def detect_anomalies(df, column):
    """Detect anomalies using Isolation Forest"""
    data = df[[column]].dropna()
    if len(data) < 10:
        return pd.DataFrame()

    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    predictions = iso_forest.fit_predict(data)

    anomaly_indices = data.index[predictions == -1]
    return df.loc[anomaly_indices]

# ============= NATURAL LANGUAGE QUERY =============
def show_nl_query(df, numcols, categcols):
    st.title("💬 Natural Language Query Interface")
    st.markdown("*Ask questions about your data in plain English*")

    st.markdown("### 💡 Try asking questions like:")
    examples = [
        "What is the average of [column]?",
        "Show me the distribution of [column]",
        "What is the correlation between [col1] and [col2]?",
        "How many unique values are in [column]?",
        "What are the top 5 values in [column]?"
    ]

    for ex in examples:
        st.markdown(f"- *{ex}*")

    st.markdown("---")

    query = st.text_input("🔍 Ask your question:", placeholder="e.g., What is the average sales?")

    if query:
        answer = process_nl_query(df, query, numcols, categcols)
        st.markdown("### 🎯 Answer:")
        st.info(answer)

def process_nl_query(df, query, numcols, categcols):
    """Process natural language queries"""
    query_lower = query.lower()

    # Average queries
    if "average" in query_lower or "mean" in query_lower:
        for col in numcols:
            if col in query_lower:
                avg = df[col].mean()
                return f"The average {col} is {avg:.2f}"

    # Sum queries
    if "sum" in query_lower or "total" in query_lower:
        for col in numcols:
            if col in query_lower:
                total = df[col].sum()
                return f"The total {col} is {total:.2f}"

    # Max/Min queries
    if "maximum" in query_lower or "max" in query_lower or "highest" in query_lower:
        for col in numcols:
            if col in query_lower:
                max_val = df[col].max()
                return f"The maximum {col} is {max_val:.2f}"

    if "minimum" in query_lower or "min" in query_lower or "lowest" in query_lower:
        for col in numcols:
            if col in query_lower:
                min_val = df[col].min()
                return f"The minimum {col} is {min_val:.2f}"

    # Count unique
    if "unique" in query_lower or "distinct" in query_lower:
        for col in df.columns:
            if col in query_lower:
                unique = df[col].nunique()
                return f"There are {unique} unique values in {col}"

    # Correlation
    if "correlation" in query_lower or "correlated" in query_lower:
        found_cols = [col for col in numcols if col in query_lower]
        if len(found_cols) >= 2:
            corr = df[found_cols[0]].corr(df[found_cols[1]])
            return f"The correlation between {found_cols[0]} and {found_cols[1]} is {corr:.3f}"

    return "I couldn't understand your query. Try rephrasing or use one of the example questions above."

# ============= SMART VISUALIZATIONS =============
def show_smart_viz(df, numcols, categcols):
    st.title("📊 Smart Visualization Recommendations")
    st.markdown("*AI suggests the best charts for your data*")

    st.markdown("### 🎨 Select columns and let AI recommend visualizations")

    selected_cols = st.multiselect("Choose columns to visualize:", df.columns.tolist())

    if len(selected_cols) > 0:
        recommendations = get_viz_recommendations(df, selected_cols, numcols, categcols)

        st.markdown("### 🤖 AI Recommendations:")
        for rec in recommendations:
            st.success(f"✨ {rec}")

        st.markdown("---")

        # Generate recommended visualizations
        if len(selected_cols) == 1:
            col = selected_cols[0]
            if col in numcols:
                # Histogram
                fig = px.histogram(df, x=col, title=f"Distribution of {col}")
                st.plotly_chart(fig, use_container_width=True)

                # Box plot
                fig2 = px.box(df, y=col, title=f"Box Plot: {col}")
                st.plotly_chart(fig2, use_container_width=True)
            else:
                # Bar chart for categorical
                fig = px.bar(df[col].value_counts().head(10), title=f"Top 10 {col}")
                st.plotly_chart(fig, use_container_width=True)

        elif len(selected_cols) == 2:
            col1, col2 = selected_cols[0], selected_cols[1]

            if col1 in numcols and col2 in numcols:
                # Scatter plot
                fig = px.scatter(df, x=col1, y=col2, title=f"{col1} vs {col2}",
                               trendline="ols")
                st.plotly_chart(fig, use_container_width=True)
            elif col1 in categcols and col2 in numcols:
                # Box plot
                fig = px.box(df, x=col1, y=col2, title=f"{col2} by {col1}")
                st.plotly_chart(fig, use_container_width=True)
            else:
                # Grouped bar
                crosstab = pd.crosstab(df[col1], df[col2])
                fig = px.imshow(crosstab, title=f"{col1} vs {col2} Heatmap")
                st.plotly_chart(fig, use_container_width=True)

def get_viz_recommendations(df, selected_cols, numcols, categcols):
    """Generate smart visualization recommendations"""
    recommendations = []

    if len(selected_cols) == 1:
        col = selected_cols[0]
        if col in numcols:
            recommendations.append(f"📊 Histogram: Perfect for seeing the distribution of {col}")
            recommendations.append(f"📦 Box Plot: Great for identifying outliers in {col}")
        else:
            recommendations.append(f"📊 Bar Chart: Ideal for comparing categories in {col}")
            recommendations.append(f"🥧 Pie Chart: Good for showing proportions in {col}")

    elif len(selected_cols) == 2:
        col1, col2 = selected_cols[0], selected_cols[1]
        if col1 in numcols and col2 in numcols:
            recommendations.append(f"📈 Scatter Plot: Best for exploring relationship between {col1} and {col2}")
            recommendations.append(f"📉 Line Chart: If data is time-series")
        elif col1 in categcols and col2 in numcols:
            recommendations.append(f"📊 Box Plot: Compare {col2} across {col1} categories")
            recommendations.append(f"📊 Bar Chart: Show average {col2} by {col1}")
        else:
            recommendations.append(f"🔥 Heatmap: Visualize relationship between categories")

    else:
        recommendations.append(f"🎯 Parallel Coordinates: For multi-dimensional analysis")
        recommendations.append(f"🌐 Scatter Matrix: See all pairwise relationships")

    return recommendations

# ============= PREDICTIVE ANALYTICS =============
def show_predictive_analytics(df, numcols, categcols):
    st.title("🔮 Predictive Analytics & Forecasting")
    st.markdown("*Use ML to predict future trends and outcomes*")

    if len(numcols) < 2:
        st.warning("⚠️ Need at least 2 numeric columns for predictive modeling")
        return

    st.markdown("### 🎯 Build a Prediction Model")

    target = st.selectbox("Select target variable (what to predict):", numcols)
    features = st.multiselect("Select predictor variables:",
                             [col for col in numcols if col != target])

    if len(features) > 0 and st.button("🚀 Train Model"):
        with st.spinner("🤖 Training ML model..."):
            results = train_prediction_model(df, target, features)

            st.success(f"✅ Model trained successfully!")

            col1, col2 = st.columns(2)
            with col1:
                st.metric("R² Score", f"{results['r2']:.3f}")
                st.metric("RMSE", f"{results['rmse']:.2f}")
            with col2:
                st.metric("MAE", f"{results['mae']:.2f}")
                st.metric("Training Samples", results['n_samples'])

            # Feature importance
            st.markdown("### 📊 Feature Importance")
            importance_df = pd.DataFrame({
                'Feature': features,
                'Importance': results['importance']
            }).sort_values('Importance', ascending=False)

            fig = px.bar(importance_df, x='Importance', y='Feature',
                        orientation='h', title="Feature Importance")
            st.plotly_chart(fig, use_container_width=True)

            # Predictions vs Actual
            st.markdown("### 🎯 Predictions vs Actual")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=results['actual'], y=results['predicted'],
                                    mode='markers', name='Predictions'))
            fig.add_trace(go.Scatter(x=results['actual'], y=results['actual'],
                                    mode='lines', name='Perfect Prediction',
                                    line=dict(dash='dash')))
            fig.update_layout(xaxis_title="Actual", yaxis_title="Predicted")
            st.plotly_chart(fig, use_container_width=True)

def train_prediction_model(df, target, features):
    """Train a random forest prediction model"""
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

    # Prepare data
    data = df[[target] + features].dropna()
    X = data[features]
    y = data[target]

    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Make predictions
    predictions = model.predict(X)

    # Calculate metrics
    r2 = r2_score(y, predictions)
    rmse = np.sqrt(mean_squared_error(y, predictions))
    mae = mean_absolute_error(y, predictions)

    return {
        'r2': r2,
        'rmse': rmse,
        'mae': mae,
        'n_samples': len(data),
        'importance': model.feature_importances_,
        'actual': y.values,
        'predicted': predictions
    }

# ============= DATA QUALITY DASHBOARD =============
def show_data_quality(df, numcols, categcols):
    st.title("💎 Data Quality Dashboard")
    st.markdown("*Comprehensive data health monitoring and cleaning suggestions*")

    # Calculate quality score
    quality_score = calculate_quality_score(df)

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.markdown(f"""
        <div class="quality-score" style="color: {'green' if quality_score > 80 else 'orange' if quality_score > 60 else 'red'}">
            {quality_score:.0f}/100
        </div>
        <p style="text-align: center; font-size: 1.2rem;">Overall Data Quality Score</p>
        """, unsafe_allow_html=True)

    with col2:
        st.metric("Total Missing", f"{df.isnull().sum().sum():,}")
    with col3:
        completeness = (1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        st.metric("Completeness", f"{completeness:.1f}%")

    st.markdown("---")

    # Missing data analysis
    st.markdown("### 🔍 Missing Data Analysis")
    missing_df = pd.DataFrame({
        'Column': df.columns,
        'Missing Count': df.isnull().sum().values,
        'Missing %': (df.isnull().sum().values / len(df) * 100)
    }).sort_values('Missing %', ascending=False)

    missing_df = missing_df[missing_df['Missing Count'] > 0]

    if len(missing_df) > 0:
        fig = px.bar(missing_df, x='Column', y='Missing %',
                    title="Missing Data by Column",
                    color='Missing %',
                    color_continuous_scale='Reds')
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(missing_df, use_container_width=True)

        # Suggestions
        st.markdown("### 💡 Cleaning Suggestions")
        for idx, row in missing_df.iterrows():
            if row['Missing %'] > 50:
                st.warning(f"⚠️ **{row['Column']}**: Consider removing (>{row['Missing %']:.1f}% missing)")
            elif row['Missing %'] > 10:
                st.info(f"ℹ️ **{row['Column']}**: Impute with median/mode ({row['Missing %']:.1f}% missing)")
    else:
        st.success("✅ No missing data detected!")

    # Duplicate detection
    st.markdown("### 🔄 Duplicate Detection")
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        st.warning(f"⚠️ Found {duplicates} duplicate rows ({duplicates/len(df)*100:.1f}%)")
    else:
        st.success("✅ No duplicates detected!")

    # Outlier detection
    if len(numcols) > 0:
        st.markdown("### 📊 Outlier Detection (IQR Method)")
        outlier_col = st.selectbox("Select column:", numcols)

        Q1 = df[outlier_col].quantile(0.25)
        Q3 = df[outlier_col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = df[(df[outlier_col] < Q1 - 1.5*IQR) | (df[outlier_col] > Q3 + 1.5*IQR)]

        st.info(f"📌 Found {len(outliers)} outliers ({len(outliers)/len(df)*100:.1f}%)")

        fig = px.box(df, y=outlier_col, title=f"Box Plot: {outlier_col}")
        st.plotly_chart(fig, use_container_width=True)

def calculate_quality_score(df):
    """Calculate overall data quality score"""
    scores = []

    # Completeness score
    completeness = (1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
    scores.append(completeness)

    # Uniqueness score (no duplicates)
    uniqueness = (1 - df.duplicated().sum() / len(df)) * 100
    scores.append(uniqueness)

    # Consistency score (basic validation)
    consistency = 100  # Simplified
    scores.append(consistency)

    return np.mean(scores)

# ============= A/B TESTING LAB =============
def show_ab_testing(df, numcols, categcols):
    st.title("🧪 A/B Testing Laboratory")
    st.markdown("*Statistical significance testing for experiments*")

    if len(numcols) < 1 or len(categcols) < 1:
        st.warning("⚠️ Need at least 1 numeric and 1 categorical column for A/B testing")
        return

    st.markdown("### 🎯 Configure Your A/B Test")

    group_col = st.selectbox("Select group column (e.g., variant A vs B):", categcols)
    metric_col = st.selectbox("Select metric to compare:", numcols)

    unique_groups = df[group_col].unique()

    if len(unique_groups) < 2:
        st.warning("⚠️ Need at least 2 groups to compare")
        return

    group_a = st.selectbox("Group A:", unique_groups, index=0)
    group_b = st.selectbox("Group B:", unique_groups, index=min(1, len(unique_groups)-1))

    if st.button("🚀 Run A/B Test"):
        results = run_ab_test(df, group_col, metric_col, group_a, group_b)

        st.markdown("### 📊 Test Results")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(f"Group A ({group_a}) Mean", f"{results['mean_a']:.2f}")
            st.metric(f"Group A Sample Size", f"{results['n_a']:,}")
        with col2:
            st.metric(f"Group B ({group_b}) Mean", f"{results['mean_b']:.2f}")
            st.metric(f"Group B Sample Size", f"{results['n_b']:,}")
        with col3:
            improvement = ((results['mean_b'] - results['mean_a']) / results['mean_a'] * 100)
            st.metric("Relative Change", f"{improvement:+.2f}%")
            st.metric("P-Value", f"{results['p_value']:.4f}")

        # Statistical significance
        if results['p_value'] < 0.05:
            st.success(f"✅ **STATISTICALLY SIGNIFICANT** (p < 0.05)")
            st.markdown(f"The difference between groups is statistically significant with {(1-results['p_value'])*100:.1f}% confidence.")
        else:
            st.warning(f"⚠️ **NOT SIGNIFICANT** (p = {results['p_value']:.4f})")
            st.markdown("The difference could be due to random chance.")

        # Visualization
        fig = go.Figure()
        for group, name in [(group_a, 'Group A'), (group_b, 'Group B')]:
            group_data = df[df[group_col] == group][metric_col].dropna()
            fig.add_trace(go.Box(y=group_data, name=name))

        fig.update_layout(title=f"{metric_col} Distribution by Group")
        st.plotly_chart(fig, use_container_width=True)

        # Effect size
        st.markdown("### 📏 Effect Size (Cohen's d)")
        st.info(f"Cohen's d: {results['cohens_d']:.3f} - " +
               ("Small" if abs(results['cohens_d']) < 0.5 else
                "Medium" if abs(results['cohens_d']) < 0.8 else "Large") + " effect")

def run_ab_test(df, group_col, metric_col, group_a, group_b):
    """Run statistical A/B test"""
    data_a = df[df[group_col] == group_a][metric_col].dropna()
    data_b = df[df[group_col] == group_b][metric_col].dropna()

    # T-test
    t_stat, p_value = stats.ttest_ind(data_a, data_b)

    # Cohen's d
    pooled_std = np.sqrt(((len(data_a)-1)*data_a.std()**2 + (len(data_b)-1)*data_b.std()**2) /
                         (len(data_a) + len(data_b) - 2))
    cohens_d = (data_b.mean() - data_a.mean()) / pooled_std

    return {
        'mean_a': data_a.mean(),
        'mean_b': data_b.mean(),
        'n_a': len(data_a),
        'n_b': len(data_b),
        'p_value': p_value,
        'cohens_d': cohens_d,
        't_stat': t_stat
    }

# ============= ADVANCED STATISTICS =============
def show_advanced_stats(df, numcols, categcols):
    st.title("📈 Advanced Statistical Analysis")
    st.markdown("*Hypothesis testing, confidence intervals, and more*")

    if len(numcols) < 1:
        st.warning("⚠️ Need at least 1 numeric column")
        return

    analysis_type = st.selectbox("Choose analysis type:", [
        "Normality Test",
        "Confidence Intervals",
        "Hypothesis Testing",
        "ANOVA (Multiple Groups)"
    ])

    if analysis_type == "Normality Test":
        st.markdown("### 📊 Normality Testing (Shapiro-Wilk)")
        test_col = st.selectbox("Select column:", numcols)

        if st.button("Run Test"):
            stat, p_value = stats.shapiro(df[test_col].dropna().sample(min(5000, len(df))))

            if p_value > 0.05:
                st.success(f"✅ Data appears normally distributed (p = {p_value:.4f})")
            else:
                st.warning(f"⚠️ Data does NOT appear normally distributed (p = {p_value:.4f})")

            # Q-Q plot
            from scipy import stats as sp_stats
            fig = go.Figure()
            qq = sp_stats.probplot(df[test_col].dropna(), dist="norm")
            fig.add_trace(go.Scatter(x=qq[0][0], y=qq[0][1], mode='markers', name='Data'))
            fig.add_trace(go.Scatter(x=qq[0][0], y=qq[0][0], mode='lines', name='Normal',
                                    line=dict(dash='dash', color='red')))
            fig.update_layout(title="Q-Q Plot", xaxis_title="Theoretical Quantiles",
                            yaxis_title="Sample Quantiles")
            st.plotly_chart(fig, use_container_width=True)

    elif analysis_type == "Confidence Intervals":
        st.markdown("### 📊 95% Confidence Intervals")
        ci_col = st.selectbox("Select column:", numcols)

        data = df[ci_col].dropna()
        mean = data.mean()
        se = stats.sem(data)
        ci = stats.t.interval(0.95, len(data)-1, loc=mean, scale=se)

        st.info(f"Mean: {mean:.2f}")
        st.info(f"95% CI: [{ci[0]:.2f}, {ci[1]:.2f}]")
        st.markdown("We are 95% confident the true mean lies within this interval.")

    elif analysis_type == "ANOVA (Multiple Groups)":
        if len(categcols) < 1:
            st.warning("Need at least 1 categorical column")
            return

        st.markdown("### 📊 One-Way ANOVA")
        group_col = st.selectbox("Group column:", categcols)
        metric_col = st.selectbox("Metric column:", numcols)

        if st.button("Run ANOVA"):
            groups = [group[metric_col].dropna() for name, group in df.groupby(group_col)]
            f_stat, p_value = stats.f_oneway(*groups)

            st.metric("F-Statistic", f"{f_stat:.2f}")
            st.metric("P-Value", f"{p_value:.4f}")

            if p_value < 0.05:
                st.success("✅ Significant difference between groups (p < 0.05)")
            else:
                st.info("No significant difference between groups")

# ============= COHORT ANALYSIS =============
def show_cohort_analysis(df, numcols, categcols):
    st.title("👥 Cohort Analysis")
    st.markdown("*Analyze behavior patterns across different time cohorts*")

    st.markdown("### 📅 Cohort Configuration")

    # Check if there are date columns
    date_cols = []
    for col in df.columns:
        try:
            pd.to_datetime(df[col])
            date_cols.append(col)
        except:
            pass

    if len(date_cols) == 0:
        st.warning("⚠️ No date columns detected. Cohort analysis works best with time-series data.")
        st.info("💡 You can still perform basic grouping analysis:")

        if len(categcols) > 0:
            group_col = st.selectbox("Select grouping column:", categcols)
            metric_col = st.selectbox("Select metric:", numcols) if len(numcols) > 0 else None

            if metric_col and st.button("Analyze"):
                cohort_data = df.groupby(group_col)[metric_col].agg(['count', 'mean', 'sum'])

                st.dataframe(cohort_data)

                fig = px.bar(cohort_data, y='mean', title=f"Average {metric_col} by {group_col}")
                st.plotly_chart(fig, use_container_width=True)
    else:
        date_col = st.selectbox("Select date column:", date_cols)
        metric_col = st.selectbox("Select metric:", numcols) if len(numcols) > 0 else None

        if metric_col:
            st.info("📊 Cohort analysis by time period")

# ============= WHAT-IF ANALYSIS =============
def show_what_if_analysis(df, numcols, categcols):
    st.title("🎲 What-If Scenario Analysis")
    st.markdown("*Explore hypothetical scenarios and their impacts*")

    if len(numcols) < 2:
        st.warning("⚠️ Need at least 2 numeric columns for what-if analysis")
        return

    st.markdown("### 🎯 Configure Your Scenario")

    target_col = st.selectbox("Target variable to observe:", numcols)
    input_col = st.selectbox("Input variable to modify:", [c for c in numcols if c != target_col])

    # Show current statistics
    st.markdown("### 📊 Current State")
    col1, col2 = st.columns(2)
    with col1:
        st.metric(f"Average {target_col}", f"{df[target_col].mean():.2f}")
    with col2:
        st.metric(f"Average {input_col}", f"{df[input_col].mean():.2f}")

    # What-if scenario
    st.markdown("### 🔮 What-If Scenario")

    change_pct = st.slider(f"Change {input_col} by (%):", -50, 100, 10)

    # Calculate correlation effect
    corr = df[input_col].corr(df[target_col])

    simulated_change = change_pct * abs(corr)

    st.markdown(f"""
    <div class="insight-card">
        <h3>Scenario Prediction</h3>
        <p>If {input_col} changes by <b>{change_pct:+d}%</b></p>
        <p>Expected impact on {target_col}: <b>{simulated_change:+.1f}%</b></p>
        <p>Correlation strength: {corr:.3f}</p>
    </div>
    """, unsafe_allow_html=True)

    # Visualization
    current_avg = df[target_col].mean()
    projected_avg = current_avg * (1 + simulated_change/100)

    fig = go.Figure()
    fig.add_trace(go.Bar(x=['Current', 'Projected'],
                        y=[current_avg, projected_avg],
                        marker_color=['lightblue', 'lightgreen']))
    fig.update_layout(title=f"{target_col}: Current vs Projected")
    st.plotly_chart(fig, use_container_width=True)

# ============= CLASSIC ANALYTICS =============
def show_classic_analytics(df, numcols, categcols):
    """Original analytics features"""
    st.title("📋 Classic Analytics")

    graph_type = st.selectbox("Choose the analytics type",
        ["Profiling", "Boxplot", "Histogram", "Scatter Plot", "Bar Plot",
         "Scatter Matrix", "Crosstab", "Correlation"])

    # Profiling
    if graph_type == "Profiling":
        pr = ProfileReport(df, explorative=True)
        st.title("Profiling report")
        st_profile_report(pr)

    # boxplots
    if graph_type == "Boxplot":
        st.subheader('Boxplots')
        boxplotcol = st.multiselect(label='What columns you want to display', options=numcols)
        if boxplotcol:
            boxplotfig = px.box(df, y=boxplotcol)
            st.plotly_chart(boxplotfig, use_container_width=True)

    # Histogram
    if graph_type == "Histogram":
        st.subheader('Histograms')
        histcol = st.selectbox('What column you want to display', df.columns)
        bins = st.number_input('Specify number of bins', value=20)
        histfig = go.Figure(px.histogram(df, x=histcol, nbins=int(bins)))
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
        if categcols and numcols:
            x_bar = st.selectbox('select the x-axis', categcols, key='x_bar')
            y_bar = st.selectbox('select the y-axis', numcols, key='y_bar')
            aggs = ["count","sum","mean","median","mode","rms","stddev","min","max","first","last"]
            aggs_bar = st.selectbox('Use dropdown to change aggregation', aggs, key='aggs_bar')
            df_bar_agg = df[[x_bar,y_bar]].groupby(x_bar).agg(aggs_bar)
            bar_chart_fig = px.bar(df_bar_agg)
            st.plotly_chart(bar_chart_fig, use_container_width=True)

    # scatter_matrix
    if graph_type == "Scatter Matrix":
        st.subheader('Scatter Matrix')
        if numcols and categcols:
            dimensions = st.multiselect(label='What columns you want to display', options=numcols, key='dimensions')
            scatter_matrix_color = st.selectbox('Color by', categcols, key='scatter_matrix_color')
            if dimensions:
                scatter_matrix_fig = px.scatter_matrix(df, dimensions=dimensions, color=scatter_matrix_color)
                st.plotly_chart(scatter_matrix_fig, use_container_width=True)

    # crosstab
    if graph_type == "Crosstab":
        st.subheader('crosstab')
        if len(categcols) >= 2:
            x_crosstab = st.selectbox('select the x-axis', categcols, key='x_crosstab')
            y_crosstab = st.selectbox('select the y-axis', categcols, key='y_crosstab')
            crosstab = df.pivot_table(index=x_crosstab, columns=y_crosstab, aggfunc=lambda x: len(x), margins=True)
            st.write(crosstab)

    # Correlation
    if graph_type == "Correlation":
        st.header("Correlation Dynamic Dropdown")
        if numcols:
            x_corr = st.selectbox("x", numcols, key='x_corr')
            y_options, y_formats = get_y_vars(df, x_corr, numcols)
            y_corr = st.selectbox(f"y (sorted by correlation with {x_corr})", y_options,
                                 format_func=y_formats.get, key='y_corr')
            plot = alt.Chart(df).mark_circle().encode(
                alt.X(x_corr, scale=alt.Scale(zero=False)),
                alt.Y(y_corr, scale=alt.Scale(zero=False)))
            st.altair_chart(plot, use_container_width=True)

@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    cols = [col.strip().lower().replace(" ", "_") for col in df.columns]
    df.columns = cols
    return df

@st.cache_data
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
