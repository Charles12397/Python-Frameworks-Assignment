# streamlit_app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import re
import os
import numpy as np

# Set page configuration
st.set_page_config(
    page_title="CORD-19 Data Explorer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #2e86ab;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and cache the cleaned dataset"""
    try:
        if not os.path.exists('cleaned_metadata.csv'):
            st.warning("Cleaned data not found. Please run data_cleaning.py first.")
            return None
        
        df = pd.read_csv('cleaned_metadata.csv')
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def generate_word_cloud(text_series, title):
    """Generate word cloud from text series"""
    try:
        # Combine all text (exclude placeholder values)
        valid_text = text_series[
            (~text_series.isnull()) & 
            (~text_series.str.contains('Unknown', case=False, na=False))
        ]
        
        if len(valid_text) == 0:
            return None
            
        all_text = ' '.join(valid_text.astype(str))
        
        # Clean text
        words = re.findall(r'\b[a-zA-Z]{3,}\b', all_text.lower())
        
        # Remove common stop words
        stop_words = {
            'the', 'and', 'of', 'in', 'to', 'a', 'for', 'with', 'on', 'by', 
            'as', 'an', 'from', 'that', 'this', 'is', 'are', 'was', 'were',
            'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'which',
            'who', 'what', 'when', 'where', 'why', 'how', 'their', 'there',
            'study', 'research', 'paper', 'article', 'journal'
        }
        
        filtered_words = [word for word in words if word not in stop_words]
        
        if len(filtered_words) > 10:
            wordcloud = WordCloud(
                width=800, height=400, 
                background_color='white',
                max_words=100,
                colormap='viridis',
                relative_scaling=0.5
            ).generate(' '.join(filtered_words))
            
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            ax.set_title(title, fontsize=16, fontweight='bold')
            return fig
    except Exception as e:
        st.error(f"Error generating word cloud: {e}")
    return None

def main():
    # Main header
    st.markdown('<h1 class="main-header">🔬 CORD-19 COVID-19 Research Explorer</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    This interactive dashboard explores the CORD-19 dataset containing metadata about COVID-19 research papers.
    Use the filters in the sidebar to explore different aspects of the data.
    
    **Note:** Make sure you've run `data_cleaning.py` first to generate the cleaned dataset.
    """)
    
    # Load data
    df = load_data()
    
    if df is None:
        st.error("""
        ❌ No cleaned data found! Please run the following commands in order:
        
        1. `python data_exploration.py`
        2. `python data_cleaning.py` 
        3. `python analysis_visualization.py`
        
        Then refresh this app.
        """)
        return
    
    # Sidebar filters
    st.sidebar.title("📊 Filters")
    
    # Year range filter
    if 'publication_year' in df.columns:
        min_year = int(df['publication_year'].min())
        max_year = int(df['publication_year'].max())
        
        year_range = st.sidebar.slider(
            "Select Publication Year Range",
            min_value=min_year,
            max_value=max_year,
            value=(min_year, max_year)
        )
    else:
        year_range = (2020, 2022)
        st.sidebar.info("Using default year range")
    
    # Journal filter
    journal_options = []
    if 'journal' in df.columns:
        top_journals = df['journal'].value_counts().head(20)
        journal_options = top_journals.index.tolist()
    
    selected_journals = st.sidebar.multiselect(
        "Select Journals (Top 20 shown)",
        options=journal_options,
        default=journal_options[:3] if journal_options else []
    )
    
    # Abstract length filter
    if 'abstract_word_count' in df.columns:
        min_words = int(df['abstract_word_count'].min())
        max_words = int(df['abstract_word_count'].max())
        
        word_range = st.sidebar.slider(
            "Abstract Word Count Range",
            min_value=min_words,
            max_value=min(max_words, 1000),  # Cap at 1000 for practicality
            value=(0, min(500, max_words))
        )
    
    # Filter data based on selections
    filtered_df = df.copy()
    
    if 'publication_year' in df.columns:
        filtered_df = filtered_df[
            (filtered_df['publication_year'] >= year_range[0]) & 
            (filtered_df['publication_year'] <= year_range[1])
        ]
    
    if selected_journals and 'journal' in df.columns:
        filtered_df = filtered_df[filtered_df['journal'].isin(selected_journals)]
    
    if 'abstract_word_count' in df.columns:
        filtered_df = filtered_df[
            (filtered_df['abstract_word_count'] >= word_range[0]) & 
            (filtered_df['abstract_word_count'] <= word_range[1])
        ]
    
    # Main content - Metrics
    st.markdown('<h2 class="section-header">📊 Overview</h2>', 
                unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Papers", f"{len(filtered_df):,}")
    
    with col2:
        if 'publication_year' in df.columns:
            st.metric("Date Range", f"{year_range[0]} - {year_range[1]}")
        else:
            st.metric("Sample Data", "Demo")
    
    with col3:
        if 'journal' in df.columns:
            unique_journals = filtered_df['journal'].nunique()
            st.metric("Unique Journals", unique_journals)
        else:
            st.metric("Data Columns", len(df.columns))
    
    with col4:
        if 'abstract_word_count' in df.columns:
            avg_words = filtered_df['abstract_word_count'].mean()
            st.metric("Avg Abstract Length", f"{avg_words:.1f} words")
        else:
            st.metric("Records", f"{len(filtered_df):,}")
    
    # Publications over time
    st.markdown('<h2 class="section-header">📈 Publications Over Time</h2>', 
                unsafe_allow_html=True)
    
    if 'publication_year' in filtered_df.columns:
        yearly_counts = filtered_df['publication_year'].value_counts().sort_index()
        
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(yearly_counts.index, yearly_counts.values, alpha=0.7, color='#1f77b4', edgecolor='black')
        ax.set_title('Number of Publications by Year', fontsize=14, fontweight='bold')
        ax.set_xlabel('Year')
        ax.set_ylabel('Number of Publications')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value annotations on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max(yearly_counts.values)*0.01,
                   f'{int(height):,}', ha='center', va='bottom', fontweight='bold')
        
        st.pyplot(fig)
    else:
        st.info("Publication year data not available in the dataset.")
    
    # Top journals and word cloud
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<h3 class="section-header">🏥 Top Publishing Journals</h3>', 
                    unsafe_allow_html=True)
        
        if 'journal' in filtered_df.columns:
            top_n_journals = st.slider("Number of top journals to show", 5, 20, 10, key="journals_slider")
            
            journal_counts = filtered_df['journal'].value_counts().head(top_n_journals)
            
            if len(journal_counts) > 0:
                fig, ax = plt.subplots(figsize=(10, 6))
                bars = ax.barh(range(len(journal_counts)), journal_counts.values, color='#2e86ab')
                ax.set_yticks(range(len(journal_counts)))
                ax.set_yticklabels(journal_counts.index)
                ax.set_xlabel('Number of Publications')
                ax.set_title(f'Top {top_n_journals} Journals', fontweight='bold')
                
                # Add value labels
                for i, bar in enumerate(bars):
                    width = bar.get_width()
                    ax.text(width + (max(journal_counts.values) * 0.01), 
                           bar.get_y() + bar.get_height()/2, 
                           f'{int(width):,}', ha='left', va='center', fontweight='bold')
                
                ax.invert_yaxis()
                st.pyplot(fig)
            else:
                st.info("No journal data available with current filters.")
        else:
            st.info("Journal information not available in the dataset.")
    
    with col2:
        st.markdown('<h3 class="section-header">☁️ Title Word Cloud</h3>', 
                    unsafe_allow_html=True)
        
        if 'title' in filtered_df.columns:
            wordcloud_fig = generate_word_cloud(filtered_df['title'], 'Most Frequent Words in Titles')
            if wordcloud_fig:
                st.pyplot(wordcloud_fig)
            else:
                st.info("Not enough data to generate word cloud with current filters.")
        else:
            st.info("Title information not available for word cloud.")
    
    # Abstract length distribution
    st.markdown('<h2 class="section-header">📊 Abstract Length Analysis</h2>', 
                unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if 'abstract_word_count' in filtered_df.columns:
            abstracts_with_data = filtered_df[
                (filtered_df['abstract_word_count'] > 0) & 
                (filtered_df['abstract_word_count'] < 5000)
            ]
            
            if len(abstracts_with_data) > 0:
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.hist(abstracts_with_data['abstract_word_count'], bins=30, alpha=0.7, 
                       edgecolor='black', color='skyblue')
                ax.set_xlabel('Word Count')
                ax.set_ylabel('Number of Papers')
                ax.set_title('Distribution of Abstract Word Counts', fontweight='bold')
                ax.grid(True, alpha=0.3)
                
                mean_count = abstracts_with_data['abstract_word_count'].mean()
                median_count = abstracts_with_data['abstract_word_count'].median()
                
                ax.axvline(mean_count, color='red', linestyle='--', linewidth=2, 
                          label=f'Mean: {mean_count:.1f}')
                ax.axvline(median_count, color='green', linestyle='--', linewidth=2, 
                          label=f'Median: {median_count:.1f}')
                ax.legend()
                
                st.pyplot(fig)
            else:
                st.info("No abstract length data available with current filters.")
        else:
            st.info("Abstract word count data not available.")
    
    with col2:
        st.markdown("### Abstract Statistics")
        if 'abstract_word_count' in filtered_df.columns:
            abstracts_with_data = filtered_df[filtered_df['abstract_word_count'] > 0]
            if len(abstracts_with_data) > 0:
                st.metric("Papers with Abstracts", f"{len(abstracts_with_data):,}")
                st.metric("Average Length", f"{abstracts_with_data['abstract_word_count'].mean():.1f} words")
                st.metric("Median Length", f"{abstracts_with_data['abstract_word_count'].median():.1f} words")
                st.metric("Longest Abstract", f"{abstracts_with_data['abstract_word_count'].max():,} words")
            else:
                st.info("No abstract statistics available.")
        else:
            st.info("Abstract word count data not available.")
    
    # Data sample and dataset info
    tab1, tab2 = st.tabs(["📋 Sample Data", "📊 Dataset Info"])
    
    with tab1:
        st.markdown('<h3 class="section-header">Sample Data</h3>', 
                    unsafe_allow_html=True)
        
        sample_size = st.slider("Number of sample records to show", 5, 50, 10, key="sample_slider")
        
        display_columns = []
        for col in ['title', 'journal', 'publication_year', 'authors', 'abstract_word_count']:
            if col in filtered_df.columns:
                display_columns.append(col)
        
        if display_columns:
            st.dataframe(
                filtered_df[display_columns].head(sample_size),
                use_container_width=True,
                height=400
            )
        else:
            st.dataframe(filtered_df.head(sample_size), use_container_width=True)
    
    with tab2:
        st.markdown('<h3 class="section-header">Dataset Information</h3>', 
                    unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Basic Information:**")
            st.write(f"- Total records: {len(filtered_df):,}")
            st.write(f"- Number of columns: {len(filtered_df.columns)}")
            st.write(f"- Memory usage: {filtered_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            
            if 'publication_year' in filtered_df.columns:
                st.write(f"- Year range: {int(filtered_df['publication_year'].min())} - {int(filtered_df['publication_year'].max())}")
            
            if 'journal' in filtered_df.columns:
                st.write(f"- Unique journals: {filtered_df['journal'].nunique()}")
        
        with col2:
            st.write("**Column Summary:**")
            for col in filtered_df.columns:
                non_null = filtered_df[col].count()
                null_count = len(filtered_df) - non_null
                null_pct = (null_count / len(filtered_df)) * 100
                dtype = filtered_df[col].dtype
                st.write(f"- **{col}**: {dtype} ({non_null:,} non-null, {null_pct:.1f}% missing)")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "**CORD-19 Dataset Explorer** | Created with Streamlit | "
        "COVID-19 Research Analysis Dashboard"
    )

if __name__ == "__main__":
    main()