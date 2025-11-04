# data_exploration.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
import os

warnings.filterwarnings('ignore')

class CORD19Analyzer:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        
    def load_data(self):
        """Load the metadata.csv file"""
        try:
            # Check if file exists
            if not os.path.exists(self.file_path):
                print(f"File not found: {self.file_path}")
                print("Please make sure the file exists or download it from:")
                print("https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge")
                return False
                
            self.df = pd.read_csv(self.file_path)
            print(f"Data loaded successfully! Shape: {self.df.shape}")
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def basic_exploration(self):
        """Perform basic data exploration"""
        print("=== BASIC DATA EXPLORATION ===")
        
        # DataFrame dimensions
        print(f"Dataset shape: {self.df.shape}")
        print(f"Number of rows: {self.df.shape[0]:,}")
        print(f"Number of columns: {self.df.shape[1]}")
        
        # Column names
        print("\n=== COLUMN NAMES ===")
        print(list(self.df.columns))
        
        # First few rows
        print("\n=== FIRST 5 ROWS ===")
        print(self.df.head())
        
        # Data types
        print("\n=== DATA TYPES ===")
        print(self.df.dtypes)
        
    def check_missing_values(self):
        """Analyze missing values in the dataset"""
        print("\n=== MISSING VALUES ANALYSIS ===")
        
        # Total missing values per column
        missing_data = self.df.isnull().sum()
        missing_percent = (missing_data / len(self.df)) * 100
        
        missing_df = pd.DataFrame({
            'Column': missing_data.index,
            'Missing_Count': missing_data.values,
            'Missing_Percent': missing_percent.values
        })
        
        missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values('Missing_Percent', ascending=False)
        
        print("Columns with missing values:")
        print(missing_df.to_string())
        
        # Plot missing values for top 10 columns
        if len(missing_df) > 0:
            plt.figure(figsize=(12, 6))
            top_missing = missing_df[missing_df['Missing_Percent'] > 0].head(10)
            plt.bar(top_missing['Column'], top_missing['Missing_Percent'])
            plt.title('Top 10 Columns with Highest Missing Values Percentage', fontsize=14)
            plt.xlabel('Columns')
            plt.ylabel('Missing Values (%)')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig('missing_values.png', dpi=300, bbox_inches='tight')
            plt.show()
        else:
            print("No missing values found in the dataset.")
        
        return missing_df

# Main execution
if __name__ == "__main__":
    # Try different possible file paths
    possible_paths = [
        'metadata.csv',
        '../metadata.csv',
        './metadata.csv',
        'CORD-19-research-challenge/metadata.csv'
    ]
    
    file_path = None
    for path in possible_paths:
        if os.path.exists(path):
            file_path = path
            break
    
    if file_path is None:
        print("metadata.csv not found in common locations.")
        print("Please specify the full path to your metadata.csv file:")
        file_path = input("Enter file path: ").strip().strip('"')
    
    # Initialize analyzer
    analyzer = CORD19Analyzer(file_path)
    
    # Load data
    if analyzer.load_data():
        # Perform exploration
        analyzer.basic_exploration()
        missing_df = analyzer.check_missing_values()
        
        # Save basic info for next steps
        analyzer.df.to_csv('raw_metadata_sample.csv', index=False)
        print("\nRaw data sample saved to 'raw_metadata_sample.csv'")
    else:
        print("\nFailed to load data. Please check the file path and try again.")

# data_cleaning.py
import pandas as pd
import numpy as np
import os

class CORD19Cleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.cleaned_df = None
        
    def clean_data(self):
        """Perform comprehensive data cleaning"""
        print("Starting data cleaning process...")
        print(f"Original data shape: {self.df.shape}")
        
        # Create a cleaned copy
        self.cleaned_df = self.df.copy()
        
        # Handle publication dates
        self._clean_dates()
        
        # Handle missing values
        self._handle_missing_values()
        
        # Create new features
        self._create_new_features()
        
        print(f"Cleaned data shape: {self.cleaned_df.shape}")
        print("Data cleaning completed!")
        return self.cleaned_df
    
    def _clean_dates(self):
        """Clean and convert date columns"""
        print("Cleaning date columns...")
        
        # Check if publish_time column exists
        if 'publish_time' in self.cleaned_df.columns:
            # Convert publish_time to datetime
            self.cleaned_df['publish_time'] = pd.to_datetime(
                self.cleaned_df['publish_time'], errors='coerce'
            )
            
            # Extract year, month
            self.cleaned_df['publication_year'] = self.cleaned_df['publish_time'].dt.year
            self.cleaned_df['publication_month'] = self.cleaned_df['publish_time'].dt.month
            
            # Fill missing years with mode or most recent year
            if not self.cleaned_df['publication_year'].isnull().all():
                mode_year = self.cleaned_df['publication_year'].mode()
                if len(mode_year) > 0:
                    self.cleaned_df['publication_year'] = self.cleaned_df['publication_year'].fillna(mode_year[0])
                else:
                    self.cleaned_df['publication_year'] = self.cleaned_df['publication_year'].fillna(2020)
            else:
                self.cleaned_df['publication_year'] = 2020
        else:
            print("Warning: 'publish_time' column not found. Creating default year column.")
            self.cleaned_df['publication_year'] = 2020
        
    def _handle_missing_values(self):
        """Handle missing values in important columns"""
        print("Handling missing values...")
        
        # For text columns, fill with placeholder
        text_columns = ['title', 'abstract', 'journal', 'authors']
        for col in text_columns:
            if col in self.cleaned_df.columns:
                self.cleaned_df[col] = self.cleaned_df[col].fillna(f'Unknown {col}')
                print(f" - Filled missing values in '{col}'")
        
        # Handle other important columns
        if 'source_x' in self.cleaned_df.columns:
            self.cleaned_df['source_x'] = self.cleaned_df['source_x'].fillna('Unknown Source')
            
    def _create_new_features(self):
        """Create new features for analysis"""
        print("Creating new features...")
        
        # Abstract word count
        if 'abstract' in self.cleaned_df.columns:
            self.cleaned_df['abstract_word_count'] = self.cleaned_df['abstract'].apply(
                lambda x: len(str(x).split()) if pd.notnull(x) and x != 'Unknown abstract' else 0
            )
            print(" - Created 'abstract_word_count'")
        
        # Title word count
        if 'title' in self.cleaned_df.columns:
            self.cleaned_df['title_word_count'] = self.cleaned_df['title'].apply(
                lambda x: len(str(x).split()) if pd.notnull(x) and x != 'Unknown title' else 0
            )
            print(" - Created 'title_word_count'")
        
        # Has abstract flag
        if 'abstract' in self.cleaned_df.columns:
            self.cleaned_df['has_abstract'] = ~(
                (self.cleaned_df['abstract'].isnull()) | 
                (self.cleaned_df['abstract'] == 'Unknown abstract')
            )
            print(" - Created 'has_abstract'")
        
        # Paper ID
        if 'cord_uid' in self.cleaned_df.columns:
            self.cleaned_df['paper_id'] = self.cleaned_df['cord_uid']
        else:
            self.cleaned_df['paper_id'] = range(1, len(self.cleaned_df) + 1)
        print(" - Created 'paper_id'")

# Example usage
if __name__ == "__main__":
    # Try to load the raw data sample first, then original file
    file_path = None
    possible_files = ['raw_metadata_sample.csv', 'metadata.csv']
    
    for file in possible_files:
        if os.path.exists(file):
            file_path = file
            break
    
    if file_path is None:
        print("No data file found. Please run data_exploration.py first or ensure metadata.csv exists.")
        exit()
    
    print(f"Loading data from: {file_path}")
    original_df = pd.read_csv(file_path)
    
    # Clean data
    cleaner = CORD19Cleaner(original_df)
    cleaned_df = cleaner.clean_data()
    
    # Save cleaned data
    cleaned_df.to_csv('cleaned_metadata.csv', index=False)
    print("Cleaned data saved to 'cleaned_metadata.csv'")
    
    # Show basic info about cleaned data
    print(f"\nCleaned dataset info:")
    print(f" - Total records: {len(cleaned_df):,}")
    print(f" - Columns: {len(cleaned_df.columns)}")
    if 'publication_year' in cleaned_df.columns:
        print(f" - Publication years: {sorted(cleaned_df['publication_year'].unique())}")

# analysis_visualization.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import re
import os

# Set style
plt.style.use('default')
sns.set_palette("husl")

class CORD19Visualizer:
    def __init__(self, df):
        self.df = df
        
    def plot_publications_over_time(self):
        """Plot number of publications over time"""
        if 'publication_year' not in self.df.columns:
            print("Warning: 'publication_year' column not found. Skipping time series plot.")
            return
            
        plt.figure(figsize=(12, 6))
        
        # Group by year and count
        yearly_counts = self.df['publication_year'].value_counts().sort_index()
        
        # Filter out any NaN years and convert to int for plotting
        yearly_counts = yearly_counts[~yearly_counts.index.isna()]
        yearly_counts.index = yearly_counts.index.astype(int)
        
        if len(yearly_counts) > 0:
            plt.bar(yearly_counts.index, yearly_counts.values, alpha=0.7, edgecolor='black')
            plt.title('COVID-19 Research Publications Over Time', fontsize=16, fontweight='bold')
            plt.xlabel('Publication Year', fontsize=12)
            plt.ylabel('Number of Publications', fontsize=12)
            plt.grid(True, alpha=0.3, axis='y')
            
            # Add value annotations on bars
            for i, (x, y) in enumerate(zip(yearly_counts.index, yearly_counts.values)):
                plt.annotate(f'{y:,}', (x, y), textcoords="offset points", 
                            xytext=(0,5), ha='center', fontweight='bold', fontsize=9)
            
            plt.tight_layout()
            plt.savefig('publications_over_time.png', dpi=300, bbox_inches='tight')
            plt.show()
        else:
            print("No publication year data available for plotting.")
        
    def plot_top_journals(self, top_n=15):
        """Plot top publishing journals"""
        if 'journal' not in self.df.columns:
            print("Warning: 'journal' column not found. Skipping journals plot.")
            return
            
        plt.figure(figsize=(12, 8))
        
        # Get top journals (exclude 'Unknown journal')
        journal_counts = self.df['journal'].value_counts()
        top_journals = journal_counts[journal_counts.index != 'Unknown journal'].head(top_n)
        
        if len(top_journals) > 0:
            # Create horizontal bar chart
            bars = plt.barh(range(len(top_journals)), top_journals.values)
            plt.yticks(range(len(top_journals)), top_journals.index)
            plt.title(f'Top {len(top_journals)} Journals Publishing COVID-19 Research', 
                     fontsize=16, fontweight='bold')
            plt.xlabel('Number of Publications', fontsize=12)
            
            # Add value labels on bars
            for i, bar in enumerate(bars):
                width = bar.get_width()
                plt.text(width + (max(top_journals.values) * 0.01), 
                        bar.get_y() + bar.get_height()/2, 
                        f'{int(width):,}', ha='left', va='center', fontweight='bold')
            
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig('top_journals.png', dpi=300, bbox_inches='tight')
            plt.show()
        else:
            print("No journal data available for plotting.")
        
    def generate_word_cloud(self, text_column='title'):
        """Generate word cloud from text data"""
        if text_column not in self.df.columns:
            print(f"Warning: '{text_column}' column not found. Skipping word cloud.")
            return
            
        # Combine all text (exclude placeholder values)
        valid_text = self.df[text_column][
            (~self.df[text_column].isnull()) & 
            (self.df[text_column] != f'Unknown {text_column}')
        ]
        
        if len(valid_text) == 0:
            print(f"No valid text data in '{text_column}' for word cloud.")
            return
            
        all_text = ' '.join(valid_text.astype(str))
        
        # Clean text (remove special characters, single letters)
        words = re.findall(r'\b[a-zA-Z]{3,}\b', all_text.lower())
        
        # Remove common stop words
        stop_words = {'the', 'and', 'of', 'in', 'to', 'a', 'for', 'with', 'on', 'by', 
                     'as', 'an', 'from', 'that', 'this', 'is', 'are', 'was', 'were', 
                     'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'which',
                     'who', 'what', 'when', 'where', 'why', 'how', 'their', 'there'}
        
        filtered_words = [word for word in words if word not in stop_words]
        
        if len(filtered_words) > 0:
            # Create word cloud
            wordcloud = WordCloud(
                width=800, 
                height=400, 
                background_color='white',
                max_words=100,
                colormap='viridis'
            ).generate(' '.join(filtered_words))
            
            plt.figure(figsize=(12, 6))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title(f'Word Cloud of Paper {text_column.capitalize()}', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig(f'wordcloud_{text_column}.png', dpi=300, bbox_inches='tight')
            plt.show()
        else:
            print(f"Not enough words to generate word cloud for '{text_column}'.")
        
    def plot_abstract_length_distribution(self):
        """Plot distribution of abstract lengths"""
        if 'abstract_word_count' not in self.df.columns:
            print("Warning: 'abstract_word_count' column not found. Skipping abstract length plot.")
            return
            
        plt.figure(figsize=(12, 6))
        
        # Filter out papers without abstracts
        abstracts_with_data = self.df[
            (self.df['abstract_word_count'] > 0) & 
            (self.df['abstract_word_count'] < 10000)  # Remove outliers
        ]
        
        if len(abstracts_with_data) > 0:
            plt.hist(abstracts_with_data['abstract_word_count'], bins=50, alpha=0.7, 
                    edgecolor='black', color='skyblue')
            plt.title('Distribution of Abstract Word Counts', fontsize=16, fontweight='bold')
            plt.xlabel('Word Count', fontsize=12)
            plt.ylabel('Number of Papers', fontsize=12)
            plt.grid(True, alpha=0.3)
            
            # Add statistics
            mean_count = abstracts_with_data['abstract_word_count'].mean()
            median_count = abstracts_with_data['abstract_word_count'].median()
            
            plt.axvline(mean_count, color='red', linestyle='--', linewidth=2, 
                       label=f'Mean: {mean_count:.1f}')
            plt.axvline(median_count, color='green', linestyle='--', linewidth=2, 
                       label=f'Median: {median_count:.1f}')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig('abstract_length_distribution.png', dpi=300, bbox_inches='tight')
            plt.show()
        else:
            print("No abstract length data available for plotting.")

# Main analysis script
if __name__ == "__main__":
    # Load cleaned data
    if not os.path.exists('cleaned_metadata.csv'):
        print("Cleaned data file not found. Please run data_cleaning.py first.")
        exit()
    
    df = pd.read_csv('cleaned_metadata.csv')
    print(f"Loaded cleaned data with {len(df):,} records")
    
    # Initialize visualizer
    visualizer = CORD19Visualizer(df)
    
    # Create all visualizations
    print("Creating visualizations...")
    visualizer.plot_publications_over_time()
    visualizer.plot_top_journals()
    visualizer.generate_word_cloud('title')
    visualizer.plot_abstract_length_distribution()
    
    print("All visualizations completed!")