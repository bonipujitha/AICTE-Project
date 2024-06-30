import nltk
import pandas as pd
import streamlit as st
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob  # For multi-language support
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
import datetime

# Download necessary NLTK data
nltk.download('vader_lexicon', quiet=True)

class SentimentAnalyzer:
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()

    def analyze_sentiment(self, text, language='english'):
        if language != 'english':
            # Use TextBlob for multi-language sentiment analysis
            blob = TextBlob(text)
            sentiment = blob.sentiment.polarity
            if sentiment > 0:
                return 'positive'
            elif sentiment < 0:
                return 'negative'
            else:
                return 'neutral'
        else:
            # Use NLTK's VADER for English sentiment analysis
            sentiment_score = self.sia.polarity_scores(text)
            if sentiment_score['compound'] >= 0.05:
                return 'positive'
            elif sentiment_score['compound'] <= -0.05:
                return 'negative'
            else:
                return 'neutral'

    def analyze_file(self, df, text_column, language='english'):
        with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count() * 2) as executor:
            chunksize = max(1, len(df) // (multiprocessing.cpu_count() * 4))
            results = list(executor.map(lambda text: self.analyze_sentiment(text, language), df[text_column], chunksize=chunksize))
        df['sentiment'] = results
        return df

# Streamlit app
def main():
    st.title("Sentiment Analysis App")

    analyzer = SentimentAnalyzer()

    # File upload
    st.header("Analyze Sentiments from CSV")
    uploaded_file = st.file_uploader("Choose a CSV file (up to 200MB)", type="csv")
    if uploaded_file is not None:
        try:
            # Read CSV file
            data = pd.read_csv(uploaded_file)

            # Check if 'Text' column exists
            text_column = 'Text'  # Adjust based on your column name
            if text_column not in data.columns:
                st.error(f"The CSV file must contain a '{text_column}' column.")
                return

            # Display first few rows of the data
            st.write("First few rows of your data:")
            st.write(data.head())

            # Language selection for sentiment analysis
            language = st.selectbox("Select Language for Sentiment Analysis", options=['English', 'Other'])

            # Perform sentiment analysis
            if st.button("Analyze Sentiments"):
                with st.spinner('Analyzing sentiments... This may take a while for large files.'):
                    result_df = analyzer.analyze_file(data, text_column, language.lower())
                st.success("Sentiment analysis completed!")

                # Display results
                st.subheader("Results:")
                st.write(result_df)

                # Show sentiment distribution
                sentiment_counts = result_df['sentiment'].value_counts()
                st.subheader("Sentiment Distribution:")
                st.bar_chart(sentiment_counts)

                # Export results to CSV
                if st.button("Export Results as CSV"):
                    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    csv_filename = f"sentiment_analysis_results_{timestamp}.csv"
                    csv = result_df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name=csv_filename,
                        mime="text/csv"
                    )

        except pd.errors.EmptyDataError:
            st.error("The uploaded file is empty. Please upload a valid CSV file.")
        except Exception as e:
            st.error(f"Error reading or processing the file: {str(e)}")

if __name__ == "__main__":
    main()







