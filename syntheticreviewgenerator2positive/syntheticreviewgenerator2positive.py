import pandas as pd
from transformers import pipeline

dataset_path = 'synthetic_amazon_reviews.csv'


try:
    df = pd.read_csv(dataset_path)
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print(f"Error: The file at {dataset_path} was not found.")
    exit()
except pd.errors.ParserError:
    print(f"Error: Could not parse the CSV file at {dataset_path}.")
    exit()
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    exit()

# Initialize sentiment analysis pipeline
model_name = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
sentiment_analyzer = pipeline('sentiment-analysis')

# Function to determine if the review is positive
def is_positive_review(review_text):
    result = sentiment_analyzer(review_text)[0]
    return result['label'] == 'POSITIVE' and result['score'] > 0.90  # Use a threshold

# Filter for positive reviews
positive_reviews = df[df['text'].apply(is_positive_review)]

# Create a new DataFrame for positive reviews
positive_reviews_df = positive_reviews[['rating', 'title', 'text']]

# Save the positive reviews to a new CSV file
output_path = 'positive_amazon_reviews.csv'
positive_reviews_df.to_csv(output_path, index=False)

print(f"Filtered dataset containing positive reviews saved to '{output_path}'")

