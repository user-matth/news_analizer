import json
import pandas as pd
from textblob import TextBlob

# Load JSON data from a file
with open('news.json', 'r') as file:
    data = json.load(file)

# Extract information
news_items = data['items']
titles = [item['titulo'] for item in news_items]

# Create a DataFrame
df = pd.DataFrame({'title': titles})

# Keywords related to environment
environment_keywords = ['clima', 'ecologico', 'biodiversidade', 'conservacao', 'sustentabilidade', 'meio', 'ambiente', 'meio-ambiente']

# Filter titles for environmental content
df['environmental'] = df['title'].apply(lambda x: any(keyword in x.lower() for keyword in environment_keywords))

# Analyze the length of titles
df['title_length'] = df['title'].apply(len)

# Add sentiment analysis
df['sentiment'] = df['title'].apply(lambda x: TextBlob(x).sentiment.polarity)
df['sentiment_category'] = df['sentiment'].apply(lambda x: 'good' if x > 0 else ('bad' if x < 0 else 'neutral'))

# Convert the DataFrame to JSON
result_json = df[df['environmental']].to_json(orient='records')

# Print the JSON string
print(result_json)

# Optionally, save the JSON data to a file
with open('sentiment_analysis_results.json', 'w') as json_file:
    json_file.write(result_json)
