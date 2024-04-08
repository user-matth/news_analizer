from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Sample dataset
titles = ['Climate change is accelerating', 'New policy leads to environmental improvements', ...]
sentiments = ['negative', 'positive', ...]  # Your manually labeled sentiments

# Convert titles to a matrix of token counts
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(titles)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, sentiments, test_size=0.25, random_state=42)

# Create and train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict on the test set
predictions = model.predict(X_test)

# Evaluate the model
print(classification_report(y_test, predictions))
