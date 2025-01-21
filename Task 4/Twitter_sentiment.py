import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import pandas as pd
import warnings
import pickle

warnings.filterwarnings('ignore')

# Load dataset
df = pd.read_csv('./twitter_sentiment.csv', header=None, index_col=0)
df = df[[2, 3]].reset_index(drop=True)
df.rename(columns={2: "sentiments", 3: "text"}, inplace=True)

# Drop missing values
df.dropna(inplace=True)

# Preprocess text
def process_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@[a-zA-Z0-9_]+', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

df['clean_text'] = df['text'].apply(process_text)

# Vectorize text
count_vectorizer = CountVectorizer(max_features=5000)
count_matrix = count_vectorizer.fit_transform(df['clean_text'])

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    count_matrix, df['sentiments'], test_size=0.2, random_state=42
)

# Train model
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train, y_train)

# Predict and evaluate
y_pred = nb_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

with open('count_vectorizer.pkl','wb') as vectorizer_file:
    pickle.dump(count_vectorizer , vectorizer_file)

with open('nb_classifier.pkl','wb') as classifier_file:
    pickle.dump(nb_classifier , classifier_file)