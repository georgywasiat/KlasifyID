from flask import Flask, request, render_template
import joblib
import nltk
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import string
import re

# Download necessary NLTK data
nltk.download('stopwords')

# Load the necessary models and transformers
count_vectorizer = joblib.load('C:\\Users\\M S I\\Documents\\Semester 7\\UI SKRIPSI 2\\count_vectorizer.joblib')
tfidf_transformer = joblib.load('C:\\Users\\M S I\\Documents\\Semester 7\\UI SKRIPSI 2\\tfidf_transformer.joblib')
svm_model_linear = joblib.load('C:\\Users\\M S I\\Documents\\Semester 7\\UI SKRIPSI 2\\svm_linear_model.joblib')
svm_model_rbf = joblib.load('C:\\Users\\M S I\\Documents\\Semester 7\\UI SKRIPSI 2\\svm_rbf_model.joblib')
svm_model_poly = joblib.load('C:\\Users\\M S I\\Documents\\Semester 7\\UI SKRIPSI 2\\svm_poly_model.joblib')


# Initialize the Flask app
app = Flask(__name__)

# Initialize Indonesian stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Define a function for text preprocessing (Cleaning, case folding, stopword removal, and stemming)
def preprocess_text(text):
    # Cleaning
    text = text.replace('\r\n\r\n', ' ').replace('\n\n', ' ')
    for punctuation in string.punctuation:
        text = text.replace(punctuation, f' {punctuation} ')
    text = ' '.join(text.split())
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = ' '.join(word for word in text.split() if len(word) > 2)
    
    # Case folding (lowercasing)
    text = text.lower()
    
    # Stopwords removal (after cleaning)
    stop_words = set(stopwords.words('indonesian'))
    filtered_text = ' '.join(word for word in text.split() if word.lower() not in stop_words)
    
    # Stemming (after stopword removal)
    stemmed_text = ' '.join([stemmer.stem(word) for word in filtered_text.split()])
    
    return stemmed_text  # Return processed text

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    user_input = ""
    selected_kernel = "linear"
    
    if request.method == 'POST':
        # Retrieve input from the form
        user_input = request.form.get('text', '')
        selected_kernel = request.form.get('kernel', 'linear')  # Get the selected kernel
        
        # Preprocess the input text
        processed_text = preprocess_text(user_input)  # Process the input text
        
        # Vectorize and transform the input text
        text_counts = count_vectorizer.transform([processed_text])
        text_tfidf = tfidf_transformer.transform(text_counts)
        
        # Use the appropriate model based on the selected kernel
        if selected_kernel == 'linear':
            prediction = svm_model_linear.predict(text_tfidf)[0]
        elif selected_kernel == 'rbf':
            prediction = svm_model_rbf.predict(text_tfidf)[0]
        elif selected_kernel == 'poly':
            prediction = svm_model_poly.predict(text_tfidf)[0]

    return render_template('index.html', prediction=prediction, user_input=user_input, selected_kernel=selected_kernel)

# Run the app
if __name__ == '__main__':
    app.run(debug=True, port=5007)
