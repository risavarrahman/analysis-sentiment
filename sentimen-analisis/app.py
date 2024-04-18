from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import pandas as pd
import numpy as np
import os, re, string, sys
import pickle
import ast
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

app = Flask(__name__)

model = pickle.load(open('svm_rbf_v2.pkl','rb'))
vectorizer = pickle.load(open('vectorizer_v2.pkl','rb'))


def cleanFileInput(text):
	emoji_pattern = re.compile("["
		"\U0001F600-\U0001F64F"
		"\U0001F300-\U0001F5FF"
		"\U0001F680-\U0001F6FF"
		"\U0001F700-\U0001F77F"
		"\U0001F780-\U0001F7FF"
		"\U0001F800-\U0001F8FF"
		"\U0001F900-\U0001F9FF"
		"\U0001FA00-\U0001FA6F"
		"\U0001FA70-\U0001FAFF"
		"\U00002702-\U000027B0"
		"\U000024C2-\U0001F251"
		"]+", flags=re.UNICODE)
	text = emoji_pattern.sub(r' ', text)

	text = text.lower()
	text = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', ' ', str(text))
	text = re.sub(r'[^\w\s]', ' ', text)
	text = re.sub(r'\d+', ' ', text)
	text = re.sub(r'([a-z])\1{2,}', r'\1', text)
	text = re.sub('[\s]+', ' ', text)
	text = re.sub(r'^\s*\n', '', text, flags=re.MULTILINE)
	text = re.sub(' +', ' ', text)
	return text


def cleanNormalisasi(text):
    dict_koreksi = {}
    with open("./static/files/update_combined_new_slang_words.txt", "r") as file:
        for line_number, line in enumerate(file, start=1):
            # Check if the line has the expected format
            if ":" in line:
                f = line.split(":")
                if len(f) == 2:
                    dict_koreksi.update({f[0].strip(): f[1].strip()})
                else:
                    print(f"Error in line {line_number}: {line.strip()}")

    for awal, pengganti in dict_koreksi.items():
        #text = str(text).replace(awal, pengganti)
        text = re.sub(r'\b' + awal + r'\b', pengganti, text)
    
    return text


def cleanStopword(text):
	stopword_factory = StopWordRemoverFactory()
	stopword_sastrawi = stopword_factory.get_stop_words()
	stopword_nltk = set(stopwords.words("indonesian"))
	with open("./static/files/new_list_stopword.txt", "r") as file:
		stopwords_tambahan = file.read().splitlines()

	text = text.split()
	text = [w for w in text if w not in stopword_sastrawi]
	text = [w for w in text if w not in stopword_nltk]
	text = [w for w in text if w not in stopwords_tambahan]
	text = " ".join(w for w in text)

	return text


def cleanStemming(text):
	stemmer_factory = StemmerFactory()
	stemmer_sastrawi = stemmer_factory.create_stemmer()
	text = stemmer_sastrawi.stem(text)
	
	return text


@app.route("/")
def main():
	return render_template('index2.html')

@app.route("/visualisasi")
def visualisasi():
	return render_template('visualisasi.html')

@app.route('/predict', methods=['POST'])
def predict():
    
	if request.method == 'POST':
		Sentiment = request.form['Sentiment']
		data = [Sentiment]
		vect = vectorizer.transform(data)
		prediction = model.predict(vect)
	return render_template('index2.html', Sentiment = Sentiment, prediction = prediction)

@app.route('/klasifikasi', methods=['POST'])
def klasifikasi():
	if request.method == 'POST':
		uploaded_file = request.files['fileUpload']
		filename = secure_filename(uploaded_file.filename)
		uploaded_file.save(os.path.join("static/files/", filename))
		df_text = pd.read_excel("static/files/" + filename)
		df_text['Text'] = df_text['Text'].fillna('')


		# df_text["Text"] = cleanFileInput(df_text["Text"])
		df_text["Text"] = df_text["Text"].apply(cleanFileInput)
		
		df_text["Text"] = df_text["Text"].apply(cleanNormalisasi)

		df_text["Text"] = df_text["Text"].apply(cleanStopword)
		
		df_text["Text"] = df_text["Text"].apply(cleanStemming)

		
		vect = vectorizer.transform(df_text["Text"])
		predict = model.predict(vect)

		df_text["Prediksi"] = predict

		df_text.to_excel("static/files/" + "batch_" + filename, index=False)
		
	return render_template('index2.html', filename = filename)

if __name__ == '__main__':
	app.run(port=5000, debug=True)
	
@app.route("/index")
def main2():
	return render_template('index.html')