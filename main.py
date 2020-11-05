import json
import pickle
import numpy as np
import keras
from keras.models import load_model
import tensorflow as tf

from flask import Flask, request, abort, jsonify

import helper

flask_app = Flask(__name__)


import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

from keras.models import load_model
#path_to_model=tf.keras.utils.get_file('chatbot_model.h5', origin='https://drive.google.com/uc?id=1RJEfE3Cz4QmDg2epRq9dbW0QsWL0YMIW&export=download')
model = tf.keras.models.load_model('chatbot_model.h5')
import json
import random
#path_to_intents=tf.keras.utils.get_file('intents.json', origin='https://drive.google.com/uc?id=1uo5RuJZRNMI87bZ8cSc7D5gIH4YGoqhA&export=download')
intents = json.loads(open('intents1.json',encoding='utf-8').read())
#path_to_words=tf.keras.utils.get_file('words.pkl', origin='https://drive.google.com/uc?id=1YCk-M31yCEUf8gEtVwk_jhn06Ub7EjfV&export=download')
words = pickle.load(open('words.pkl','rb'))
#path_to_class=tf.keras.utils.get_file('classes.pkl', origin='https://drive.google.com/uc?id=1UXRdDwVka80Geo4Z6AKTOTzRt1RDgxQj&export=download')
classes = pickle.load(open('classes.pkl','rb'))

def clean_up_sentence(sentence):
	# tokenize the pattern - splitting words into array
	sentence_words = nltk.word_tokenize(sentence)
	# stemming every word - reducing to base form
	sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
	return sentence_words
# return bag of words array: 0 or 1 for words that exist in sentence

def bag_of_words(sentence, words, show_details=True):
	# tokenizing patterns
	sentence_words = clean_up_sentence(sentence)
	# bag of words - vocabulary matrix
	bag = [0]*len(words)  
	for s in sentence_words:
		for i,word in enumerate(words):
			if word == s: 
				# assign 1 if current word is in the vocabulary position
				bag[i] = 1
				if show_details:
					print ("found in bag: %s" % word)
	return(np.array(bag))
	
def predict_class(sentence):
	# filter below  threshold predictions
	p = bag_of_words(sentence, words,show_details=False)
	res = model.predict(np.array([p]))[0]
	ERROR_THRESHOLD = 0.25
	results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
	# sorting strength probability
	results.sort(key=lambda x: x[1], reverse=True)
	return_list = []
	for r in results:
		return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
	return return_list
def getResponse(ints, intents_json):
	tag = ints[0]['intent']
	list_of_intents = intents_json['intents']
	for i in list_of_intents:
		if(i['tag']== tag):
			result = random.choice(i['responses'])
			break
	return result
	
def createAnswer(sentence):
    return getResponse(predict_class(sentence), intents)
	
@flask_app.route('/predict', methods=['POST'])
def create_answer():
		message = request.args['message']
		#message=[message]
		res = createAnswer(message)
		result = {
		"error" : "0",
		"message" : "success",
		"answer" : res}
		return flask_app.response_class(response=json.dumps(result), mimetype='application/json')
	   
	
	
if __name__ == "__main__":
	 flask_app.run(host='0.0.0.0', port=5000, debug=True)
