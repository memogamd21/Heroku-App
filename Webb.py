from werkzeug.utils import secure_filename
import sqlite3
import tensorflow as tf
import os
import numpy as np
from numpy import array
import pandas as pd
import matplotlib.pyplot as plt
#%matplotlib inline
import string
import os
from PIL import Image
import glob
from pickle import dump, load
from time import time
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from keras.layers import LSTM, Embedding, TimeDistributed, Dense, RepeatVector,\
                         Activation, Flatten, Reshape, concatenate, Dropout, BatchNormalization
from keras.optimizers import Adam, RMSprop
from keras.layers.wrappers import Bidirectional
from keras.layers.merge import add
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image as IMG
from keras.models import Model
from keras import Input, layers
from keras import optimizers
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from flask import Flask, redirect, url_for, render_template, request, session, flash, send_from_directory
from datetime import timedelta
import math
from collections import Counter
import operator
import webbrowser

global graph
graph = tf.get_default_graph()
import tensorflow.python.util.deprecation as deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
app = Flask(__name__, static_folder = "IMAGE_UPLOADS")
#app.secret_key = "memo"
#app.permanent_session_lifetime = timedelta(days=5)
app.config["IMAGE_UPLOADS"] = "IMAGE_UPLOADS"
app.config["ALLOWED_IMAGE_EXTENSIONS"] = ["JPEG", "JPG", "PNG", "GIF"]

######################################################### Processing/Helper Functions ##################################################
def preprocess(image_path):
    # Convert all the images to size 299x299 as expected by the inception v3 model
    img = IMG.load_img(image_path, target_size=(299, 299))
    # Convert PIL image to numpy array of 3-dimensions
    x = IMG.img_to_array(img)
    # Add one more dimension
    x = np.expand_dims(x, axis=0)
    # preprocess the images using preprocess_input() from inception module
    x = preprocess_input(x)
    return x

def encode(image):
    with graph.as_default():
        image = preprocess(image) # preprocess the image
        fea_vec = model_new.predict(image) # Get the encoding vector for the image
        fea_vec = np.reshape(fea_vec, fea_vec.shape[1]) # reshape from (1, 2048) to (2048, )
    return fea_vec


def allowed_image(filename):

    if not "." in filename:
        return False

    ext = filename.rsplit(".", 1)[1]

    if ext.upper() in app.config["ALLOWED_IMAGE_EXTENSIONS"]:
        return True
    else:
        return False

    
model = InceptionV3(weights='imagenet')
model_new = Model(model.input, model.layers[-2].output)
ixtoword = load(open('ixtoword.pkl', 'rb'))
wordtoix = load(open('wordtoix.pkl', 'rb'))
#embedding_matrix = load(open('embedding_matrix.pkl', 'rb'))
#embeddings_index = load(open('embeddings_index.pkl', 'rb'))
#train_features = load(open("encoded_train_images.pkl", "rb"))
max_length = 34
vocab_size = 1652
embedding_dim = 200

################### Deep Learning Model ###################################################
inputs1 = Input(shape=(2048,))
fe1 = Dropout(0.5)(inputs1)
fe2 = Dense(256, activation='relu')(fe1)
inputs2 = Input(shape=(max_length,))
se1 = Embedding(vocab_size, embedding_dim, mask_zero=True)(inputs2)
se2 = Dropout(0.5)(se1)
se3 = LSTM(256)(se2)
decoder1 = add([fe2, se3])
decoder2 = Dense(256, activation='relu')(decoder1)
outputs = Dense(vocab_size, activation='softmax')(decoder2)
model = Model(inputs=[inputs1, inputs2], outputs=outputs)

model.load_weights(r'C:\Users\Lenovo\Desktop\GraduationProject\moddell_7602.h5')

model.compile(loss='categorical_crossentropy', optimizer='adam')

def greedysearch(photo):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = [wordtoix[w] for w in in_text.split() if w in wordtoix]
        sequence = pad_sequences([sequence], maxlen=max_length)
        z = encode(photo)
        with graph.as_default():
            yhat = model.predict([z.reshape((1,2048)),sequence], verbose=0)
            yhat = np.argmax(yhat)
            word = ixtoword[yhat]
        in_text += ' ' + word
        if word == 'endseq':
            break
    final = in_text.split()
    final = final[1:-1]
    final = ' '.join(final)
    return final
    
########################################################## Search Engine Functions and Variables #######################################    
inverted_index = load(open('inverted_index.pkl', 'rb'))
img_dict = load(open('img_dict.pkl', 'rb'))

df = {}
idf = {}
for key in inverted_index.keys():
    df[key] = len(inverted_index[key].keys())
    idf[key] = math.log(len(img_dict.values()) / df[key], 2)
    
def tf_idf(w, doc):
    return inverted_index[w][doc] * idf[w]

def inner_product_similarities(query):
    # dictionary in which I'll sum up the similarities of each word of the query with each document in
    # which the word is present, key is the doc number,
    # value is the similarity between query and document
    similarity = {} 
    for w in str(query).split(' '):
        wq = idf.get(w, 0)
        if wq != 0:
            for doc in inverted_index[w].keys():
                similarity[doc] = similarity.get(doc, 0) + tf_idf(w, doc) * wq
    return similarity

def doc_length(userid):
    words_accounted_for = []
    length = 0
    for w in img_dict[userid].split(' '):
        if w not in words_accounted_for:
            length += tf_idf(w, userid) ** 2
            words_accounted_for.append(w)
    return math.sqrt(length)

def query_length(query):
    # IMPORTANT: in this HW no query has repeated words, so I can skip the term frequency calculation
    # for the query, and just use idfs quared
    length = 0
    cnt = Counter()
    for w in str(query).split(' '):
        cnt[w] += 1
    for w in cnt.keys():
        length += (cnt[w]*idf.get(w, 0)) ** 2
    return math.sqrt(length) 

def cosine_similarities(query):
    similarity = inner_product_similarities(query)
    for doc in similarity.keys():
        similarity[doc] = similarity[doc] / doc_length(doc) / query_length(query)
    return similarity

def rank_docs(similarities):
    return sorted(similarities.items(), key=operator.itemgetter(1), reverse=True)


########################################################## Website Functions############################################################

@app.route("/")
def index():
    return render_template("upload_image.html")
    
@app.route('/search-image', methods=["GET", "POST"])
def search_image():
    if request.method == "POST":
        Query = request.form["Query"]
        session["Query"] = Query
        return redirect(url_for("Searchresults"))
    else:
#         if "Query" in session:
#             return redirect(url_for("Search_results"))
        return render_template('Query.html')
    
@app.route('/Searchresults', methods=["GET", "POST"])
def Searchresults():
    Query = session["Query"]
    query_tokens = Query
    ranked_similarities = rank_docs(cosine_similarities(query_tokens))
    image_res_paths = []
    image_captions = []
    for i in range(len(ranked_similarities)):
        image_res_paths.append(ranked_similarities[i][0])
        image_captions.append(img_dict[ranked_similarities[i][0]])
    #return redirect(url_for("Search_results"))
    lenth = len(image_captions)
    img_res = image_res_paths
    img_captions = image_captions
    return render_template("display_images.html", data = [img_res, img_captions , lenth])
    
@app.route("/upload-image", methods=["POST"])
def upload_image(): 
    if request.files:
        image = request.files["image"]
        filename = secure_filename(image.filename)
        image.save(os.path.join(app.config["IMAGE_UPLOADS"], filename))
        img_path = os.path.join(app.config["IMAGE_UPLOADS"], filename)
        caption = greedysearch(img_path)
        data = [caption, filename]
        return render_template("disp.html", dt = data)
    else:
        return redirect(url_for("index"))

@app.route('/upload/<filename>')
def send_image(filename):
    return send_from_directory("IMAGE_UPLOADS", filename)

if __name__ == '__main__':
    app.run(debug=True)
