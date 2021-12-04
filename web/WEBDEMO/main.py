from flask import Flask, render_template, request, redirect, url_for, send_file
import requests
from lxml import html
import joblib
from pyvi import ViTokenizer
import numpy as np
import pandas as pd
import sys, os
import requests
import csv
from bs4 import BeautifulSoup
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('english'))
from nltk.stem import PorterStemmer
import re
nltk.download('punkt')
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import pickle
sys.path.append('./')
from utils import clean_text, preprocess
# saving model
from tensorflow import keras
ln_model = pickle.load(open(os.path.join('./models/',"linear_classifier.pkl"), 'rb'))
nb_model = pickle.load(open(os.path.join('./models/',"naive_bayes.pkl"), 'rb'))
UIT_model = keras.models.load_model('./models/model_vietnam_UIT.h5')
model_vn_text = keras.models.load_model('./models/model_vietnamese_text.h5')
model = keras.models.load_model('./models/model_vietnam.h5')
model_english = keras.models.load_model('./models/model_english.h5')
# implements binary protocols for serializing and de-serializing a Python object structure.
import pickle

with open('./models/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    
    # load label encoder object
with open('./models/label_encoder.pickle', 'rb') as enc:
        lbl_encoder = pickle.load(enc)
        
with open('./models/tokenizer_UIT.pickle', 'rb') as handle:
        tokenizer_UIT = pickle.load(handle)
    
    # load label encoder object
with open('./models/label_encoder_UIT.pickle', 'rb') as enc:
        lbl_encoder_UIT = pickle.load(enc)
with open('./models/tokenizer_text.pickle', 'rb') as handle:
        tokenizer_text = pickle.load(handle)
    
    # load label encoder object
with open('./models/label_encoder_text.pickle', 'rb') as enc:
        lbl_encoder_text = pickle.load(enc)
# parameters
max_len = 20

app = Flask(__name__)

@app.route('/')
def main_page():
    return redirect((url_for('index')), code=302)

@app.route('/index', methods=['GET', 'POST'])
def index():

    if request.method == 'POST':
        try:
            url = request.form['input_url']
            page = requests.get(url, allow_redirects=False)
            tree = html.fromstring(page.text)
        except:
            return render_template('index.html', article='Không tồn tại bài báo này hoặc tên miền này chưa được hỗ trợ !')

        if url.startswith('https://vnexpress.net'):
            article_name = tree.xpath("//h1[@class='title-detail']/text()")[0].strip()
            article_description = tree.xpath("//p[@class='description']/text()")[0].strip()
            article_content = ' '.join((s.strip() for s in tree.xpath("//p[@class='Normal']/text()"))).strip()

            article = article_name + '\n' + article_description + '\n' + article_content
            article_predict = article_name + ' ' + article_description + ' ' + article_content
            article_predict = preprocess(article_predict)
            article_predict = ViTokenizer.tokenize(article_predict)
            predict = UIT_model.predict(keras.preprocessing.sequence.pad_sequences(tokenizer_UIT.texts_to_sequences([article_predict]),
                                         truncating='post', maxlen=max_len))
            tag = lbl_encoder_UIT.inverse_transform([np.argmax(predict)])
            tag = str(tag)
            if tag == '[0]':
                tag = 'Công nghệ'
            if tag == '[1]':
                tag = 'Du lịch'
            if tag == '[2]':
                tag = 'Giáo dục'
            if tag == '[3]':
                tag = 'Giải trí'
            if tag == '[4]':
                tag = 'Khoa học'
            if tag == '[5]':
                tag = 'Kinh doanh'
            if tag == '[6]':
                tag = 'Luật pháp'
            if tag == '[7]':
                tag = 'Sức khỏe'
            if tag == '[8]':
                tag = 'Thế giới'
            if tag == '[9]':
                tag = 'Thể thao'
            if tag == '[10]':
                tag = 'Tin tức'
            if tag == '[11]':
                tag = 'Xe cộ'
            if tag == '[12]':
                tag = 'Đời sống'
            classifi = open('./web/WEBDEMO/file_classification.csv','a',encoding='utf8')
            classifi.write(url + ',' + tag + '\n')
            return render_template('index.html', article=article_predict, predict=tag)
        else:
            response = requests.get(url)
            soup = BeautifulSoup(response.text, "lxml")

            # kill all script and style elements
            for script in soup(["script", "style"]):
               script.extract()    # rip it out

            # get text
            text = soup.get_text()

            # break into lines and remove leading and trailing space on each
            lines = (line.strip() for line in text.splitlines())
            # break multi-headlines into a line each
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            # drop blank lines
            text = '\n'.join(chunk for chunk in chunks if chunk)
            content= text
            article_predict = ViTokenizer.tokenize(content)
            #Converting Dataset to Dataframe :
            predict = UIT_model.predict(keras.preprocessing.sequence.pad_sequences(tokenizer_UIT.texts_to_sequences([article_predict]),
                                         truncating='post', maxlen=max_len))
            tag = lbl_encoder_UIT.inverse_transform([np.argmax(predict)])
            tag = str(tag)
            if tag == '[0]':
                tag = 'Công nghệ'
            if tag == '[1]':
                tag = 'Du lịch'
            if tag == '[2]':
                tag = 'Giáo dục'
            if tag == '[3]':
                tag = 'Giải trí'
            if tag == '[4]':
                tag = 'Khoa học'
            if tag == '[5]':
                tag = 'Kinh doanh'
            if tag == '[6]':
                tag = 'Luật pháp'
            if tag == '[7]':
                tag = 'Sức khỏe'
            if tag == '[8]':
                tag = 'Thế giới'
            if tag == '[9]':
                tag = 'Thể thao'
            if tag == '[10]':
                tag = 'Tin tức'
            if tag == '[11]':
                tag = 'Xe cộ'
            if tag == '[12]':
                tag = 'Đời sống'
            classifi = open('./web/WEBDEMO/file_classification.csv','a',encoding='utf8')
            classifi.write(url + ',' + tag + '\n')
            return render_template('index.html', article=article_predict, predict=tag)
        return render_template('index.html')
    return render_template('index.html')

@app.route('/index_english', methods=['GET', 'POST'])
def index_english():
    models = [ln_model, nb_model, model, model_english]
    if request.method == 'POST':
        try:
            url = request.form['input_url']
            page = requests.get(url, allow_redirects=False)
            tree = html.fromstring(page.text)
        except:
            return render_template('index_english.html', article='Không tồn tại bài báo này hoặc tên miền này chưa được hỗ trợ !')

        if url.startswith('https://vnexpress.net'):
            article_name = tree.xpath("//h1[@class='title-detail']/text()")[0].strip()
            article_description = tree.xpath("//p[@class='description']/text()")[0].strip()
            article_content = ' '.join((s.strip() for s in tree.xpath("//p[@class='Normal']/text()"))).strip()

            article = article_name + '\n' + article_description + '\n' + article_content
            article_predict = article_name + ' ' + article_description + ' ' + article_content
            predict = model.predict(keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences([preprocess(article_predict)]),
                                         truncating='post', maxlen=max_len))
            tag = lbl_encoder.inverse_transform([np.argmax(predict)])
            return render_template('index_english.html', article=preprocess(article_predict), predict=tag)
        else:
            #Converting Dataset to Dataframe :
            articles = []
            labels = []

            with open("bbc-text.csv", 'r') as csvfile:
                reader = csv.reader(csvfile, delimiter=',')
                next(reader)
                for row in reader:
                    labels.append(row[0])
                    article = row[1]
                    for word in STOPWORDS:
                        token = ' ' + word + ' '
                        article = article.replace(token, ' ')
                        article = article.replace(' ', ' ')
                    articles.append(article)
            response = requests.get(url)
            soup = BeautifulSoup(response.text, "lxml")

            # kill all script and style elements
            for script in soup(["script", "style"]):
               script.extract()    # rip it out

            # get text
            text = soup.get_text()

            # break into lines and remove leading and trailing space on each
            lines = (line.strip() for line in text.splitlines())
            # break multi-headlines into a line each
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            # drop blank lines
            text = '\n'.join(chunk for chunk in chunks if chunk)
            content= text.encode('utf-8')
            content = preprocess(content)
            #Converting Dataset to Dataframe :
            raw_df=pd.DataFrame({"Text":articles,"Labels":labels})
            dataset=raw_df
            dataset['Text']=dataset['Text'].apply(clean_text)
            #Splitting raw data for Training and Testing :
            text = dataset["Text"].values
            labels = dataset['Labels'].values

            X_train, y_train, X_test, y_test = train_test_split(text,labels, test_size = 0.20, random_state = 42)
            X_test=X_test.reshape(-1,1)
            y_test=y_test.reshape(-1,1)
            #Let's Fix Some Common Parameters :
            # The maximum number of words to be used. (most frequent)
            vocab_size = 50000

            # Dimension of the dense embedding.
            embedding_dim = 128

            # Max number of words in each complaint.
            max_length = 200

            # Truncate and padding options
            trunc_type = 'post'
            padding_type = 'post'
            oov_tok = '<OOV>'
            tokenizer_en = Tokenizer(num_words=vocab_size, oov_token='<OOV>')
            tokenizer_en.fit_on_texts(X_train)
            tokenizer_en.word_index
            # Converting into Text to sequences and padding :
            train_seq = tokenizer_en.texts_to_sequences(X_train)
            train_padded = pad_sequences(train_seq, maxlen=max_length, padding=padding_type, truncating=trunc_type)

            tokenizer_en.texts_to_sequences(y_train)
            X_train[3],train_seq[3],train_padded[3]
            #Using One Hot Enocder to Enocde our Multi class Labels  :
            new_text = [clean_text(str(content))]
            seq = tokenizer_en.texts_to_sequences(new_text)
            padded = pad_sequences(seq, maxlen=max_length, padding=padding_type, truncating=trunc_type)
            pred = model_english.predict(padded)
            encode = OneHotEncoder()
            encode.fit_transform(X_test)
            encode.transform(y_test)
            predicted_label = encode.inverse_transform(pred)
            return render_template('index_english.html', article= content, predict = predicted_label[0])
        return render_template('index_english.html')
    return render_template('index_english.html',models=models)
@app.route('/portal', methods=['GET', 'POST'])
def portal():
    if request.method == 'POST':
        try:
            url = request.form['input_url']
            page = requests.get(url, allow_redirects=False)
            tree = html.fromstring(page.text)
        except:
            return render_template('portal.html', article='Không tồn tại bài báo này hoặc tên miền này chưa được hỗ trợ !')

        if url.startswith('https://vnexpress.net'):
            article_name = tree.xpath("//h1[@class='title-detail']/text()")[0].strip()
            article_description = tree.xpath("//p[@class='description']/text()")[0].strip()
            article_content = ' '.join((s.strip() for s in tree.xpath("//p[@class='Normal']/text()"))).strip()

            article = article_name + '\n' + article_description + '\n' + article_content
            article_predict = article_name + ' ' + article_description + ' ' + article_content
            article_name = preprocess(article_name)
            article_name = ViTokenizer.tokenize(article_name)
            predict = UIT_model.predict(keras.preprocessing.sequence.pad_sequences(tokenizer_UIT.texts_to_sequences([article_name]),
                                         truncating='post', maxlen=max_len))
            tag = lbl_encoder_UIT.inverse_transform([np.argmax(predict)])
            tag = str(tag)
            if tag == '[0]':
                tag = 'Công nghệ'
            if tag == '[1]':
                tag = 'Du lịch'
            if tag == '[2]':
                tag = 'Giáo dục'
            if tag == '[3]':
                tag = 'Giải trí'
            if tag == '[4]':
                tag = 'Khoa học'
            if tag == '[5]':
                tag = 'Kinh doanh'
            if tag == '[6]':
                tag = 'Luật pháp'
            if tag == '[7]':
                tag = 'Sức khỏe'
            if tag == '[8]':
                tag = 'Thế giới'
            if tag == '[9]':
                tag = 'Thể thao'
            if tag == '[10]':
                tag = 'Tin tức'
            if tag == '[11]':
                tag = 'Xe cộ'
            if tag == '[12]':
                tag = 'Đời sống'
            return render_template('portal.html', article=article_name, predict=tag)
    return render_template('portal.html')

@app.route('/index_vietnamese', methods=['GET', 'POST'])
def index_vietnamese():
    if request.method == 'POST':
        url = request.form['input_url']
        url = preprocess(url)
        url = ViTokenizer.tokenize(url)
        predict = UIT_model.predict(keras.preprocessing.sequence.pad_sequences(tokenizer_UIT.texts_to_sequences([url]),
                                         truncating='post', maxlen=max_len))
        tag = lbl_encoder_UIT.inverse_transform([np.argmax(predict)])
        tag = str(tag)
        if tag == '[0]':
            tag = 'Công nghệ'
        if tag == '[1]':
            tag = 'Du lịch'
        if tag == '[2]':
            tag = 'Giáo dục'
        if tag == '[3]':
            tag = 'Giải trí'
        if tag == '[4]':
            tag = 'Khoa học'
        if tag == '[5]':
            tag = 'Kinh doanh'
        if tag == '[6]':
            tag = 'Luật pháp'
        if tag == '[7]':
            tag = 'Sức khỏe'
        if tag == '[8]':
            tag = 'Thế giới'
        if tag == '[9]':
            tag = 'Thể thao'
        if tag == '[10]':
            tag = 'Tin tức'
        if tag == '[11]':
            tag = 'Xe cộ'
        if tag == '[12]':
            tag = 'Đời sống'
        return render_template('index_vietnamese.html', article=url, predict=tag)
    return render_template('index_vietnamese.html')
@app.route('/crawl', methods=['GET', 'POST'])
def crawl():
        categories = ['giao-duc', 'khoa-hoc', 'the-thao', 'kinh-doanh', 'suc-khoe', 'the-gioi', 'giai-tri', 'du-lich', 'so-hoa', 'thoi-su', 'phap-luat']
        for item in categories:

            url =  'https://vnexpress.net/{0}'.format(item)

            page = requests.get(url)

            soup = BeautifulSoup(page.content, 'html.parser')

            # Fetch link of articles in a category
            link_categories = list(soup.find_all('h3', class_=["title_news", "title-news"]))
            # print(link_categories[0].prettify())

            link_articles = []
            for link in link_categories:
                link_articles.append(link.find('a')['href'])
            # print(len(link_articles))

            # Fetch content of articles
            articles = []
            for link in link_articles:
                articles.append(link)
                a = open('./web/WEBDEMO/vnexpress.csv','a',encoding='utf8')
                a.write(link + '\n')
                if request.method == 'POST':
                    response = requests.get(link)
                    soup = BeautifulSoup(response.text, "lxml")

                    # kill all script and style elements
                    for script in soup(["script", "style"]):
                         script.extract()    # rip it out

                    # get text
                    text = soup.get_text()

                    # break into lines and remove leading and trailing space on each
                    lines = (line.strip() for line in text.splitlines())
                    # break multi-headlines into a line each
                    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                    # drop blank lines
                    text = '\n'.join(chunk for chunk in chunks if chunk)
                    content= text
                    article_predict = ViTokenizer.tokenize(content)
                    #Converting Dataset to Dataframe :
                    predict = UIT_model.predict(keras.preprocessing.sequence.pad_sequences(tokenizer_UIT.texts_to_sequences([article_predict]),
                                                truncating='post', maxlen=max_len))
                    tag = lbl_encoder_UIT.inverse_transform([np.argmax(predict)])
                    tag = str(tag)
                    if tag == '[0]':
                        tag = 'Công nghệ'
                    if tag == '[1]':
                        tag = 'Du lịch'
                    if tag == '[2]':
                        tag = 'Giáo dục'
                    if tag == '[3]':
                        tag = 'Giải trí'
                    if tag == '[4]':
                        tag = 'Khoa học'
                    if tag == '[5]':
                        tag = 'Kinh doanh'
                    if tag == '[6]':
                        tag = 'Luật pháp'
                    if tag == '[7]':
                        tag = 'Sức khỏe'
                    if tag == '[8]':
                        tag = 'Thế giới'
                    if tag == '[9]':
                        tag = 'Thể thao'
                    if tag == '[10]':
                        tag = 'Tin tức'
                    if tag == '[11]':
                        tag = 'Xe cộ'
                    if tag == '[12]':
                        tag = 'Đời sống'
                    classifi = open('./web/WEBDEMO/classification_vnexpress.csv','a',encoding='utf8')
                    classifi.writelines(link+ ',' +tag + '\n')
        return render_template('crawl.html', article= '\n'.join(articles))
   
@app.route('/download')
def download():
    try:
        return send_file('file_classification.csv', attachment_filename='file_classification.csv', as_attachment=True, cache_timeout=0)
    except Exception as e:
        try:
            return str(e)
        finally:
            e = None
            del e
@app.route('/download_crawl')
def download_crawl():
    try:
        return send_file('classification_vnexpress.csv', attachment_filename='classification_vnexpress.csv', as_attachment=True, cache_timeout=0)
    except Exception as e:
        try:
            return str(e)
        finally:
            e = None
            del e
if __name__ == '__main__':
    app.run()