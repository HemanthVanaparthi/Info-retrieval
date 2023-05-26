#triloka bhukya, sri sai amrutha chalasani, sai kiran chekuri, hemanth eswar vanaparthi
from tkinter import *
import tkinter
from tkinter import filedialog
import numpy as np
from tkinter import simpledialog
import matplotlib.pyplot as plt
import pandas as pd
import os, re, pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from string import punctuation
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import seaborn as sns
from sklearn.metrics import confusion_matrix

#creating interface
main = tkinter.Tk()
main.title("Movie Review Sentiment Prediction using Naive Bayes ML Algorithm") 
main.geometry("1000x650")

global filename
global naivebayes
global accuracy, precision, recall, Fscore
global dataset

#text or word normalization techniques
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stopWords = set(stopwords.words('english'))
punctuation = str.maketrans('', '', punctuation)
global X, Y
global X_train, X_test, y_train, y_test

#uploading the dataset
def upload():
  global filename
  global dataset
  filename = filedialog.askopenfilename(initialdir = "Dataset")
  text.delete('1.0', END)
  text.insert(END,filename+' Loaded\n\n')
  text.insert(END,"Dataset Loaded\n\n")
  dataset = pd.read_csv(filename,nrows=1000)
  text.insert(END,dataset.head())

#Tokenization, removal of punctuations and stopwords , stemming and lemmatization
def processReview(review):
  terms = word_tokenize(review.strip())
  terms = [term.translate(punctuation) for term in terms]
  terms = [term for term in terms if term.isalpha()]
  terms = [term for term in terms if not term in stopWords]
  terms = [lemmatizer.lemmatize(term) for term in terms]
  terms = [stemmer.stem(term) for term in terms if len(term) > 3]
  return ' '.join(terms)
# Data Preprocessing
def preprocessDataset():
  global dataset
  global X, Y
  X = []
  Y = []
  text.delete('1.0', END)
  dataset = dataset.values
  for i in range(len(dataset)):
    review = dataset[i,0]
    review = processReview(review)
    X.append(review)
    sentiment = dataset[i,1]
    sentiment = sentiment.strip("\n").strip()
    if sentiment == 'positive':
        Y.append(1)
    else:
        Y.append(0)
    text.insert(END,"Processed Review: "+review+"\n\n")
    text.update_idletasks()
# Vectorization  and train & test spilt
def tfidfVector():
  text.delete('1.0', END)
  global X, Y
  global X_train, X_test, y_train, y_test
  if os.path.exists('model/tfidf.txt'):
    with open('model/tfidf.txt', 'rb') as file:
      X = pickle.load(file)
    file.close()
    Y = np.load("model/Y.txt.npy")
  else:
    tfidf_vectorizer = TfidfVectorizer(stop_words=stop_words, use_idf=True, smooth_idf=False, norm=None, decode_error='replace', max_features=1000)
    tfidf = tfidf_vectorizer.fit_transform(X).toarray()        
    X = pd.DataFrame(tfidf, columns=tfidf_vectorizer.get_feature_names())
    with open('model/tfidf.txt', 'wb') as file:
      pickle.dump(X, file)
    file.close()  
  text.insert(END,str(X.head())+"\n\n")
  X = X.values
  X = X[:, 0:1000]
  indices = np.arange(X.shape[0])
  np.random.shuffle(indices)
  X = X[indices]
  Y = Y[indices]
  X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
  text.insert(END,"Total reviews found in dataset: "+str(X.shape[0])+"\n\n")
  text.insert(END,"Dataset train (80%) & test (20%) split\n\n")
  text.insert(END,"Number of Training Records: "+str(X_train.shape[0])+"\n")
  text.insert(END,"Number of Testing Records: "+str(X_test.shape[0])+"\n")
  
#training the Model(Naive bayes) and metrics evaluation
def runNaiveBayes():
  global accuracy, precision, recall, Fscore
  global naivebayes
  text.delete('1.0', END)
  global X_train, X_test, y_train, y_test
  if os.path.exists('model/nb.txt'):
    with open('model/nb.txt', 'rb') as file:
      naivebayes = pickle.load(file)
    file.close()
  else:
    naivebayes = GaussianNB()
    naivebayes.fit(X_train, y_train)
    with open('model/nb.txt', 'wb') as file:
      pickle.dump(naivebayes, file)
    file.close()
  predict = naivebayes.predict(X_test)
  accuracy = accuracy_score(y_test,predict)*100
  precision = precision_score(y_test,predict,average='macro') * 100
  recall = recall_score(y_test,predict,average='macro') * 100
  Fscore = f1_score(y_test,predict,average='macro') * 100
  text.insert(END,"Naive Bayes Precision  : "+str(precision)+"\n")
  text.insert(END,"Naive Bayes Recall     : "+str(recall)+"\n")
  text.insert(END,"Naive Bayes F1-Score   : "+str(Fscore)+"\n")
  text.insert(END,"Naive Bayes Accuracy   : "+str(accuracy)+"\n\n")
  LABELS = ['Negative','Positive'] 
  conf_matrix = confusion_matrix(y_test,predict) 
  plt.figure(figsize =(6, 6)) 
  ax = sns.heatmap(conf_matrix, xticklabels = LABELS, yticklabels = LABELS, annot = True, cmap="viridis" ,fmt ="g");
  ax.set_ylim([0,2])
  plt.title("Naive Bayes Confusion matrix") 
  plt.ylabel('True class') 
  plt.xlabel('Predicted class') 
  plt.show()    

#predicting the sentiment, whether positive review or negative review
def predictSentiment():
  global naivebayes
  text.delete('1.0', END)
  with open('model/tfidfvector.txt', 'rb') as file:
    tfidf_vector = pickle.load(file)
  file.close()
  textfile = filedialog.askopenfilename(initialdir = "Dataset")
  testData = pd.read_csv(textfile)
  testData = testData.values
  for i in range(len(testData)):
    review = testData[i,0]
    review = processReview(review)
    review = tfidf_vector.transform([review]).toarray()
    predict = naivebayes.predict(review)
    predict = predict[0]
    if predict == 1:
      text.insert(END,"Test Review: "+str(testData[i,0])+"  PREDICTED SENTIMENT =====> POSITIVE\n\n")
    else:
      text.insert(END,"Test Review: "+str(testData[i,0])+"  PREDICTED SENTIMENT =====> NEGATIVE\n\n") 
  
#metrics graph
def graph():
  height = [accuracy, precision, recall, Fscore]
  bars = ('Accuracy', 'Precision', 'Recall', 'FSCORE')
  f, ax = plt.subplots(figsize=(5,5))
  y_pos = np.arange(len(bars))
  plt.bar(y_pos, height)
  plt.xticks(y_pos, bars)
  ax.legend(fontsize = 12)
  plt.title("Naive Bayes Precision, Recall, Accuracy & FSCORE Graph")
  plt.show()

#adding the following buttons to display the results
font = ('times', 16, 'bold')
title = Label(main, text='Movie Review Sentiment Prediction using Naive Bayes ML Algorithm', justify=LEFT)
title.config(bg='lavender blush', fg='DarkOrchid1')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=100,y=5)
title.pack()

font1 = ('times', 13, 'bold')
uploadButton = Button(main, text="Upload IMDB Movie Reviews Dataset", command=upload)
uploadButton.place(x=10,y=100)
uploadButton.config(font=font1)

preprocessButton = Button(main, text="Preprocess Dataset", command=preprocessDataset)
preprocessButton.place(x=360,y=100)
preprocessButton.config(font=font1) 

tfidfButton = Button(main, text="TF-IDF Vectorization", command=tfidfVector)
tfidfButton.place(x=680,y=100)
tfidfButton.config(font=font1) 

nbButton = Button(main, text="Run Naive Bayes Algorithm", command=runNaiveBayes)
nbButton.place(x=10,y=150)
nbButton.config(font=font1)

predictButton = Button(main, text="Predict Sentiments from Test Data", command=predictSentiment)
predictButton.place(x=360,y=150)
predictButton.config(font=font1)

graphButton = Button(main, text="Metrics Comparison Graph", command=graph)
graphButton.place(x=680,y=150)
graphButton.config(font=font1)

font1 = ('times', 12, 'bold')
text=Text(main,height=20,width=120)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=200)
text.config(font=font1) 

main.config(bg='light coral')
main.mainloop()
