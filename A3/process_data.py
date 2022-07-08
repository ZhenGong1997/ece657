import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import os, glob
from sklearn.feature_extraction.text import TfidfVectorizer


def get_dataset(path, label):
    data = []
    for filename in glob.glob(os.path.join(path,'*.txt')):
       with open(os.path.join(os.getcwd(), filename), 'r', encoding='utf-8') as f:
           data.append(f.readline())
    if(label == "positive"):
        labels = [1 for i in range(len(data))]
    if(label == "negative"):
        labels = [0 for i in range(len(data))]
    return data, labels

def combine_dataset(data1, data2, labels1, labels2):
    return data1 + data2, labels1 + labels2

def data_filter(dataset):
    nltk.download('stopwords')
    stopwords_lst = stopwords.words('english')

    # remove punctuation and stopwords
    dataset_filtered = []
    tokenizer = RegexpTokenizer(r'\w+')
    for sentence in dataset:
        sentence_filtered = []
        sentence = sentence.lower()
        sentence = tokenizer.tokenize(sentence)
        for word in sentence:
            if word not in stopwords_lst:
                sentence_filtered.append(word)
        sentence_filtered = ' '.join(sentence_filtered)
        dataset_filtered.append(sentence_filtered)
    return dataset_filtered


def vectorization(dataset):
    vectorizer = TfidfVectorizer()
    features = vectorizer.fit_transform(dataset)
    return vectorizer, features
        
