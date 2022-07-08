import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import os, glob
from sklearn.feature_extraction.text import TfidfVectorizer
# reference: https://github.com/paulwong16/ECE657-Tools_of_Intelligent_Sys_Design/tree/master/a3

def get_dataset(path, label):
    data = []
    # read all text file, each file only contains 1 line
    for filename in glob.glob(os.path.join(path,'*.txt')):
       with open(os.path.join(os.getcwd(), filename), 'r', encoding='utf-8') as f:
           data.append(f.readline())

    # positive = 1, negative = 0
    if(label == "positive"):
        labels = [1 for i in range(len(data))]
    if(label == "negative"):
        labels = [0 for i in range(len(data))]
    return data, labels

# combine positive and negative dataset
def combine_dataset(data1, data2, labels1, labels2):
    return data1 + data2, labels1 + labels2

def data_filter(dataset):
    nltk.download('stopwords')
    stopwords_lst = stopwords.words('english')

    dataset_filtered = []
    # remove punctuation with regex
    tokenizer = RegexpTokenizer(r'\w+')
    for sentence in dataset:
        sentence_filtered = []
        sentence = sentence.lower()
        sentence = tokenizer.tokenize(sentence)
        # remove stopwords that not useful
        for word in sentence:
            if word not in stopwords_lst:
                sentence_filtered.append(word)
        sentence_filtered = ' '.join(sentence_filtered)
        dataset_filtered.append(sentence_filtered)
    return dataset_filtered

# convert word to vector
def vectorization(dataset):
    vectorizer = TfidfVectorizer()
    features = vectorizer.fit_transform(dataset)
    return vectorizer, features
        
