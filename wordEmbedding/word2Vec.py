from gensim.models import Word2Vec
import nltk
from nltk.corpus import stopwords
import numpy as np
import json
filepaths = ["eclipse_nohttp.json","mozilla_nohttp.json","gcc_nohttp.json"]
split_corpus = []
stop_words = stopwords.words("english")
for i in range(len(filepaths)):
    lines = []
    filepath = "../data/nohttp_data/"+filepaths[i]
    print(filepath)
    with open(filepath,"r") as fin:
        for line in fin.readlines():
            temp  = json.loads(line.strip())
            lines += temp["summary"]
    split_corpus += [[word for word in nltk.word_tokenize(x) if word not in stop_words] for x in lines]
split_corpus = np.array(split_corpus)
model  = Word2Vec(split_corpus,size = 300,min_count =5,window =3)
model.wv.save_word2vec_format("word2Vec.model.txt", binary=False)