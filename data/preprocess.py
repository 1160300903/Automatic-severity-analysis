import nltk
from nltk.corpus import stopwords
import numpy as np
import json
import nltk.stem
stemmer = nltk.stem.SnowballStemmer('english')
filepaths = ["eclipse_nohttp.json","mozilla_nohttp.json","gcc_nohttp.json"]
outputpaths = ["eclipse_pre.json","mozilla_pre.json","gcc_pre.json"]
stop_words = stopwords.words("english")
for i in range(len(filepaths)):
    filepath = "nohttp_data/"+filepaths[i]
    outputpath = "preprocessed_data/"+outputpaths[i]
    print(filepath,outputpath)
    results = []
    with open(filepath, encoding='utf-8') as fin:
        for line in fin.readlines():
            temp = json.loads(line.strip())
            new_summary = [stemmer.stem(word) for x in temp["summary"] for word in nltk.word_tokenize(x) if stemmer.stem(word) not in stop_words ]
            temp["summary"] = new_summary
            if new_summary:
                results.append(temp)
    with open(outputpath,"w",encoding="utf-8") as fout:
        for bug in results:
            fout.write(json.dumps(bug)+"\n")