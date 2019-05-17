from knn import *
from utils import *
import json
input1 = ["eclipse_so_pre.json","mozilla_so_pre.json","gcc_so_pre.json"]
input2 = ["eclipse_pre.json","mozilla_pre.json","gcc_pre.json"]
output = ["eclipse_full.json","mozilla_full.json","gcc_full.json"]
severity = ['blocker', 'critical','major', 'minor', 'trivial']
if __name__=="__main__":
    for i in range(len(input1)):
        model = load_word_embedding("wordEmbedding/Word2Vec.model.txt")
        k = Knn(model,"data/preprocessed_data/"+input2[i],"knn/knn.txt")
        k.init_corpus()
        k.train_corpus = k.corpus
        k.train_tag = k.tag
        print("training")
        k.train()
        print("predicting")
        k.test_corpus = []
        with open("data/so_data/"+input1[i],"r") as fin:
            for line in fin.readlines():
                temp =  json.loads(line.strip())
                k.test_corpus.append(temp["summary"])
        k.predict()
        tag = k.get_tag(25)
        results = []
        for j in range(len(k.test_corpus)):
            if k.nn_sim[j][24] >= 0.8:
                temp = {"summary":k.test_corpus[j],"severity":severity[tag[j]]}
                results.append(temp)
        with open("data/full_data/"+output[i],"w") as fout:
            for temp in results:
                fout.write(json.dumps(temp)+"\n")
