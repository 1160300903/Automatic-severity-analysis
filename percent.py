import json
import numpy as np
outputpaths = ["eclipse_nohttp.json","mozilla_nohttp.json","gcc_nohttp.json"]
severity_index = { 'blocker':0, 'critical':1,'major':2, 'normal':3, 'minor':4, 'trivial':5,'enhancement':6}
for i in range(len(outputpaths)):
    path = "data/nohttp_data/"+outputpaths[i]
    result = [0]*7
    with open(path,"r",encoding="utf-8") as f:
        for line in f.readlines():
            temp = json.loads(line.strip())
            result[severity_index[temp["severity"]]]+=1
    sum_count = sum(result)
    print(result)
    result = np.array(result)
    print(result/sum_count)
