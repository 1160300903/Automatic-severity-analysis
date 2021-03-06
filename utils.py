def one_hot_to_num(vectors,class_num):
    result = []
    for vec in vectors:
        for i in range(class_num):
            if vec[i]==1:
                result.append(i)
                break
    return result

def calculate_f_score(right,predict,class_num):
    result = []
    print("f-score-init")
    for i in range(class_num):
        result.append([0,0,0])
    assert len(right)==len(predict)
    print("couting right case")
    for i in range(len(right)):
        #print(right[i],predict[i])
        result[right[i]][1]+=1
        result[predict[i]][2]+=1
        if right[i]==predict[i]:
            result[predict[i]][0]+=1
    f_score = []
    for i in range(class_num):
        p = result[i][0]/(result[i][2] or 1)
        r = result[i][0]/(result[i][1] or 1)
        f_score.append((2*p*r)/((p+r)or 1))
    print(result)
    return f_score
def load_word_embedding(path):
    model = dict()
    with open(path,encoding="utf-8") as w:
        lines = w.readlines()
        for i in range(1,len(lines)):
            line = lines[i].strip()
            word, vec = line.split(" ",1)
            temp = []
            for num in vec.split():
                temp.append(float(num))
            model[word] = temp
    return model