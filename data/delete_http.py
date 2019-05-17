import re
import json
filepaths = ["eclipse.json","mozilla.json","gcc.json"]
outputpaths = ["eclipse_nohttp.json","mozilla_nohttp.json","gcc_nohttp.json"]
p1 = re.compile(r"https?://.*?\s")#匹配位于句子中间的网址
p2 = re.compile(r"https?://.*")#用于匹配位于句子末尾的网址
for i in range(len(filepaths)):
    filepath = "raw_data/"+filepaths[i]
    outputpath = "nohttp_data/"+outputpaths[i]
    print(filepath,outputpath)
    results = []
    with open(filepath, encoding='utf-8') as fin:
        for line in fin.readlines():
            temp = json.loads(line.strip())
            if temp["severity"]=="normal" or temp["severity"]=="enhancement":
                continue
            new_summary = []
            for line in temp["summary"]:
                line = p1.sub(" ",line)
                line = p2.sub(" ",line)
                new_summary.append(line)
            temp["summary"] = new_summary
            results.append(temp)
    with open(outputpath,"w",encoding="utf-8") as fout:
        for bug in results:
            fout.write(json.dumps(bug)+"\n")

