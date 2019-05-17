import json
file_name = "nohttp_data/mozilla_nohttp.json"
with open(file_name, 'r', encoding='utf-8') as fin:
    all_bugs_info = [json.loads(line.strip()) for line in fin.readlines()]
a = list(sorted(all_bugs_info,key=lambda d:d["time"]))
print(len(a))
print(a[0]["time"],a[-1]["time"])
b = set()
with open(file_name,"w",encoding="utf-8") as fout:
    for info in a:
        b.add(info["severity"])
        fout.write(json.dumps(info,ensure_ascii=False)+"\n")
print(b)
