from threading import Thread,Lock
import time
import re
import urllib.request
from bs4 import BeautifulSoup
import json
import nltk
class crawler():
    def __init__(self):
        self.year_num = 1
        self.time_limit = "2000-01-01"
        self.file_name = "raw_data/mozilla.json"
        self.download_count = 0
        self.baseurl = 'https://bugzilla.mozilla.org/show_bug.cgi?ctype=xml&id='
        self.first_url = "https://bugzilla.mozilla.org/buglist.cgi?bug_status=UNCONFIRMED&bug_status=NEW&bug_status=ASSIGNED&bug_status=REOPENED&f1=classification&field0-0-0=creation_ts&o1=notequals&query_format=advanced&type0-0-0=substring&v1=Graveyard&order=bug_status%2Cpriority%2Cassigned_to%2Cbug_id&limit=0&value0-0-0="
        self.all_bugs_info = []
        self.info_lock = Lock()
        self.duplicate_check = set()
        with open(self.file_name,"r",encoding="utf-8") as e:
            for line in e.readlines():
                temp = json.loads(line.strip())
                self.duplicate_check.add(temp["id"])
        #open(self.file_name, 'w', encoding='utf-8').close()#清空
    def output_html(self):
        fout = open(self.file_name, 'a', encoding='utf-8')
        for sample in self.all_bugs_info:
            try:
                fout.write(json.dumps(sample, ensure_ascii=False)+'\n')
            except:
                continue
        self.all_bugs_info = []
        fout.close()
    def count(self, words):
        count = 0
        for word in words:
            count= count+1 if word.isalpha() else count
        return count
    def get_summary(self, data,summaries):
        data["summary"] = []
        for summary in summaries:
            lines = summary.find("thetext").get_text().split("\n")
            for line in lines:
                words = nltk.word_tokenize(line)
                if self.count(words)>=10:
                    data["summary"].append(line)
                if len(data["summary"])>=10:
                    return
    def crawl_bug(self,link):
        try:
            data = {}
            bug_id = link["href"].strip().split("id=")[1]
            if bug_id in self.duplicate_check:
                return
            self.duplicate_check.add(bug_id)
            print(bug_id,end=" ")
            url = self.baseurl+bug_id
            request = urllib.request.Request(url)
            response = urllib.request.urlopen(request)
            html = response.read().decode(encoding="utf-8",errors="strict")
            soup = BeautifulSoup(html,features="html.parser")
            data["id"] = bug_id
            serverity_node = soup.find("bug_severity")
            if serverity_node==None:
                return
            data["severity"] = serverity_node.get_text()
            summaries = soup.find_all("long_desc",attrs={"isprivate":"0"})
            if summaries==None or not summaries:
                return
            self.get_summary(data,summaries)
            data["component"] = soup.find("component").get_text()
            data["product"] = soup.find("product").get_text()
            data["time"] = soup.find("creation_ts").get_text()
            if data["time"]<self.time_limit:
                return
            self.info_lock.acquire()
            self.all_bugs_info.append(data)
            self.info_lock.release()
        except Exception as e:
                print(e,end="")
    def scraw(self,year):
        start = time.time()
        print("thread"+str(year)+"built")
        request = urllib.request.Request(self.first_url+str(year))
        response = urllib.request.urlopen(request)
        html = response.read().decode(encoding="utf-8",errors="strict")
        soup = BeautifulSoup(html,features="html.parser")
        links = soup.find_all("a",href = re.compile(r"id="))
        for i in range(len(links)):
            link = links[i]
            t = Thread(target=self.crawl_bug,args=(link,))
            t.start()
            if i%1000:
                self.info_lock.acquire()
                self.output_html()
                self.info_lock.release()
            time.sleep(0.5)
        print(len(links))
        print("thread"+str(year)+":"+str(time.time()-start))
        self.info_lock.acquire()
        self.output_html()
        self.info_lock.release()
    def run(self):
        for i in range(self.year_num):
            self.scraw(2019+i)
if __name__=="__main__":
    crawler().run()
    


