import sys
import numpy as np

from sklearn.ensemble import GradientBoostingClassifier

from csv import DictReader
import io 

reload(sys)  
sys.setdefaultencoding('utf8')

class DataSet():
    def __init__(self, name="train", path="./fnc-1-baseline/fnc-1"):
        self.path = path

        print("Reading dataset")
        bodies = name+"_bodies.csv"
        stances = name+"_stances.csv"

        self.stances = self.read(stances)
        articles = self.read(bodies)
        self.articles = dict()

        #make the body ID an integer value
        for s in self.stances:
            s['Body ID'] = int(s['Body ID'])

        #copy all bodies into a dictionary
        for article in articles:
            self.articles[int(article['Body ID'])] = article['articleBody']

        print("Total stances: " + str(len(self.stances)))
        print("Total bodies: " + str(len(self.articles)))



    def read(self,filename):
        rows = []
        with io.open(self.path + "/" + filename, "r" , encoding='utf-8') as table:
            r = DictReader(table)

            for line in r:
                rows.append(line)
        return rows

d = DataSet()

print("Read Dataset")

headlines = []
bodies = []
y_out = []
for stance in d.stances:
    bid = stance['Body ID']
    bodies.append(d.articles[bid])
    headlines.append(stance['Headline'])
    y_out.append(stance['Stance']=='related')

print("Reading Skipthoughts model")

import skipthoughts
model = skipthoughts.load_model()

encoder = skipthoughts.Encoder(model)
ascBodies = [str(body).decode('utf-8') for body in bodies]
print("Encoding body Vectors")

#bodyVectors = []
#vec = [0]*4800
#for body in bodies:
#    bodySen = str(body).decode('utf-8').split(".")
#    bodySen = [ body for body in bodySen if (not body.isspace()) and body]
#    vecT = encoder.encode(bodySen)
#    vec = [(x+y) for x,y in zip(vec,vecT)]
#    bodyVectors.append(vec)

bodyDictVector = {}
count = 0
for bid in d.articles:
    text = str(d.articles[bid]).decode('utf-8')
    textList = text.split(".")
    if count%100 == 0 :
        print("body number " + str(count))
    bodySen = [ t for t in textList if (not t.isspace()) and t]
    vecT = encoder.encode(bodySen,verbose=False)
    vec = map(sum,zip(*vecT))
    bodyDictVector[bid]=vec
    count +=1 

print("Pickling body vectors")
import pickle
with open('bodyVec.pkl','w') as f:
    pickle.dump(bodyDictVector,f)
