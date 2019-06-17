import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import  TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB as MNB
from sklearn.model_selection  import cross_val_score
def proces(review):
    review_text=re.sub("[^a-zA-Z]"," ",review)
    words=review_text.lower().split()
    return words

train=pd.read_csv('train_new.csv',lineterminator='\n')
print(train.head(5))
x=train['review']
y=train['label']
test=pd.read_csv('20190520_test.csv', lineterminator='\n')
z=test['review']

train_data=[]
for i in range(len(x)):
    train_data.append(' '.join(proces(x[i])))
    pass
test_data=[]
for i in range(len(z)):
    test_data.append(' '.join(proces(z[i])))
    pass

data_all=train_data+test_data
count_vec = TfidfVectorizer(min_df=2,  # 最小支持度为2
                            analyzer='word',
                            ngram_range=(1, 2),  # 二元文法模型
                            use_idf=1,
                            smooth_idf=1,
                            sublinear_tf=1,
                            stop_words=set([
                                "ke",
                                "ka",
                                "ek",
                                "mein",
                                "kee",
                                "hai",
                                "yah",
                                "aur",
                                "se",
                                "hain",
                                "ko",
                                "par",
                                "is",
                                "hota",
                                "ki",
                                "jo",
                                "kar",
                                "me",
                                "gaya",
                                "karane",
                                "kiya",
                                "liye",
                                "apane",
                                "ne",
                                "banee",
                                "nahin",
                                "to",
                                "hee",
                                "ya",
                                "evan",
                                "diya",
                                "ho",
                                "isaka",
                                "tha",
                                "dvaara",
                                "hua",
                                "tak",
                                "saath",
                                "karana",
                                "vaale",
                                "baad",
                                "lie",
                                "aap",
                                "kuchh",
                                "sakate",
                                "kisee",
                                "ye",
                                "isake",
                                "sabase",
                                "isamen",
                                "the",
                                "do",
                                "hone",
                                "vah",
                                "ve",
                                "karate",
                                "bahut",
                                "kaha",
                                "varg",
                                "kaee",
                                "karen",
                                "hotee",
                                "apanee",
                                "unake",
                                "thee",
                                "yadi",
                                "huee",
                                "ja",
                                "na",
                                "ise",
                                "kahate",
                                "jab",
                                "hote",
                                "koee",
                                "hue",
                                "va",
                                "na",
                                "abhee",
                                "jaise",
                                "sabhee",
                                "karata",
                                "unakee",
                                "tarah",
                                "us",
                                "aadi",
                                "kul",
                                "es",
                                "raha",
                                "isakee",
                                "sakata",
                                "rahe",
                                "unaka",
                                "isee",
                                "rakhen",
                                "apana",
                                "pe",
                                "usake",
                            ])
                            )
lenth=len(train_data)
count_vec.fit(data_all)
data_all=count_vec.transform(data_all)
train_data=data_all[:lenth]
test_data=data_all[lenth:]

model=MNB()
model.fit(train_data,y)
MNB(alpha=1.0, class_prior=None, fit_prior=True)
print("多项式贝叶斯分类器10折交叉验证得分: ", np.mean(cross_val_score(model, train_data,y, cv=10, scoring='roc_auc')))

test_pred = np.array(model.predict_proba(test_data))
print ("保存结果")
nb_output = pd.DataFrame(data=test_pred, columns=['Neg','Pred'])
nb_output['ID'] = test['ID']
nb_output = nb_output[['ID', 'Pred']]
nb_output.to_csv('pred.csv', index=False)
print ("完成")
