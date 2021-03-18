#-*- coding: utf-8-*-

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.inspection import permutation_importance

path="C:\Msc thesis\rt"
inputfile="set317float.xlsx"

#输入数据，分为训练和测试集

df = pd.read_excel(inputfile, header=0)



y = df["type"]
x = df.drop(["type"], axis=1)



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0,
                                                   shuffle=True,stratify = y)

'''
#调参
scorel = []
for i in [ .1, .4, .5, .6, .7]:
    rfc = RandomForestClassifier(n_estimators=417,
                                 max_depth=41,
                                 max_features=i,
                                 random_state=60)
    score = cross_val_score(rfc,x_train,y_train,cv=10).mean()
    scorel.append(score)

print(max(scorel),(scorel.index(max(scorel))*1)+0)
plt.figure(figsize=[20,5])
plt.plot([ .1, .4, .5, .6, .7],scorel)
plt.show()



'''
#训练分类器

forest = RandomForestClassifier(n_estimators=417,
                                oob_score=True,
                                random_state=60,
                                max_features=0.4,
                                max_depth=41)
                                # criterion="entropy")
                                #min_samples_leaf=5)
forest.fit(x_train, y_train)

#预测
pred_train = forest.predict(x_train)
pred_test = forest.predict(x_test)

#计算分类器的准确率

train_acc = accuracy_score(y_train, pred_train)
test_acc = accuracy_score(y_test, pred_test)
print ("训练集准确率: {0:.2f}, 测试集准确率: {1:.2f}".format(train_acc, test_acc))

'''
#其他模型评估指标
#precision, recall, F1, _ = precision_recall_fscore_support(np.array(y_test),
#                                                          np.array(pred_test),
#                                                          average='binary')

'''
#特征重要度
'''
r = permutation_importance(forest, x_test, y_test,
                           n_repeats=30,
                           random_state=0)
for i in r.importances_mean.argsort()[::-1]:
    if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
        print(f"{data.feature_names[i]:<8}"
              f"{r.importances_mean[i]:.3f}"
              f" +/- {r.importances_std[i]:.3f}")
'''
features = list(x_test.columns)
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]
num_features = len(importances)

#将特征重要度以柱状图展示
plt.figure()
plt.title("Feature importances")
plt.bar(range(num_features), importances[indices], color="g", align="center")
plt.xticks(range(num_features), [features[i] for i in indices], rotation='45')
plt.xlim([-1, num_features])
plt.show()


#输出各个特征的重要度
for i in indices:
    print ("{0} - {1:.3f}".format(features[i], importances[i]))


