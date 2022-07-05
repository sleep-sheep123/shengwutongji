from classi_copy import classifier
from sklearn.model_selection import train_test_split
a = classifier()
seqs_list = a.readFa2seqslist('*.fa')

seqs_train,seqs_test = train_test_split(seqs_list,test_size=0.2,random_state=1008600)

# for i in range(6,7):
#     df_train = a.seqs2df(seqs_train,i)
#     x_train = df_train.iloc[:,:df_train.shape[1]-1]
#     y_train = df_train['target']
#     #print (x_train.head(5))
#     #print (y_train.head(5))

#     df_test = a.seqs2df(seqs_test,i)
#     x_test = df_test.iloc[:,:df_test.shape[1]-1]
#     y_test = df_test['target']


    # print (f"{i}mer_svm:",a.svm_cross_validation(x_train,x_test,y_train,y_test))
    # print ("===============================")
    # print (f"{i}mer_knn_classifier:",a.knn_classifier(x_train,x_test,y_train,y_test))
    # print ("===============================")
    # print (f"{i}mer_logistic_regression_classifier:",a.logistic_regression_classifier(x_train,x_test,y_train,y_test))
    # print ("===============================")
    # print (f"{i}mer_random_forest_classifier:",a.random_forest_classifier(x_train,x_test,y_train,y_test))
    # print ("===============================")
    # print (f"{i}mer_decision_tree_classifier:",a.decision_tree_classifier(x_train,x_test,y_train,y_test))
    # print ("===============================")
    # print (f"{i}mer_GBDT:",a.gradient_boosting_classifier(x_train,x_test,y_train,y_test))
    # print ("===============================")
    # print (f"{i}mer_MLP:",a.MLP_classifier(x_train,x_test,y_train,y_test))
    # print ("===============================")
    # print ("=================================================================================================================")



df_train = a.seqs2df(seqs_train,5)
x_train = df_train.iloc[:,:df_train.shape[1]-1]
y_train = df_train['target']

df_test = a.seqs2df(seqs_test,5)
x_test = df_test.iloc[:,:df_test.shape[1]-1]
y_test = df_test['target']
print ("done")
# print (x_test.shape)


from sklearn import svm
model = svm.SVC(C = 1, kernel= "rbf",probability=True)

# import joblib
# import ray
 
# # 2
# ray.init()
# from ray.util.joblib import register_ray
# register_ray()
# with joblib.parallel_backend('ray'):
#     model.fit(x_train,y_train)

model.fit(x_train,y_train)
y_prob = model.predict_proba(x_test)
# print (y_prob)

from sklearn.metrics import roc_auc_score as AUC
from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_test,y_prob[:,1], pos_label=1)

# 画出ROC曲线
import matplotlib.pyplot as plt
plt.figure()
plt.plot(fpr, tpr, color='red'
         ,label='ROC curve ')
plt.plot([0, 1], [0, 1], color='black', linestyle='--')
# 为了让曲线不黏在图的边缘
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('Recall')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()
