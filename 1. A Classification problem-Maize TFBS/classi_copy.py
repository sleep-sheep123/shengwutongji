

import numpy as np
import pandas as pd
import operator
import itertools
from sklearn import metrics


class classifier:

    def __inint__(self,):
        pass

    def readFa2seqslist(self,filename):
        '''
        @msg: 读取一个fasta文件
        @param fa {str}  fasta 文件路径
        @return: {generator} 返回一个生成器，能迭代得到fasta文件的每一个序列名和序列
        '''
        import glob
        seqs_list = []
        fafile_list = glob.glob(filename)
        for i in fafile_list:
            with open(i,'r') as FA:
                seqName,seq='',''
                while 1:
                    line=FA.readline()
                    line=line.strip('\n')
                    if (line.startswith('>') or not line) and seqName:
                        seqs_list.append([seqName,seq])
                    if line.startswith('>'):
                        seqName = i[:3]+','+line[1:]
                        seq=''
                    else:
                        seq+=line
                    if not line:break
        return seqs_list


    def seqs2df(self,fa_list, k_size):
        """a function to calculate kmers from seq and generate a dict"""
        k_mers = list(("".join(x) for x in itertools.product("ATCGN",repeat=k_size)))
        dicts_list = []
        for seqname,seq in fa_list:
            mers_dict = {}
            n_kmers = len(seq) - k_size + 1
            for i in range(n_kmers):
                kmer = seq[i:i + k_size]
                if kmer not in mers_dict:
                    mers_dict[kmer] = 1
                else:
                    mers_dict[kmer] += 1

            if seqname.startswith('pos'):
                mers_dict['target'] = 1
            else:
                mers_dict['target'] = 0
            for i in k_mers:
                if i not in mers_dict:
                    mers_dict[i] = 0
            mers_dict['seqname'] = seqname[4:]
            sort_mers_dict = dict(sorted(mers_dict.items(), key=operator.itemgetter(0)))
            # for i in list(sort_mers_dict.keys()):
            #     if i.count("N")/len(i) >= 0.9:
            #         sort_mers_dict.pop(i)
            dicts_list.append(sort_mers_dict)
        df = pd.DataFrame(dicts_list)
        df = df.set_index('seqname')
        # df = df.fillna(0)
        cols = df.columns.tolist()                   
        cols.append(cols.pop(cols.index('target')))
        df = df[cols]
        # df = df.loc[:, (df != 0).any(axis=0)]
        return df


    # 切分训练数据和测试数据
    def split_train_test(self,x,y,test_size):
        from sklearn.model_selection import train_test_split
        x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=test_size,random_state=1)
        return x_train,x_test,y_train,y_test


    # # KNN Classifier
    def knn_classifier(self,x_train,x_test,y_train,y_test):
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.model_selection import GridSearchCV
        param_grid = [{'weights':['uniform'],
              'n_neighbors':[i for i in range(1,11)]},
             {'weights':['distance'],
              'n_neighbors':[i for i in range(1,11)],
             'p':[i for i in range(1,6)]}]

        model = KNeighborsClassifier()
        grid_search = GridSearchCV(model,param_grid,n_jobs = -1,cv = 5,scoring="roc_auc")
        grid_search.fit(x_train,y_train)
        score = grid_search.score(x_test, y_test)
        best_model = grid_search.best_estimator_
        y_predict = best_model.predict(x_test)
        score1 = metrics.accuracy_score(y_test,y_predict)
        return f"最优参数为：{grid_search.best_params_}",f"auc:{score}",f"acc:{score1}"


    # Logistic Regression Classifier
    def logistic_regression_classifier(self,x_train,x_test,y_train,y_test):
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import GridSearchCV

        params = {'C': [0.001,0.1,1, 10, 100, 1000],
                    'solver':['liblinear','newton-cg','lbfgs','sag','saga'],
                    "max_iter" : [i for i in range(500,5000,500)]}
        classifier = LogisticRegression()
        grid_search = GridSearchCV(classifier, params,  verbose=0, cv=5,n_jobs = -1,error_score='raise',scoring="roc_auc")
        grid_search.fit(x_train,y_train)
        score = grid_search.score(x_test, y_test)
        best_model = grid_search.best_estimator_
        y_predict = best_model.predict(x_test)
        score1 = metrics.accuracy_score(y_test,y_predict)
        return f"最优参数为：{grid_search.best_params_}",f"auc:{score}",f"acc:{score1}"


    # Random Forest Classifier
    def random_forest_classifier(self,x_train,x_test,y_train,y_test):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import GridSearchCV

        params = {'n_estimators':[50,120,160,200,250],'max_depth':[1,2,3,5,7,9,11,13]}
        model = RandomForestClassifier()
        grid_search = GridSearchCV(model, params,  verbose=0, cv=5,n_jobs = -1,scoring="roc_auc")
        grid_search.fit(x_train,y_train)
        score = grid_search.score(x_test, y_test)
        best_model = grid_search.best_estimator_
        y_predict = best_model.predict(x_test)
        score1 = metrics.accuracy_score(y_test,y_predict)
        return f"最优参数为：{grid_search.best_params_}",f"auc:{score}",f"acc:{score1}"


    # Decision Tree Classifier
    def decision_tree_classifier(self,x_train,x_test,y_train,y_test):
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.model_selection import GridSearchCV

        model = DecisionTreeClassifier(random_state=80)
        params = {'max_depth':range(1,21),'criterion':['entropy','gini'],"max_leaf_nodes" : [3,5,6,7,8]}
        grid_search = GridSearchCV(model, params,  verbose=0, cv=5,n_jobs = -1,scoring="roc_auc")
        grid_search.fit(x_train,y_train)
        score = grid_search.score(x_test, y_test)
        best_model = grid_search.best_estimator_
        y_predict = best_model.predict(x_test)
        score1 = metrics.accuracy_score(y_test,y_predict)
        return f"最优参数为：{grid_search.best_params_}",f"auc:{score}",f"acc:{score1}"


    # GBDT(Gradient Boosting Decision Tree) Classifier
    def gradient_boosting_classifier(self,x_train,x_test,y_train,y_test):
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.model_selection import GridSearchCV

        params = {'learning_rate': np.linspace(0.10,0.3,num=10),'n_estimators':range(50,100,5),'max_depth': [3, 4, 5, 6, 7, 8]}
        model = GradientBoostingClassifier()
        grid_search = GridSearchCV(model, params,  verbose=0, cv=5,n_jobs = -1,scoring = "roc_auc")
        grid_search.fit(x_train,y_train)
        score = grid_search.score(x_test, y_test)
        best_model = grid_search.best_estimator_
        y_predict = best_model.predict(x_test)
        score1 = metrics.accuracy_score(y_test,y_predict)
        return f"最优参数为：{grid_search.best_params_}",f"auc:{score}",f"acc:{score1}"


    # SVM Classifier 
    def svm_cross_validation(self,x_train,x_test,y_train,y_test):
        from sklearn import svm
        from sklearn.model_selection import GridSearchCV

        params = {'kernel': ['linear','rbf'],'C': np.linspace(0.10,1,num=5)}
        model = svm.SVC()
        grid_search = GridSearchCV(model, params, cv=5,n_jobs = -1,scoring="roc_auc")
        grid_search.fit(x_train, y_train)
        score = grid_search.score(x_test, y_test)
        best_model = grid_search.best_estimator_
        y_predict = best_model.predict(x_test)
        score1 = metrics.accuracy_score(y_test,y_predict)
        return f"最优参数为：{grid_search.best_params_}",f"auc:{score}",f"acc:{score1}"


    #MLP
    def MLP_classifier(self,x_train,x_test,y_train,y_test):
        from sklearn.neural_network import MLPClassifier
        from sklearn.model_selection import GridSearchCV

        params = {
                    'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,)],
                    'activation': ['tanh', 'relu'],
                    'solver': ['sgd', 'adam'],
                    'alpha': [0.0001, 0.05],
                    'learning_rate': ['constant','adaptive'],
                    }
        model = MLPClassifier(max_iter=1000)
        grid_search = GridSearchCV(model, params, cv=5,n_jobs = -1,scoring="roc_auc")
        grid_search.fit(x_train, y_train)
        score = grid_search.score(x_test, y_test)
        best_model = grid_search.best_estimator_
        y_predict = best_model.predict(x_test)
        score1 = metrics.accuracy_score(y_test,y_predict)
        return f"最优参数为：{grid_search.best_params_}",f"auc:{score}",f"acc:{score1}"