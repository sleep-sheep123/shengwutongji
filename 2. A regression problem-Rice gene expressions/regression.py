

import pandas as pd
import numpy as np
import operator
import itertools
from sklearn import metrics
from sklearn.linear_model import Ridge



class classifier:

    def __inint__(self,):
        pass


        #读取文件中的序列
    def readFa2seqslist(self,filename):
        '''
        @msg: 读取一个fasta文件
        @param fa {str}  fasta 文件路径
        @return: {generator} 返回一个生成器，能迭代得到fasta文件的每一个序列名和序列
        '''
        seqs_list = []


        with open(filename,'r') as FA:
            seqName,seq='',''
            while 1:
                line=FA.readline()
                line=line.strip('\n')
                if (line.startswith('>') or not line) and seqName:
                    seqs_list.append([seqName,seq])
                if line.startswith('>'):
                    seqName = line[1:]
                    seq=''
                else:
                    seq += line
                if not line:
                    break
        return seqs_list


    #将序列计算k-mer并转为dataframe
    def seqs2df(self,fa_list, k_size):
        """a function to calculate kmers from seq and generate a dict"""
        data = pd.read_csv(r'y.csv',sep=',',usecols=["Geneid","TPM"])
        data = data.set_index("Geneid")
        dicts_list = []
        k_mers = list(("".join(x) for x in itertools.product("ATCGN",repeat=k_size)))
        # arr = np.zeros((len(fa_list),len(k_mers)))
        # print (arr.shape)
        # df = pd.DataFrame(arr,columns = k_mers)
        for seqname,seq in fa_list:
            mers_dict = {}
            n_kmers = len(seq) - k_size + 1
            for i in range(n_kmers):
                kmer = seq[i:i + k_size]
                if kmer not in mers_dict:
                    mers_dict[kmer] = 1
                else:
                    mers_dict[kmer] += 1
            for i in k_mers:
                if i not in mers_dict:
                    mers_dict[i] = 0
            mers_dict['target'] = data['TPM'][seqname]
            
            mers_dict['seqname'] = seqname
            sort_mers_dict = dict(sorted(mers_dict.items(), key=operator.itemgetter(0)))
            # for i in list(sort_mers_dict.keys()):
            #     if i.count("N")/len(i) >= 0.5:
            #         sort_mers_dict.pop(i)
            dicts_list.append(sort_mers_dict)
            # df = df.append(mers_dict,ignore_index= True)
        df = pd.DataFrame(dicts_list)
        df = df.set_index('seqname')
        # df = df.fillna(0)
        cols = df.columns.tolist()                   
        cols.append(cols.pop(cols.index('target')))
        df = df[cols]
        # df = df.loc[:, (df != 0.0).any(axis=0)]
        return df


    ###########具体方法选择##########
    ####决策树回归####
    def DecisionTree_Regressor(self,x_train,x_test,y_train,y_test):
        from sklearn.tree import DecisionTreeRegressor
        from sklearn.model_selection import GridSearchCV

        params = {
                        'max_depth':[i for i in range(1,10,2)],
                        'min_samples_leaf':[i for i in range(1,10,2)],
                        'min_samples_split':[i for i in range(2,10,2)]
                    }
        model = DecisionTreeRegressor()
        grid_search = GridSearchCV(model, params, cv=5,n_jobs = -1,scoring="r2")
        grid_search.fit(x_train, y_train)
        score = grid_search.score(x_test, y_test)
        best_model = grid_search.best_estimator_
        y_predict = best_model.predict(x_test)
        score1 = metrics.mean_squared_error(y_test,y_predict)
        print ("最优参数为：",grid_search.best_params_)
        return score,score1


    ####3.2线性回归####
    def linear_regression(self,x_train,x_test,y_train,y_test):
        from sklearn.linear_model import LinearRegression
        from scipy.stats import pearsonr

        model = LinearRegression(n_jobs= -1)
        model.fit(x_train,y_train)
        y_predict = model.predict(x_test)
        mse = metrics.mean_squared_error(y_test,y_predict)
        p = pearsonr(y_predict,y_test)
        return mse,p




    #岭回归
    def ridge(self,x_train,x_test,y_train,y_test):
        from sklearn.linear_model import Ridge
        from sklearn.model_selection import GridSearchCV
        from scipy.stats import pearsonr

        params = {'alpha': [1,0.1,0.01,0.001,0.0001,0] , 
                        "fit_intercept": [True, False], 
                        "solver": ['svd', 'cholesky', 'lsqr',  'sag', 'saga']
                        }
        model = Ridge()
        grid_search = GridSearchCV(model,params,cv=5,n_jobs = -1,scoring = "r2")
        grid_search.fit(x_train,y_train)
        score = grid_search.score(x_test, y_test)
        best_model = grid_search.best_estimator_
        y_predict = best_model.predict(x_test)
        score1 = metrics.mean_squared_error(y_test,y_predict)
        print ("最优参数为：",grid_search.best_params_)
        p = pearsonr(y_predict,y_test)
        return score,score1,p


    #lasso
    def lasso_regressor(self,x_train,x_test,y_train,y_test):
        from sklearn.linear_model import Lasso
        from sklearn.model_selection import GridSearchCV
        from scipy.stats import pearsonr

        params = {"alpha" : [0.01,0.1,1.0,10.0],
                    "max_iter" : [1000,2000,5000,50000]}
        model = Lasso()
        grid_search = GridSearchCV(model,params,cv=5,n_jobs = -1,scoring = "r2")
        grid_search.fit(x_train,y_train)
        score = grid_search.score(x_test, y_test)
        best_model = grid_search.best_estimator_
        y_predict = best_model.predict(x_test)
        score1 = metrics.mean_squared_error(y_test,y_predict)
        print ("最优参数为：",grid_search.best_params_)
        p = pearsonr(y_predict,y_test)
        return score,score1,p




    # ####3.3SVM回归####
    # from sklearn import svm
    # model_SVR = svm.SVR()
    # ####3.4KNN回归####
    def knn_regressor(self,x_train,x_test,y_train,y_test):
        from sklearn import neighbors
        from sklearn.model_selection import GridSearchCV
        from scipy.stats import pearsonr

        params = [
                        {
                            'weights': ['uniform'],
                            'n_neighbors':[i for i in range(1,11)]
                        },
                        {
                            'weights':['distance'],
                            'n_neighbors':[i for i in range(1,11)],
                            'p':[i for i in range(1,6)]
                        }
                    ]

        model = neighbors.KNeighborsRegressor()
        grid_search=GridSearchCV(model,params,cv=5,n_jobs = -1,scoring="r2")
        grid_search.fit(x_train, y_train)
        score = grid_search.score(x_test, y_test)
        best_model = grid_search.best_estimator_
        y_predict = best_model.predict(x_test)
        score1 = metrics.mean_squared_error(y_test,y_predict)
        print ("最优参数为：",grid_search.best_params_)
        p = pearsonr(y_predict,y_test)
        return score,score1,p




    ####3.5随机森林回归####
    def RandomForest_Regressor(self,x_train,x_test,y_train,y_test):
        from sklearn import ensemble
        from sklearn.model_selection import GridSearchCV

        params = {'n_estimators': [100,200,300,400,500], 
                    'max_features': ["None", "sqrt", "log2"]
                    }
        model = ensemble.RandomForestRegressor()
        grid_search = GridSearchCV(model,params,cv=5,n_jobs = -1,scoring="r2")
        grid_search.fit(x_train,y_train)
        score = grid_search.score(x_test, y_test)
        best_model = grid_search.best_estimator_
        y_predict = best_model.predict(x_test)
        score1 = metrics.mean_squared_error(y_test,y_predict)
        print ("最优参数为：",grid_search.best_params_)
        return score,score1
