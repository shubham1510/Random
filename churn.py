# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 16:11:50 2018

@author: shubham.kumar
"""

import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.backends.backend_pdf
import numpy as np
import math
from scipy import stats
from sklearn.utils.multiclass import type_of_target
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from collections import Counter
import time

from scipy.spatial.distance import squareform,pdist
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

import zipfile

################################# MODULE1 ##############################################
    
def module1(filePath,dep_var):
    ## Timestamp
    timestr = time.strftime("%Y%m%d-%H%M%S")
    Module1_Data = pd.read_csv(filePath)
    wd_start = os.getcwd()
    
    wd = os.getcwd() + '/outputs'
    newpath = wd + '/EDA_Results_' + timestr
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    os.chdir(newpath)
    
    Module1_Data = cont_to_categorical(Module1_Data)
    ##Separating cont and categorical variable
    Module1_Data_cont = Module1_Data.select_dtypes(exclude=['category','object','bool'])
    Module1_Data_cat = Module1_Data.select_dtypes(include=['category','object','bool'])
    
    frequency_plots(Module1_Data_cont,Module1_Data_cat)
    
    # Dropping columns having more than 50% NA
    Module1_Data = Module1_Data.dropna(thresh=len(Module1_Data)/2, axis=1)
    
    inference_plots(Module1_Data_cont,Module1_Data_cat,Module1_Data,dep_var)
    
    os.chdir(wd)
    wd_for_zip = wd + "/EDA_Results_" + timestr
    ##Final output data for Module 1
    zip_files(wd,wd_for_zip,timestr,'EDA_Results_')      
    Module1_Data.to_csv("EDA_Data_"+ timestr +".csv",index=False)
    
    os.chdir(wd_start)
    outputs_module1 = {
                      "dependentVar" : dep_var,
                      "workingDirectory" : wd,
                      "files" : {
                          "FileNameToBeUsed" :[{
                            "fileName" : 'EDA_Data_'+ timestr +'.csv',
                            "filePath" : wd + '/EDA_Data_'+ timestr +'.csv' 
                          }],
                          "ZipFolder" : {
                            "fileName" : 'EDA_Results_' + timestr + '.zip',
                            "filePath" : wd + '/EDA_Results_' + timestr + '.zip'
                        }
                      }
                    }
                          
    return outputs_module1
    
        
def zip_files(wd, wd_for_zip,timestr,fileName):
    abs_src = os.path.abspath(wd_for_zip)
    zf = zipfile.ZipFile(fileName + timestr + '.zip', "w")
    for dirname, subdirs, files in os.walk(wd_for_zip):
        for filename in files:
            absname = os.path.abspath(os.path.join(dirname, filename))
            arcname = absname[len(abs_src):]
            zf.write(absname, arcname)
            
    zf.close()
    
    

def cont_to_categorical(Module1_Data):
    #Converting independent Variable with less than 10 unique entries into category
    for column in Module1_Data:
        if(Module1_Data[column].unique().size < 10):
            if(Module1_Data[column].dtype== 'bool'):
                continue
            Module1_Data[column] = Module1_Data[column].astype('category')
    
    ## Storing the description of varibles
    Summary_Statistics= Module1_Data.describe().transpose()
    Summary_Statistics.to_csv("Summary_Statistics.csv")
    return Module1_Data 

def frequency_plots(Module1_Data_cont,Module1_Data_cat):
    #Frequency Distribution of continuous variables
    pdf = matplotlib.backends.backend_pdf.PdfPages('Frequency_Distribution_of_continuous_variables.pdf')
    for column in Module1_Data_cont:
        fig = plt.figure(figsize=(10, 10)) 
        plt.hist(Module1_Data_cont[column], facecolor='green')
        plt.title("Frequency distribution of " + column)
        plt.xlabel(column)
        plt.ylabel('Frequency')
        pdf.savefig(fig)
    pdf.close()
        
    #Frequency Distribution of categorical variables
    pdf = matplotlib.backends.backend_pdf.PdfPages('Frequency_Distribution_of_categorical_variables.pdf')
    for column in Module1_Data_cat:
        fig = plt.figure(figsize=(10, 10))
        df = Module1_Data_cat[column].value_counts()
        df = df.sort_index()
        df.plot(kind = 'bar')  
        plt.title("Frequency distribution of " + column)
        plt.xlabel(column)
        plt.ylabel("Frequency")
        pdf.savefig(fig)
    pdf.close()


def inference_plots(Module1_Data_cont,Module1_Data_cat,Module1_Data,dep_var):
    #Inference plots for continuous variable
    Cont_plot_data = pd.concat([Module1_Data_cont.reset_index(drop=True), Module1_Data[dep_var]], axis=1)
    pdf = matplotlib.backends.backend_pdf.PdfPages('Inference_plots_for_continuous_variable.pdf')
    for column in Cont_plot_data:
        if(column != dep_var):
            fig = plt.figure(figsize=(10, 10))
            sns.barplot(x=dep_var, y=column, data=Cont_plot_data)
            plt.title("Average " + column)
            pdf.savefig(fig)
    pdf.close()
    
    #Inference plots for categorical variable
    pdf = matplotlib.backends.backend_pdf.PdfPages('Inference_plots_for_categorical_variable.pdf')
    for column in Module1_Data_cat:
        if(column != dep_var):
            fig = plt.figure(figsize=(10, 10))
            sns.countplot(x=column, hue=dep_var, data=Module1_Data_cat)
            plt.title(column)
            pdf.savefig(fig)
    pdf.close()
    

    
################################# MODULE 1 END ##############################################
    
################################# MODULE 2 ##################################################
def inputData_module2(Input_Data_Module2):
    for column in Input_Data_Module2:
        if(Input_Data_Module2[column].unique().size < 10):
            if(Input_Data_Module2[column].dtype== 'bool'):
                continue
            Input_Data_Module2[column] = Input_Data_Module2[column].astype('category')
    
    
    Module2_Data_cont = Input_Data_Module2.select_dtypes(exclude=['category','object','bool'])
    Module2_Data_cat = Input_Data_Module2.select_dtypes(include=['category','object','bool'])
    return Input_Data_Module2,Module2_Data_cont,Module2_Data_cat

def no_var_removal(Module2_Data_cont):
    ##Remove varibles with no variance
    for column in Module2_Data_cont:
        if(max(Module2_Data_cont[column].value_counts()) > len(Module2_Data_cont)/2):
            Module2_Data_cont = Module2_Data_cont.drop([column],axis=1)

def outlier_and_NA_treatment(Module2_Data_cont,Module2_Data_cat):
    ##Replacing NA values in cont variables
    for column in Module2_Data_cont:
        if(Module2_Data_cont[column].mean() <= 1.1*(Module2_Data_cont[column].median()) and (Module2_Data_cont[column].mean()) >= 0.9*(Module2_Data_cont[column].median())):
            Module2_Data_cont[column] = Module2_Data_cont[column].fillna(Module2_Data_cont[column].mean())
        else:
            Module2_Data_cont[column] = Module2_Data_cont[column].fillna(Module2_Data_cont[column].median())
    
    ##Replacing NA values in categorical variables
    for column in Module2_Data_cat:
        zxc = list(Module2_Data_cat[column])
        var = max(set(zxc), key=zxc.count)
        Module2_Data_cat[column].fillna(var, inplace=True)
    
    ##Outlier Treatment
    for col in Module2_Data_cont.columns:
        percentiles = Module2_Data_cont[col].quantile([0.01,0.99]).values
        Module2_Data_cont[col][Module2_Data_cont[col] <= percentiles[0]] = percentiles[0]
        Module2_Data_cont[col][Module2_Data_cont[col] >= percentiles[1]] = percentiles[1]
        
def module2(filePath,dep_var,wd):
    ## Timestamp
    timestr = time.strftime("%Y%m%d-%H%M%S")
    Input_Data_Module2 = pd.read_csv(filePath)
    os.chdir(wd)
    Input_Data_Module2,Module2_Data_cont,Module2_Data_cat = inputData_module2(Input_Data_Module2)
    no_var_removal(Module2_Data_cont)
    outlier_and_NA_treatment(Module2_Data_cont,Module2_Data_cat)

    # Final output data for Module 2
    Output_Data_Module2 = pd.concat([Module2_Data_cont.reset_index(drop=True), Module2_Data_cat], axis=1)
    Output_Data_Module2.to_csv("Variable_cleaning_results_"+ timestr +".csv",index=False)
    
    column_list = list(Output_Data_Module2)
    column_list.remove(dep_var)
    
    step1_name_list = ["Factor Analysis","Principal Component Analysis","K-Means"]
    step2_name_list = ["Gradient Boosting Machine" ,"Information Value"]
    
    output_dict_2 = {
                      "inputFileName" : "Variable_cleaning_results_"+ timestr +".csv",
                      "inputFilePath" : wd + "/Variable_cleaning_results_"+ timestr +".csv",
                      "dependentVar" : dep_var,
                      "workingDirectory" : wd,
                         "columns" : {
                          "selected" : [],
                          "default" : "",
                          "columnList" : column_list
                        },
                        "step1" : {
                          "selected" : "",
                          "default" : "Factor Analysis",
                          "step1ValueList" : step1_name_list
                        },
                        "step2" : {
                          "selected" : "",
                          "default" : "Gradient Boosting Machine",
                          "step2ValueList" : step2_name_list
                        } 
                      
                    }
    return output_dict_2

################################# MODULE 2 END ##############################################
    
################################# MODULE 3 ##################################################
    
class WOE:
    def __init__(self):
        self._WOE_MIN = 0
        self._WOE_MAX = 0

    def woe(self, X, y, event=1):
        '''
        Calculate woe of each feature category and information value
        :param X: 2-D numpy array explanatory features which should be discreted already
        :param y: 1-D numpy array target variable which should be binary
        :param event: value of binary stands for the event to predict
        :return: numpy array of woe dictionaries, each dictionary contains woe values for categories of each feature
                 numpy array of information value of each feature
        '''
        self.check_target_binary(y)
        X1 = self.feature_discretion(X)

        res_woe = []
        res_iv = []
        for i in range(0, X1.shape[-1]):
            x = X1[:, i]
            woe_dict, iv1 = self.woe_single_x(x, y, event)
            res_woe.append(woe_dict)
            res_iv.append(iv1)
        return np.array(res_woe), np.array(res_iv)

    def woe_single_x(self, x, y, event=1):
        '''
        calculate woe and information for a single feature
        :param x: 1-D numpy starnds for single feature
        :param y: 1-D numpy array target variable
        :param event: value of binary stands for the event to predict
        :return: dictionary contains woe values for categories of this feature
                 information value of this feature
        '''
        self.check_target_binary(y)

        event_total, non_event_total = self.count_binary(y, event=event)
        x_labels = np.unique(x)
        woe_dict = {}
        iv = 0
        for x1 in x_labels:
            y1 = y[np.where(x == x1)[0]]
            event_count, non_event_count = self.count_binary(y1, event=event)
            rate_event = 1.0 * event_count / event_total
            rate_non_event = 1.0 * non_event_count / non_event_total
            if rate_event == 0:
                woe1 = self._WOE_MIN
            elif rate_non_event == 0:
                woe1 = self._WOE_MAX
            else:
                woe1 = math.log(rate_event / rate_non_event)
            woe_dict[x1] = woe1
            iv += (rate_event - rate_non_event) * woe1
        return woe_dict, iv

    def woe_replace(self, X, woe_arr):
        '''
        replace the explanatory feature categories with its woe value
        :param X: 2-D numpy array explanatory features which should be discreted already
        :param woe_arr: numpy array of woe dictionaries, each dictionary contains woe values for categories of each feature
        :return: the new numpy array in which woe values filled
        '''
        if X.shape[-1] != woe_arr.shape[-1]:
            raise ValueError('WOE dict array length must be equal with features length')

        res = np.copy(X).astype(float)
        idx = 0
        for woe_dict in woe_arr:
            for k in woe_dict.keys():
                woe = woe_dict[k]
                res[:, idx][np.where(res[:, idx] == k)[0]] = woe * 1.0
            idx += 1

        return res

    def combined_iv(self, X, y, masks, event=1):
        '''
        calcute the information vlaue of combination features
        :param X: 2-D numpy array explanatory features which should be discreted already
        :param y: 1-D numpy array target variable
        :param masks: 1-D numpy array of masks stands for which features are included in combination,
                      e.g. np.array([0,0,1,1,1,0,0,0,0,0,1]), the length should be same as features length
        :param event: value of binary stands for the event to predict
        :return: woe dictionary and information value of combined features
        '''
        if masks.shape[-1] != X.shape[-1]:
            raise ValueError('Masks array length must be equal with features length')

        x = X[:, np.where(masks == 1)[0]]
        tmp = []
        for i in range(x.shape[0]):
            tmp.append(self.combine(x[i, :]))

        dumy = np.array(tmp)
        # dumy_labels = np.unique(dumy)
        woe, iv = self.woe_single_x(dumy, y, event)
        return woe, iv

    def combine(self, list):
        res = ''
        for item in list:
            res += str(item)
        return res

    def count_binary(self, a, event=1):
        event_count = (a == event).sum()
        non_event_count = a.shape[-1] - event_count
        return event_count, non_event_count

    def check_target_binary(self, y):
        '''
        check if the target variable is binary, raise error if not.
        :param y:
        :return:
        '''
        y_type = type_of_target(y)
        if y_type not in ['binary']:
            raise ValueError('Label type must be binary')

    def feature_discretion(self, X):
        '''
        Discrete the continuous features of input data X, and keep other features unchanged.
        :param X : numpy array
        :return: the numpy array in which all continuous features are discreted
        '''
        temp = []
        for i in range(0, X.shape[-1]):
            x = X[:, i]
            x_type = type_of_target(x)
            if x_type == 'continuous':
                x1 = self.discrete(x)
                temp.append(x1)
            else:
                temp.append(x)
        return np.array(temp).T

    def discrete(self, x):
        '''
        Discrete the input 1-D numpy array using 5 equal percentiles
        :param x: 1-D numpy array
        :return: discreted 1-D numpy array
        '''
        res = np.array([0] * x.shape[-1], dtype=int)
        for i in range(5):
            point1 = stats.scoreatpercentile(x, i * 20)
            point2 = stats.scoreatpercentile(x, (i + 1) * 20)
            x1 = x[np.where((x >= point1) & (x <= point2))]
            mask = np.in1d(x, x1)
            res[mask] = (i + 1)
        return res

    @property
    def WOE_MIN(self):
        return self._WOE_MIN
    @WOE_MIN.setter
    def WOE_MIN(self, woe_min):
        self._WOE_MIN = woe_min
    @property
    def WOE_MAX(self):
        return self._WOE_MAX
    @WOE_MAX.setter
    def WOE_MAX(self, woe_max):
        self._WOE_MAX = woe_max




class Varclus:
    def __init__(self, df, stand_scale = True, maxeigen=0.7, report_method='r-square_ratio'):
        '''
        Input DataFrame : Features with Mean = 0 and std = 1
        
        If input is not Standardized then give stand_scale = True, else give False
        
        maxeigen: 2nd EigenValue threshold in each cluster (based on cluster correllaton matrix)
            Higher value will result in less clusters
            Lower value will result in more clsuters
        
        report_method:
            'correlation' : Within each cluster select the feature that has the maximum correlation with others
            'centroid' : Within each cluster select the feature that is closest to the centroid of the cluster
            'r-square_ratio': minimize (1-r_squre)/(1-r_square_nearest_cluster)
            'closest_to_PCA' : feature closest to first component of PCA of the cluster
        '''
        self.maxeigen = maxeigen
        self.report_method = report_method
        self.df = df
        self.stand_scale = stand_scale
    
    def standerdize(self):
        colnames = self.df.columns.tolist()
        SC = StandardScaler(copy=False, with_mean=True, with_std=True)
        return pd.DataFrame(SC.fit_transform(self.df), columns = colnames)

    def varclus_rec(self, X, maxeigen):
        if X.shape[1] < 2:
            return list(X.columns)
        PC = PCA(n_components=2, copy=True, whiten=False, svd_solver='auto', tol=0.0, iterated_power='auto', random_state=None)
        PC.fit(X)
        if PC.explained_variance_[1]>=maxeigen:
            Clusters = []
            for var in range(X.shape[1]):
                Clusters.append(np.argmax(np.abs(PC.components_[:,var])))
            vars1 = [X.columns[i] for i,x in enumerate(Clusters) if x==0]
            vars2 = [X.columns[j] for j,x in enumerate(Clusters) if x==1]

            return [self.varclus_rec(X[vars1], maxeigen), self.varclus_rec(X[vars2], maxeigen)]
        else:
            return list(X.columns)  
        
    def fit(self):
        start = time()
        if self.stand_scale ==True:
            print("Standardizing data")
            self.df = self.standerdize()
        else:
            print ("Data is already Standardized")
            
        print("Recursive PCA Hierarchical Clustering")
        print("Number of features = " + str(len(self.df.columns)))
        self.heir_list = self.varclus_rec(self.df, self.maxeigen)
        print ("Time taken = " + str(round((time()-start)/60,2)) + "min")
    
    def flatten(self, S):
        if len(S)==0:
            return S
        if type(S[0])== str:
            return S
        elif type(S[0])==list:
            if type(S[0][0])==str and type(S[1][0])==str:
                return S
            elif type(S[0][0])==str and type(S[1][0])==list:
                return [S[0]] + self.flatten(S[1])
            elif type(S[0][0])==list and type(S[1][0])==str:
                return self.flatten(S[0]) + [S[1]]
            else:
                return self.flatten(S[0])+self.flatten(S[1])
        
    def cluster_list(self):
        self.flat_list = self.flatten(self.heir_list)
        print ("Number of clusters = " + str(len(self.flat_list)))
        return self.flat_list
            
    def report(self):
        start = time()
        all_feats = []
        clus_nums = []
        report = pd.DataFrame()
        for i,clus in enumerate(self.flat_list):
            all_feats+=clus
            clus_nums+=list(np.zeros(len(clus),dtype=int)+i)
        C = Counter(clus_nums)
        a = pd.DataFrame()
        a['feat_name'] = all_feats
        a['cluster'] = clus_nums
        clus_list = C.keys()
        selected_feats = []

        if self.report_method== 'r-square_ratio':
            all_clus_PCA = []
            for clus in clus_list:
                clus_feats = list(a[a['cluster']==clus]['feat_name'])
                pca = PCA(n_components=1)
                trans = pca.fit_transform(self.df[clus_feats])
                all_clus_PCA.append(trans[:,0])
            inter_clus_d = squareform(pdist(np.array(all_clus_PCA)))
            nearest_neighbours = np.argmin(inter_clus_d + np.max(inter_clus_d)*np.identity(len(clus_list)),axis = 1)
        report_list = []
        for clus_num,clus in enumerate(clus_list):
            clus_feats = list(a[a['cluster']==clus]['feat_name'])
            if len(clus_feats)==0:
                continue
            else:
                if self.report_method== 'correlation':
                    cols = ['cluster','Variable','sum_cross_corr']
                    corre = self.df[clus_feats].corr().fillna(-1)
                    sum_cross_corr = np.sum(abs((np.array(corre) - np.identity(corre.shape[0]))),axis = 1)
                    for i,feat in enumerate(clus_feats):
                        report_list.append([clus_num,feat,sum_cross_corr[i]])
                    selected_feat = clus_feats[np.argmax(sum_cross_corr)]
                    selected_feats.append(selected_feat)
                    report = pd.DataFrame(report_list,columns = cols)

                elif self.report_method== 'r-square_ratio':
                    cols = 'Cluster Variable R2_Own Next_Closest R2_NC R2_Ratio'.split()
                    r2_ratio_list = []
                    for feat in clus_feats:
                        lrg = LinearRegression()
                        lrg.fit(self.df[feat].values.reshape(self.df[feat].shape[0],1),all_clus_PCA[clus_num].reshape(all_clus_PCA[clus_num].shape[0],1))
                        r2 = lrg.score(self.df[feat].values.reshape(self.df[feat].shape[0],1),all_clus_PCA[clus_num].reshape(all_clus_PCA[clus_num].shape[0],1))

                        lrg = LinearRegression()
                        lrg.fit(self.df[feat].values.reshape(self.df[feat].shape[0],1),all_clus_PCA[nearest_neighbours[clus_num]].reshape(all_clus_PCA[clus_num].shape[0],1))
                        r2_nearestClus = lrg.score(self.df[feat].values.reshape(self.df[feat].shape[0],1),all_clus_PCA[nearest_neighbours[clus_num]].reshape(all_clus_PCA[clus_num].shape[0],1))
                        if r2_nearestClus==1:
                            r2_ratio = round((1-r2)/(1-0.99999999999),2)
                        else:
                            r2_ratio = round((1-r2)/(1-r2_nearestClus),2)
                        r2_ratio_list.append(r2_ratio)
                        report_list.append([clus_num,feat,round(r2,2),nearest_neighbours[clus_num],round(r2_nearestClus,2),r2_ratio])
                    report = pd.DataFrame(report_list,columns = cols)
                    selected_feat = clus_feats[np.argmin(r2_ratio_list)]
                    selected_feats.append(selected_feat)
                elif self.report_method== 'centroid':
                    cols = ['cluster','Variable','Distance_to_centroid']
                    centroid = np.array(self.df[clus_feats].mean(axis=1))
                    dists = []
                    for f in clus_feats:
                        distance = np.linalg.norm(np.array(df1[f])-centroid)
                        dists.append(distance)
                        report_list.append([clus_num,f,distance])    
                    selected_feat = clus_feats[np.argmin(dists)]
                    selected_feats.append(selected_feat)
                    report = pd.DataFrame(report_list,columns = cols)
                elif self.report_method=='closest_to_PCA':
                    cols = ['cluster','Variable','Distance_to_PCA']
                    pca = PCA()
                    trans = pca.fit_transform(self.df[clus_feats])
                    dists = []
                    for feat in clus_feats:
                        distance = np.linalg.norm(np.array(df1[feat])-trans[:,0])
                        dists.append(distance)
                        report_list.append([clus_num,feat,distance])
                    selected_feat = clus_feats[np.argmin(dists)]
                    selected_feats.append(selected_feat)
                    report = pd.DataFrame(report_list,columns=cols)
        print( "Time taken = " + str(round((time()-start)/60,2)) + "mins")
        return report


def data_prep_step1(input_3,dep_var):
    numericData =  pd.read_csv("Output_Data_Module2.csv")        
    from sklearn.preprocessing import LabelEncoder
    lb_make = LabelEncoder()
    from sklearn.preprocessing import LabelBinarizer
    lb_make1 = LabelBinarizer()
    var_for_conversion = list(numericData.select_dtypes(include=['category','object','bool']))
    for column in var_for_conversion:
        if(numericData[column].dtype == 'bool'):
            numericData[column] = lb_make1.fit_transform(numericData[column])
        else:
            numericData[column] = lb_make.fit_transform(numericData[column])
    del(var_for_conversion)
    
    y = numericData[dep_var]
    X = numericData.drop([dep_var],axis=1)
    return X,y;    

def pca(X):
    "Specifying the parameters for clustering"
    clus = Varclus(X, stand_scale = True, maxeigen=0.7, report_method='r-square_ratio')
    "Results got clustering"
    clus.fit()
    "variable list cluster-wise"
    clus.cluster_list()
    "Final Report"
    Y = clus.report()
    "Dense ranking the variables"
    Y["group_rank"] = Y.groupby("Cluster")["R2_Ratio"].rank(ascending=1,method='dense')
    "Selecting the variable with min R2 ration in each cluster"
    df_sel_vars = Y.query('group_rank==1')
    "Resultant dataset"
    df_clus = X[df_sel_vars['Variable']]
    selected_variables = list(df_clus)
    number_of_selected_variables = len(selected_variables)
    return selected_variables,number_of_selected_variables
    
def kmeans(X):
    import numpy as np
    from scipy.cluster.vq import kmeans,vq
    from scipy.spatial.distance import cdist
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    
    ##### cluster data into K=1..10 clusters #####
    K = range(10,50)
    x = scaler.fit_transform( X )
    x = x.transpose()
    
    # X = scaler.fit_transform( X )
    # X.head()
    # scipy.cluster.vq.kmeans
    KM = [kmeans(x,k) for k in K]
    centroids = [cent for (cent,var) in KM]   # cluster centroids
    #avgWithinSS = [var for (cent,var) in KM] # mean within-cluster sum of squares
    
    # alternative: scipy.cluster.vq.vq
    #Z = [vq(X,cent) for cent in centroids]
    #avgWithinSS = [sum(dist)/X.shape[0] for (cIdx,dist) in Z]
    
    # alternative: scipy.spatial.distance.cdist
    D_k = [cdist(x, cent, 'euclidean') for cent in centroids]
    cIdx = [np.argmin(D,axis=1) for D in D_k]
    dist = [np.min(D,axis=1) for D in D_k]
    avgWithinSS = [sum(d)/x.shape[0] for d in dist]
    hello = 0
    minval = avgWithinSS[0]
    optimal_k = 10
    for i in avgWithinSS:
        hello += 1
        if(i<minval):
            optimal_k = hello
            minval = i
    x = scaler.fit_transform( X )
    import pandas as pd
    x = pd.DataFrame(x)
    x.columns = X.columns
    x = x.transpose()
    df1 = x
    df = x
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=optimal_k, init='k-means++', n_init=10)
    km.fit(df1)
    x = km.fit_predict(df1)
    df["Cluster"]= x
    df.head(10)
    df["Cluster"].head()
    selected_variables = list((df.sort_values('Cluster', ascending=True).groupby('Cluster').head(1)).index)
    number_of_selected_variables = len(selected_variables)
    return selected_variables,number_of_selected_variables
    
def factor_analysis(X):
    from factor_analyzer import FactorAnalyzer
    import numpy as np
    from numpy import linalg as LA
    fa = FactorAnalyzer()
    ##Correlation matrix of the variables##
    corr=X.corr()
    ##Eigen values##
    w,v=LA.eig(corr)
    w = w.astype(np.float64)
    y = pd.DataFrame(w)
    ##The optimal number of clusters##
    z=y[y > 1.0].count()
    
    ##Factor Analysis
    fa.analyze(corr, z[0], rotation="varimax")
    ##Loadings of the factor analysis are stored in abc##
    abc = fa.loadings
    abc = abc.astype(np.float64)
    
    
    ##Finding the absolute values
    for column in abc:
        abc[column] = abc[column].abs()
    ##Storing the name of reduced variables after factor analysis
    a=[]
    for column in abc:    
        a.append(abc[column].idxmax())
    
    
    a = pd.DataFrame(a)
    a.columns = ["1"]
    selected_variables = list(a["1"])
    number_of_selected_variables = len(selected_variables)
    return selected_variables,number_of_selected_variables

def step1_output(input_3,selected_variables,dep_var,timestr):    
    output_module3_step1 = pd.DataFrame(input_3, columns=selected_variables)
    output_module3_step1[dep_var] = input_3[dep_var]
    output_module3_step1.to_csv("Variable_Reduction_step1_results_"+ timestr +".csv",index=False)
    return output_module3_step1

def data_prep_step2(output_module3_step1,dep_var):
    numericData = output_module3_step1.copy(deep=True)
    from sklearn.preprocessing import LabelEncoder
    lb_make = LabelEncoder()
    from sklearn.preprocessing import LabelBinarizer
    lb_make1 = LabelBinarizer()
    var_for_conversion = list(numericData.select_dtypes(include=['category','object','bool']))
    for column in var_for_conversion:
        if(numericData[column].dtype == 'bool'):
            numericData[column] = lb_make1.fit_transform(numericData[column])
        else:
            numericData[column] = lb_make.fit_transform(numericData[column])
    
    del(var_for_conversion)
    y = numericData[dep_var]
    X = numericData.drop([dep_var],axis=1)
    return X,y

def information_Value(X,y):
    import numpy as np
    import pandas as pd
    hello = X.copy(deep=True)
    feature_names = list(hello)
    
    X = np.asarray(X)
    y = np.asarray(y)
    
    obj1 = WOE()
    woe,iv = obj1.woe(X,y,event=1)
    
    iv = pd.DataFrame(iv)
    iv.columns = ["IV"]
    iv["Feature_Names"] = feature_names
    
    iv = iv.sort_values(['IV'],ascending = False)
    selected_features_2 = list(iv["Feature_Names"].head(10))
    number_of_selected_features_2 = len(selected_features_2)
    return selected_features_2,number_of_selected_features_2

def gradient_Boosting_Machine(X,y):
    import numpy as np
    import matplotlib.pyplot as plt
    
    from sklearn import ensemble     
    # #############################################################################
    # Fit regression model
    params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
              'learning_rate': 0.01, 'loss': 'ls'}
    clf = ensemble.GradientBoostingRegressor(**params)
    
    clf.fit(X, y)
    
    # #############################################################################
    # Plot feature importance
    feature_names = list(X)
    feature_importance = clf.feature_importances_
    # make importances relative to max importance
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    relative_importance = pd.DataFrame()
    relative_importance["Names"] = feature_names
    relative_importance["Importance"] = pd.DataFrame(feature_importance)
    relative_importance = relative_importance.sort_values(["Importance"],ascending = False)
    relative_importance = relative_importance.reset_index(drop=True)
    sorted_idx = np.argsort(feature_importance)
    n=10
    if(sorted_idx.shape[0] < 10):
        n= sorted_idx.shape[0]
    pos = np.arange(n) + .5
    plt.subplot(1, 2, 2)
    plt.barh(pos, np.array(relative_importance["Importance"].head(n)), align='center')
    plt.yticks(pos, list(relative_importance["Names"].head(n)))
    plt.xlabel('Relative Importance')
    plt.title('Variable Importance')
    selected_features_2 = list(relative_importance["Names"].head(n))
    number_of_selected_features_2 = len(selected_features_2)
    return selected_features_2,number_of_selected_features_2
    
def step2_output(input_3,selected_features_2,dep_var,timestr):
    output_module3 = pd.DataFrame(input_3, columns=selected_features_2)
    output_module3[dep_var] = input_3[dep_var]
    output_module3.to_csv("Variable_Reduction_step2_results_"+ timestr +".csv",index=False)    
        

def module3(step1,step2,dep_var, already_selected_vars,filePath,wd):
    ## Timestamp
    timestr = time.strftime("%Y%m%d-%H%M%S")
    
    input_3 = pd.read_csv(filePath)
    os.chdir(wd)
    X,y = data_prep_step1(input_3,dep_var)
       
    if(step1 == "Pricipal Component Analysis"):
        selected_variables,number_of_selected_variables = pca(X)
    elif(step1 == "K-Means"):
        selected_variables,number_of_selected_variables = kmeans(X)
    elif(step1 == "Factor Analysis"):
        selected_variables,number_of_selected_variables = factor_analysis(X)
    else:
        print("Error")
    
    output_module3_step1 = step1_output(input_3,selected_variables,dep_var,timestr)
   
    X,y = data_prep_step2(output_module3_step1,dep_var)
    if(step2 == "Information Value"):
        selected_features_2,number_of_selected_features_2 = information_Value(X,y)
    elif(step2 == "Gradient Boosting Machine"):
        selected_features_2,number_of_selected_features_2 = gradient_Boosting_Machine(X,y)
    
    selected_features_2 = set().union(selected_features_2,already_selected_vars)
    step2_output(input_3,selected_features_2,dep_var,timestr)
    
    model_list = ["Logistic Regression","Naive Bayes","Random Forest","Artificial Neural Network"]
    output_dict_3 =  {
                      "files" : [{
                              "fileName" : "Variable_Reduction_step1_results_"+ timestr +".csv",
                              "filePath" : wd + "/Variable_Reduction_step1_results_"+ timestr +".csv"
                              },{
                              "fileName" : "Variable_Reduction_step2_results_"+ timestr +".csv",
                              "filePath" : wd + "/Variable_Reduction_step2_results_"+ timestr +".csv"
                              }],
                      "dependentVar" : dep_var,
                      "workingDirectory" : wd,
                      "models" : {
                          "modelList" : model_list,
                          "selected" : []
                      }
                    }
    return output_dict_3

################################# MODULE 3 END ##############################################
   
################################# MODULE 4 ##################################################


def module4(filePath,dep_var,wd,models):
    ## Timestamp
    timestr = time.strftime("%Y%m%d-%H%M%S")
    
    newpath = wd + '/Modelling_reports_' + timestr
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    os.chdir(newpath)
    files_list = []
    display_list = []

    X_train, X_test, y_train, y_test, X_validate, y_validate,dep_var, X, y = data_prep_module4(filePath,dep_var)
    if("Logistic Regression" in models):
       model_train, model_test = log_reg_models(X_train, X_test, y_train, y_test)
       x = report_making('LogisticRegression_' + timestr + '.pdf',model_train, model_test,X_train, X_test, y_train, y_test, X_validate, y_validate,dep_var)
       display_list.append({"Logistic Regression" : x})
       addfile = {
               "fileName" : 'LogisticRegression_' + timestr + '.pdf',
               "filePath" : newpath + '/LogisticRegression_' + timestr + '.pdf'
               }
       files_list.append(addfile)
    
    if("Naive Bayes" in models):
       model_train, model_test = naive_bayes_models(X_train, X_test, y_train, y_test)
       x = report_making('NaiveBayes_'+timestr +'.pdf',model_train, model_test,X_train, X_test, y_train, y_test, X_validate, y_validate,dep_var)
       display_list.append({"Logistic Regression" : x})
       addfile = {
               "fileName" : 'NaiveBayes_'+timestr +'.pdf',
               "filePath" : newpath + '/NaiveBayes_' + timestr +'.pdf'
               }
       files_list.append(addfile)
    
    if("Random Forest" in models):
       model_train, model_test = random_forest_models(X_train, X_test, y_train, y_test)
       x = report_making('RandomForest_'+ timestr +'.pdf',model_train, model_test,X_train, X_test, y_train, y_test, X_validate, y_validate, dep_var)
       display_list.append({"Logistic Regression" : x})
       addfile = {
               "fileName" : 'RandomForest_'+ timestr +'.pdf',
               "filePath" : newpath + '/RandomForest_'+ timestr +'.pdf'
               }
       files_list.append(addfile)
    
    if("Artificial Neural Network" in models):
       model_train, model_test = ann_models(X_train, X_test, y_train, y_test,X)
       x = report_making_ann('ArtificialNeuralNetwork_'+ timestr +'.pdf',model_train, model_test,X_train, X_test, y_train, y_test, X_validate, y_validate,dep_var)
       display_list.append({"Logistic Regression" : x})
       addfile = {
               "fileName" : 'ArtificialNeuralNetwork_'+ timestr + '.pdf',
               "filePath" : newpath + '/ArtificialNeuralNetwork_'+ timestr + '.pdf'
               }
       files_list.append(addfile)
  
    os.chdir(wd)
    zip_files(wd, newpath ,timestr,'Modelling_reports_')
    
    zipfolder = { 'fileName' : 'Modelling_reports_' + timestr + '.zip',
                  'filePath' : wd + '/Modelling_reports_' + timestr + '.zip'
            }
    output_dict_4 =  {"display" : display_list,
                      "files" :[{"download" : files_list
                              },
                              {"zipFolder" : zipfolder}
                              ],
                      "dependentVar" : dep_var,
                      "workingDirectory" : wd
                     } 
    
    return output_dict_4

modeling_data=pd.read_csv("output_module3.csv")
def data_prep_module4(filePath,dep_var):
    #################################################################################
    modeling_data = pd.read_csv(filePath)
    target = dep_var
    from sklearn.preprocessing import LabelEncoder
    lb_make = LabelEncoder()
    from sklearn.preprocessing import LabelBinarizer
    lb_make1 = LabelBinarizer()
    var_for_conversion = list(modeling_data.select_dtypes(include=['category','object','bool']))
    for column in var_for_conversion:
        if(modeling_data[column].dtype == 'bool'):
            modeling_data[column] = lb_make1.fit_transform(modeling_data[column])
        else:
            modeling_data[column] = lb_make.fit_transform(modeling_data[column])
    
    del(var_for_conversion)
    y = modeling_data[target]
    modeling_data = modeling_data.drop([target],axis=1)
    #################################################################################
        
    X, X_validate, y, y_validate = train_test_split(modeling_data, y, test_size=0.1,random_state=123)
    y = y.reset_index(drop=True)
    X = X.reset_index(drop=True)
    X_train,X_test,y_train,y_test= train_test_split(X, y, test_size=0.2,random_state=123)
    return X_train, X_test, y_train, y_test, X_validate, y_validate, dep_var, X, y

def log_reg_models(X_train, X_test, y_train, y_test):
    ############################## LOGISTIC REGRESSION ###############################
    from sklearn.linear_model import LogisticRegression
    ######## on training dataset
    LogReg = LogisticRegression(random_state=123)
    model_train = LogReg.fit(X_train, y_train)
    
    ####### on test dataset
    LogReg = LogisticRegression(random_state=123)
    model_test = LogReg.fit(X_test, y_test)
    return model_train, model_test

def report_making(x,model_train, model_test,X_train, X_test, y_train, y_test, X_validate, y_validate,dep_var):
    probs_train = model_train.predict_proba(X_train)
    result = []
        
    ##Model Scoring
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.pyplot as plt
    with PdfPages(x) as pdf:
    
        train_y = pd.DataFrame(y_train)
        train_y=train_y.reset_index(drop=True)
        probs_train=pd.DataFrame(model_train.predict_proba(X_train))
        probs_train.columns=["prob_0","prob_1"]
        del probs_train["prob_0"]
        train_y=pd.concat([probs_train.reset_index(drop=True), train_y.reset_index(drop=True)], axis=1)
        train_y['decile'] = pd.qcut(train_y['prob_1'].rank(method='first'), 10,labels=np.arange(10, 0, -1))
        a = pd.DataFrame()
        #############################
        hello =  train_y.groupby(["decile"]).prob_1.agg(['min'])
        x = hello['min'].tolist()
        b = pd.DataFrame()
        b["Min_Prob"] = x 
        #############################
        a["Min_Prob"]=train_y.groupby(["decile"]).prob_1.agg(['min']).reset_index(drop=True)
        ###############################
        hello = train_y.groupby(["decile"]).prob_1.agg(['max'])
        b["Max_Prob"] = hello['max'].tolist()
        b["Decile"] = [10,9,8,7,6,5,4,3,2,1]
        ##############################
        a["Max_Prob"] = train_y.groupby(["decile"]).prob_1.agg(['max']).reset_index(drop=True)
        a["Decile"] = [10,9,8,7,6,5,4,3,2,1]
        non_churn_count =train_y[train_y[dep_var] == 0]
        a["Non_churn_count"]=non_churn_count.groupby(["decile"]).size().reset_index(drop=True)
        churn_count =train_y[train_y[dep_var] == 1]
        a["Churn_count"]=churn_count.groupby(["decile"]).size().reset_index(drop=True)
        a=a.sort_values("Decile")
        #############################################
        hello = non_churn_count.groupby(["decile"]).size()
        b["Non_churn_count"] = list(hello)
        hello = churn_count.groupby(["decile"]).size()
        b["Churn_count"]= list(hello)
        b=b.sort_values("Decile")
        #############################################
        a["CumSum_Non_Churn"]=a.loc[:,"Non_churn_count"].cumsum()
        a["CumSum_Churn"]=a.loc[:,"Churn_count"].cumsum()
        a["Perc_cumsum_Churn"]=100*a["CumSum_Churn"]/a['Churn_count'].sum()
        a["Perc_cumsum_non_Churn"]=100*a["CumSum_Non_Churn"]/a['Non_churn_count'].sum()
        a["Perc_diff"]=a["Perc_cumsum_Churn"]-a["Perc_cumsum_non_Churn"]
    
    
    ###Display input probability###
        Input_probability=pd.DataFrame(a.Min_Prob[a.Perc_diff == max(a.Perc_diff)]).reset_index(drop=True)
    
        train_y["Prediction"]=np.where(train_y["prob_1"]>Input_probability.iloc[0]['Min_Prob'],1,0)
    
    ##Confusion matrix
        from sklearn.metrics import confusion_matrix
        from pandas.tools.plotting import table
        confusion_matrix=pd.DataFrame(confusion_matrix(train_y[dep_var], train_y["Prediction"]))
        confusion_matrix.columns=["Predicted:False","Predicted:True"]
        confusion_matrix.index=["Actual:False","Actual:True"]
        plt.figure(figsize=(8, 6))
        ax = plt.subplot(frame_on=False) # no visible frame
        ax.xaxis.set_visible(False)  # hide the x axis
        ax.yaxis.set_visible(False)  # hide the y axis
        plt.title('Confusion Matrix Train')
        table(ax, confusion_matrix, loc = 'center')  # where df is your data frame
        pdf.savefig(dpi = 1000,bbox_inches= 'tight')
        plt.close()
    
    
        ##ROC Curve with AUC
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve, auc
        fpr, tpr, threshold = roc_curve(train_y[dep_var], train_y["prob_1"])
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
        plt.title("ROC Curve Train")
        plt.legend(loc = 'lower right')
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        pdf.savefig()
        plt.close()
        
        
        ##Lift Chart
        import scikitplot as skplt
        probs1=model_train.predict_proba(X_train)
        plt.figure(figsize=(8, 6))
        skplt.metrics.plot_lift_curve(train_y[dep_var], probs1[:,])
        plt.title('Lift Chart Train')
        pdf.savefig()
        plt.close()
    
        ##KS Statistic
        KS_Statistic=pd.DataFrame()
        KS_Statistic.loc[0, 0]= round(max(a["Perc_diff"]),2)
        keshav =  "KS Statistic Train = " + str(KS_Statistic.iloc[0, 0])
        
        ##Concordance Discordance 
        churn_0=train_y.loc[train_y[dep_var]==0, [dep_var,'prob_1']].reset_index(drop=True)
        churn_1=train_y.loc[train_y[dep_var]==1, [dep_var,'prob_1']].reset_index(drop=True)
        from bisect import bisect_left, bisect_right
        ones_list = sorted([churn_1.iloc[j,1] for j in churn_1.index])
        ones_length = len(ones_list)
        conc=ties=disc=0
        for i in churn_0.index:
            cur_disc = bisect_left(ones_list, churn_0.iloc[i,1])
            cur_ties = bisect_right(ones_list, churn_0.iloc[i,1]) - cur_disc
            disc += cur_disc
            ties += cur_ties
            conc += ones_length - cur_ties - cur_disc
        
        pairs_tested = ones_length * len(churn_0.index)
        concordance = round(conc/pairs_tested,2)
        discordance = round(disc/pairs_tested,2)
        ties_perc = round(ties/pairs_tested,2)
        Somers_D=round(concordance-discordance,2)
        Gamma=round((concordance-discordance)/(concordance+discordance+ties_perc),2)
        
        
        conc_disc=pd.DataFrame()
        conc_disc.loc[0,"Pairs_tested"]=pairs_tested
        conc_disc.loc[0,"Concordance"]=concordance
        conc_disc.loc[0,"Discordance"]=discordance
        conc_disc.loc[0,"Ties_Perc"]=ties_perc
        conc_disc.loc[0,"Somers'D"]=Somers_D
        conc_disc.loc[0,"Gamma"]=Gamma
        conc_disc.index=["Values:"]
        plt.figure(figsize=(8, 6))
        ax = plt.subplot(frame_on=False) # no visible frame
        ax.xaxis.set_visible(False)  # hide the x axis
        ax.yaxis.set_visible(False)  # hide the y axis
        plt.title(keshav + '\n \n \n Concordance Discordance Train')
        table(ax, conc_disc,loc="center")  # where df is your data frame
        pdf.savefig(dpi = 1000,bbox_inches= 'tight')
        plt.close()
        
    
        display=pd.DataFrame()
        display["train"]=[round((confusion_matrix.iloc[0][0]+confusion_matrix.iloc[1][1])/len(train_y),2),
        round(confusion_matrix.iloc[1][1]/(confusion_matrix.iloc[1][1]+confusion_matrix.iloc[0][1]),2),
        round(confusion_matrix.iloc[1][1]/(confusion_matrix.iloc[1][1]+confusion_matrix.iloc[1][0]),2),
        round(confusion_matrix.iloc[0][0]/(confusion_matrix.iloc[0][0]+confusion_matrix.iloc[0][1]),2),
        round(concordance,2),round(discordance,2),round(ties_perc,2),round(Somers_D,2),round(Gamma,2),KS_Statistic.iloc[0][0]]
        display.index=["Accuracy","Precision","Sensitivity","Specificity","Concordance","Discordance","Ties","Somers'D","Gamma","KS-Statistic"]

    
    
    ######### on test dataset
        probs_test = model_test.predict_proba(X_test)
        
        
        ##Scoring##
        test_y = pd.DataFrame(y_test)
        test_y=test_y.reset_index(drop=True)
        probs_test=pd.DataFrame(model_test.predict_proba(X_test))
        probs_test.columns=["prob_0","prob_1"]
        del probs_test["prob_0"]
        test_y=pd.concat([probs_test.reset_index(drop=True), test_y.reset_index(drop=True)], axis=1)
        test_y['decile'] = pd.qcut(test_y['prob_1'].rank(method='first'), 10,labels=np.arange(10, 0, -1))
        a = pd.DataFrame()
        a["Min_Prob"]=test_y.groupby(["decile"]).prob_1.agg(['min']).reset_index(drop=True)
        a["Max_Prob"] = test_y.groupby(["decile"]).prob_1.agg(['max']).reset_index(drop=True)
        a["Decile"] = [10,9,8,7,6,5,4,3,2,1]
        non_churn_count =test_y[test_y[dep_var] == 0]
        a["Non_churn_count"]=non_churn_count.groupby(["decile"]).size().reset_index(drop=True)
        churn_count =test_y[test_y[dep_var] == 1]
        a["Churn_count"]=churn_count.groupby(["decile"]).size().reset_index(drop=True)
        a=a.sort_values("Decile")
        a["CumSum_Non_Churn"]=a.loc[:,"Non_churn_count"].cumsum()
        a["CumSum_Churn"]=a.loc[:,"Churn_count"].cumsum()
        a["Perc_cumsum_Churn"]=100*a["CumSum_Churn"]/a['Churn_count'].sum()
        a["Perc_cumsum_non_Churn"]=100*a["CumSum_Non_Churn"]/a['Non_churn_count'].sum()
        a["Perc_diff"]=a["Perc_cumsum_Churn"]-a["Perc_cumsum_non_Churn"]
        
        ##To be displayed##
        Input_probability=pd.DataFrame(a.Min_Prob[a.Perc_diff == max(a.Perc_diff)]).reset_index(drop=True)
        
        test_y["Prediction"]=np.where(test_y["prob_1"]>Input_probability.iloc[0]['Min_Prob'],1,0)
        
        ##Confusion matrix
        from sklearn.metrics import confusion_matrix
        from pandas.tools.plotting import table
        confusion_matrix=pd.DataFrame(confusion_matrix(test_y[dep_var], test_y["Prediction"]))
        confusion_matrix.columns=["Predicted:False","Predicted:True"]
        confusion_matrix.index=["Actual:False","Actual:True"]
        plt.figure(figsize=(8, 6))
        ax = plt.subplot(frame_on=False) # no visible frame
        ax.xaxis.set_visible(False)  # hide the x axis
        ax.yaxis.set_visible(False)  # hide the y axis
        plt.title('Confusion Matrix Test')
        table(ax, confusion_matrix, loc = 'center')  # where df is your data frame
        pdf.savefig(dpi = 1000,bbox_inches= 'tight')
        plt.close()
        
        ##ROC Curve with AUC
        fpr, tpr, threshold = roc_curve(test_y[dep_var], test_y["prob_1"])
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
        plt.title("ROC Curve Test")
        plt.legend(loc = 'lower right')
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        pdf.savefig()
        plt.close()
        
        ##Lift Chart
        probs1=model_test.predict_proba(X_test)
        plt.figure(figsize=(8, 6))
        skplt.metrics.plot_lift_curve(test_y[dep_var], probs1[:,])
        plt.title('Lift Chart Test')
        pdf.savefig()
        plt.close()
        
        ##KS Statistic
        KS_Statistic=pd.DataFrame()
        KS_Statistic.loc[0, 0]= round(max(a["Perc_diff"]),2)
        keshav =  "KS Statistic Test = " + str(KS_Statistic.iloc[0, 0])
        
        ##Concordance Discordance 
        churn_0=test_y.loc[test_y[dep_var]==0, [dep_var,'prob_1']].reset_index(drop=True)
        churn_1=test_y.loc[test_y[dep_var]==1, [dep_var,'prob_1']].reset_index(drop=True)
        ones_list = sorted([churn_1.iloc[j,1] for j in churn_1.index])
        ones_length = len(ones_list)
        conc=ties=disc=0
        for i in churn_0.index:
            cur_disc = bisect_left(ones_list, churn_0.iloc[i,1])
            cur_ties = bisect_right(ones_list, churn_0.iloc[i,1]) - cur_disc
            disc += cur_disc
            ties += cur_ties
            conc += ones_length - cur_ties - cur_disc
        
        pairs_tested = ones_length * len(churn_0.index)
        concordance = round(conc/pairs_tested,2)
        discordance = round(disc/pairs_tested,2)
        ties_perc = round(ties/pairs_tested,2)
        Somers_D=round(concordance-discordance,2)
        Gamma=round((concordance-discordance)/(concordance+discordance+ties_perc),2)
        
        
        conc_disc=pd.DataFrame()
        conc_disc.loc[0,"Pairs_tested"]=pairs_tested
        conc_disc.loc[0,"Concordance"]=concordance
        conc_disc.loc[0,"Discordance"]=discordance
        conc_disc.loc[0,"Ties_Perc"]=ties_perc
        conc_disc.loc[0,"Somers'D"]=Somers_D
        conc_disc.loc[0,"Gamma"]=Gamma
        conc_disc.index=["Values:"]
        plt.figure(figsize=(8, 6))
        ax = plt.subplot(frame_on=False) # no visible frame
        ax.xaxis.set_visible(False)  # hide the x axis
        ax.yaxis.set_visible(False)  # hide the y axis
        plt.title(keshav + '\n \n \n Concordance Discordance Test')
        table(ax, conc_disc,loc="center")  # where df is your data frame
        pdf.savefig(dpi = 1000,bbox_inches= 'tight')
        plt.close()
    
    
        display["test"]=[round((confusion_matrix.iloc[0][0]+confusion_matrix.iloc[1][1])/len(test_y),2),
        round(confusion_matrix.iloc[1][1]/(confusion_matrix.iloc[1][1]+confusion_matrix.iloc[0][1]),2),
        round(confusion_matrix.iloc[1][1]/(confusion_matrix.iloc[1][1]+confusion_matrix.iloc[1][0]),2),
        round(confusion_matrix.iloc[0][0]/(confusion_matrix.iloc[0][0]+confusion_matrix.iloc[0][1]),2),
        round(concordance,2),round(discordance,2),round(ties_perc,2),round(Somers_D,2),round(Gamma,2),KS_Statistic.iloc[0][0]]

    
    
        
        ######## VALIDATION
        probs_validate = model_train.predict_proba(X_validate)
        
        
        ##Perfomance Metrics
        validate_y=pd.DataFrame(y_validate)
        validate_y=validate_y.reset_index(drop=True)
        probs_validate=pd.DataFrame(model_train.predict_proba(X_validate))
        probs_validate.columns=["prob_0","prob_1"]
        del probs_validate["prob_0"]
        validate_y=pd.concat([probs_validate.reset_index(drop=True), validate_y.reset_index(drop=True)], axis=1)
        validate_y['decile'] = pd.qcut(validate_y['prob_1'].rank(method='first'), 10,labels=np.arange(10, 0, -1))
        a = pd.DataFrame()
        a["Min_Prob"]=validate_y.groupby(["decile"]).prob_1.agg(['min']).reset_index(drop=True)
        a["Max_Prob"] = validate_y.groupby(["decile"]).prob_1.agg(['max']).reset_index(drop=True)
        a["Decile"] = [10,9,8,7,6,5,4,3,2,1]
        non_churn_count =validate_y[validate_y[dep_var] == 0]
        a["Non_churn_count"]=non_churn_count.groupby(["decile"]).size().reset_index(drop=True)
        churn_count =validate_y[validate_y[dep_var] == 1]
        a["Churn_count"]=churn_count.groupby(["decile"]).size().reset_index(drop=True)
        a=a.sort_values("Decile")
        a["CumSum_Non_Churn"]=a.loc[:,"Non_churn_count"].cumsum()
        a["CumSum_Churn"]=a.loc[:,"Churn_count"].cumsum()
        a["Perc_cumsum_Churn"]=100*a["CumSum_Churn"]/a['Churn_count'].sum()
        a["Perc_cumsum_non_Churn"]=100*a["CumSum_Non_Churn"]/a['Non_churn_count'].sum()
        a["Perc_diff"]=a["Perc_cumsum_Churn"]-a["Perc_cumsum_non_Churn"]
        
        ##To be displayed##
        Input_probability=pd.DataFrame(a.Min_Prob[a.Perc_diff == max(a.Perc_diff)]).reset_index(drop=True)
        
        
        validate_y["Prediction"]=np.where(validate_y["prob_1"]>Input_probability.iloc[0]['Min_Prob'],1,0)
        
        ##Confusion matrix
        from sklearn.metrics import confusion_matrix
        from pandas.tools.plotting import table
        confusion_matrix=pd.DataFrame(confusion_matrix(validate_y[dep_var], validate_y["Prediction"]))
        confusion_matrix.columns=["Predicted:False","Predicted:True"]
        confusion_matrix.index=["Actual:False","Actual:True"]
        plt.figure(figsize=(8, 6))
        ax = plt.subplot(frame_on=False) # no visible frame
        ax.xaxis.set_visible(False)  # hide the x axis
        ax.yaxis.set_visible(False)  # hide the y axis
        plt.title('Confusion Matrix Validate')
        table(ax, confusion_matrix, loc = 'center')  # where df is your data frame
        pdf.savefig(dpi = 1000,bbox_inches= 'tight')
        plt.close()
        
        ##ROC Curve with AUC
        fpr, tpr, threshold = roc_curve(validate_y[dep_var], validate_y["prob_1"])
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
        plt.title("ROC Curve Validate")
        plt.legend(loc = 'lower right')
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        pdf.savefig()
        plt.close()
        
        ##Lift Chart
        probs1=model_train.predict_proba(X_validate)
        plt.figure(figsize=(8, 6))
        skplt.metrics.plot_lift_curve(validate_y[dep_var], probs1[:,])
        plt.title("Lift Chart Validate")
        pdf.savefig()
        plt.close()
        
        ##KS Statistic
        KS_Statistic=pd.DataFrame()
        KS_Statistic.loc[0, 0]= round(max(a["Perc_diff"]),2)
        keshav =  "KS Statistic Validate = " + str(KS_Statistic.iloc[0, 0])
        
        ##Concordance Discordance 
        churn_0=validate_y.loc[validate_y[dep_var]==0, [dep_var,'prob_1']].reset_index(drop=True)
        churn_1=validate_y.loc[validate_y[dep_var]==1, [dep_var,'prob_1']].reset_index(drop=True)
        ones_list = sorted([churn_1.iloc[j,1] for j in churn_1.index])
        ones_length = len(ones_list)
        conc=ties=disc=0
        for i in churn_0.index:
            cur_disc = bisect_left(ones_list, churn_0.iloc[i,1])
            cur_ties = bisect_right(ones_list, churn_0.iloc[i,1]) - cur_disc
            disc += cur_disc
            ties += cur_ties
            conc += ones_length - cur_ties - cur_disc
        
        pairs_tested = round(ones_length * len(churn_0.index),2)
        concordance = round(conc/pairs_tested,2)
        discordance = round(disc/pairs_tested,2)
        ties_perc = round(ties/pairs_tested,2)
        Somers_D=round(concordance-discordance,2)
        Gamma=round((concordance-discordance)/(concordance+discordance+ties_perc),2)
        conc_disc=pd.DataFrame()
        conc_disc.loc[0,"Pairs_tested"]=pairs_tested
        conc_disc.loc[0,"Concordance"]=concordance
        conc_disc.loc[0,"Discordance"]=discordance
        conc_disc.loc[0,"Ties_Perc"]=ties_perc
        conc_disc.loc[0,"Somers'D"]=Somers_D
        conc_disc.loc[0,"Gamma"]=Gamma
        conc_disc.index=["Values:"]
        plt.figure(figsize=(8, 6))
        ax = plt.subplot(frame_on=False) # no visible frame
        ax.xaxis.set_visible(False)  # hide the x axis
        ax.yaxis.set_visible(False)  # hide the y axis
        plt.title(keshav + '\n \n \n Concordance Discordance Validate')
        table(ax, conc_disc,loc="center")  # where df is your data frame
        pdf.savefig(dpi = 1000,bbox_inches= 'tight')
        plt.close()
        
        
        
        display["validate"]=[round((confusion_matrix.iloc[0][0]+confusion_matrix.iloc[1][1])/len(validate_y),2),
        round(confusion_matrix.iloc[1][1]/(confusion_matrix.iloc[1][1]+confusion_matrix.iloc[0][1]),2),
        round(confusion_matrix.iloc[1][1]/(confusion_matrix.iloc[1][1]+confusion_matrix.iloc[1][0]),2),
        round(confusion_matrix.iloc[0][0]/(confusion_matrix.iloc[0][0]+confusion_matrix.iloc[0][1]),2),
        round(concordance,2),round(discordance,2),round(ties_perc,2),round(Somers_D,2),round(Gamma,2),KS_Statistic.iloc[0][0]]
        table_dict = Table_json(display)
        
        result.append({"table" : table_dict})
    
    data=dict()
    data={0:train_y,
          1:test_y,
          2:validate_y
        }
    graph_dict = validation(data,dep_var)
    result.append(graph_dict)
    return result



#################################################################################

def naive_bayes_models(X_train, X_test, y_train, y_test):
    ############################## NAIVE BAYES ###############################
    from sklearn.naive_bayes import GaussianNB
    ######## on training dataset
    gnb = GaussianNB()
    model_train = gnb.fit(X_train, y_train)
    gnb = GaussianNB()
    model_test = gnb.fit(X_test, y_test)
    return model_train, model_test


######################### Random Forest #########################################

def random_forest_models(X_train, X_test, y_train, y_test):
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(n_estimators = 500,n_jobs=2, random_state=123)
    model_train = clf.fit(X_train, y_train)
    clf = RandomForestClassifier(n_estimators = 500,n_jobs=2, random_state=123)
    model_test = clf.fit(X_test, y_test)
    return model_train, model_test


############################ Artificial Neural Network ###################################


def ann_models(X_train, X_test, y_train, y_test,X):
    from keras.models import Sequential
    from keras.layers import Dense
    # fix random seed for reproducibility
    np.random.seed(123)
    # create model
    inputLayer = len(list(X))
    hiddenl1 = int(0.7*inputLayer)
    hiddenl2 = int(0.5*inputLayer)
    hiddenl3 = int(0.3*inputLayer)
    
    model_train = Sequential()
    model_train.add(Dense(inputLayer, input_dim=inputLayer, activation='sigmoid'))
    model_train.add(Dense(hiddenl1 , activation = 'sigmoid'))
    model_train.add(Dense(hiddenl2 ,activation = 'sigmoid'))
    model_train.add(Dense(hiddenl3 , activation = 'sigmoid'))
    model_train.add(Dense(1, activation='sigmoid'))
    # Compile model
    model_train.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # Fit the model
    model_train.fit(X_train,y_train, epochs=150, batch_size=10)
    
    
    
    # fix random seed for reproducibility
    np.random.seed(123)
    # create model
    inputLayer = len(list(X))
    hiddenl1 = int(0.7*inputLayer)
    hiddenl2 = int(0.5*inputLayer)
    hiddenl3 = int(0.3*inputLayer)
    
    model_test = Sequential()
    model_test.add(Dense(inputLayer, input_dim=inputLayer, activation='sigmoid'))
    model_test.add(Dense(hiddenl1 , activation = 'sigmoid'))
    model_test.add(Dense(hiddenl2 ,activation = 'sigmoid'))
    model_test.add(Dense(hiddenl3 , activation = 'sigmoid'))
    model_test.add(Dense(1, activation='sigmoid'))
    # Compile model
    model_test.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # Fit the model
    model_test.fit(X_test,y_test, epochs=150, batch_size=10)
    return model_train, model_test

def report_making_ann(x,model_train, model_test,X_train, X_test, y_train, y_test, X_validate, y_validate,dep_var):
    
    result = []
    ###Scoring###
    probs_train = model_train.predict(X_train)
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.pyplot as plt
    with PdfPages(x) as pdf:
        train_y = pd.DataFrame(y_train)
        train_y=train_y.reset_index(drop=True)
        probs_train=pd.DataFrame(model_train.predict_proba(X_train))
        probs_train.columns=["prob_1"]
        train_y=pd.concat([probs_train.reset_index(drop=True), train_y.reset_index(drop=True)], axis=1)
        train_y['decile'] = pd.qcut(train_y['prob_1'].rank(method='first'), 10,labels=np.arange(10, 0, -1))
        a = pd.DataFrame()
        a["Min_Prob"]=train_y.groupby(["decile"]).prob_1.agg(['min']).reset_index(drop=True)
        a["Max_Prob"] = train_y.groupby(["decile"]).prob_1.agg(['max']).reset_index(drop=True)
        a["Decile"] = [10,9,8,7,6,5,4,3,2,1]
        non_churn_count =train_y[train_y[dep_var] == 0]
        a["Non_churn_count"]=non_churn_count.groupby(["decile"]).size().reset_index(drop=True)
        churn_count =train_y[train_y[dep_var] == 1]
        a["Churn_count"]=churn_count.groupby(["decile"]).size().reset_index(drop=True)
        a=a.sort_values("Decile")
        a["CumSum_Non_Churn"]=a.loc[:,"Non_churn_count"].cumsum()
        a["CumSum_Churn"]=a.loc[:,"Churn_count"].cumsum()
        a["Perc_cumsum_Churn"]=100*a["CumSum_Churn"]/a['Churn_count'].sum()
        a["Perc_cumsum_non_Churn"]=100*a["CumSum_Non_Churn"]/a['Non_churn_count'].sum()
        a["Perc_diff"]=a["Perc_cumsum_Churn"]-a["Perc_cumsum_non_Churn"]
        
        
        ##To be displayed##
        Input_probability=pd.DataFrame(a.Min_Prob[a.Perc_diff == max(a.Perc_diff)]).reset_index(drop=True)
        
        
        
        train_y["Prediction"]=np.where(train_y["prob_1"]>Input_probability.iloc[0]['Min_Prob'],1,0)
        
        ##Confusion matrix
        from sklearn.metrics import confusion_matrix
        from pandas.tools.plotting import table
        confusion_matrix=pd.DataFrame(confusion_matrix(train_y[dep_var], train_y["Prediction"]))
        confusion_matrix.columns=["Predicted:False","Predicted:True"]
        confusion_matrix.index=["Actual:False","Actual:True"]
        plt.figure(figsize=(8, 6))
        ax = plt.subplot(frame_on=False) # no visible frame
        ax.xaxis.set_visible(False)  # hide the x axis
        ax.yaxis.set_visible(False)  # hide the y axis
        plt.title('Confusion Matrix Train')
        table(ax, confusion_matrix, loc = 'center')  # where df is your data frame
        pdf.savefig(dpi = 1000,bbox_inches= 'tight')
        plt.close()
        
        
        ##ROC Curve with AUC
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve, auc
        fpr, tpr, threshold = roc_curve(train_y[dep_var], train_y["prob_1"])
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
        plt.title("ROC Curve Train")
        plt.legend(loc = 'lower right')
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        pdf.savefig()
        plt.close()
        
        
        ##Lift Chart
        import scikitplot as skplt
        probs1=model_train.predict_proba(X_train)
        probs1=pd.DataFrame(probs1)
        probs1.columns=["1"]
        probs1["0"]= 1-probs1["1"]
        probs1 = probs1[['0', '1']]
        probs1=probs1.values
        plt.figure(figsize=(8, 6))
        skplt.metrics.plot_lift_curve(train_y[dep_var], probs1[:,])
        plt.title('Lift Chart Train')
        pdf.savefig()
        plt.close()
        
        ##KS Statistic
        KS_Statistic=pd.DataFrame()
        KS_Statistic.loc[0, 0]= round(max(a["Perc_diff"]),2)
        keshav =  "KS Statistic Train = " + str(KS_Statistic.iloc[0, 0])
        
        ##Concordance Discordance 
        churn_0=train_y.loc[train_y[dep_var]==0, [dep_var,'prob_1']].reset_index(drop=True)
        churn_1=train_y.loc[train_y[dep_var]==1, [dep_var,'prob_1']].reset_index(drop=True)
        from bisect import bisect_left, bisect_right
        ones_list = sorted([churn_1.iloc[j,1] for j in churn_1.index])
        ones_length = len(ones_list)
        conc=ties=disc=0
        for i in churn_0.index:
            cur_disc = bisect_left(ones_list, churn_0.iloc[i,1])
            cur_ties = bisect_right(ones_list, churn_0.iloc[i,1]) - cur_disc
            disc += cur_disc
            ties += cur_ties
            conc += ones_length - cur_ties - cur_disc
        
        pairs_tested = ones_length * len(churn_0.index)
        concordance = round(conc/pairs_tested,2)
        discordance = round(disc/pairs_tested,2)
        ties_perc = round(ties/pairs_tested,2)
        Somers_D=round(concordance-discordance,2)
        Gamma=round((concordance-discordance)/(concordance+discordance+ties_perc),2)
        
        
        conc_disc=pd.DataFrame()
        conc_disc.loc[0,"Pairs_tested"]=pairs_tested
        conc_disc.loc[0,"Concordance"]=concordance
        conc_disc.loc[0,"Discordance"]=discordance
        conc_disc.loc[0,"Ties_Perc"]=ties_perc
        conc_disc.loc[0,"Somers'D"]=Somers_D
        conc_disc.loc[0,"Gamma"]=Gamma
        conc_disc.index=["Values:"]
        plt.figure(figsize=(8, 6))
        ax = plt.subplot(frame_on=False) # no visible frame
        ax.xaxis.set_visible(False)  # hide the x axis
        ax.yaxis.set_visible(False)  # hide the y axis
        plt.title(keshav + '\n \n \n Concordance Discordance Train')
        table(ax, conc_disc,loc="center")  # where df is your data frame
        pdf.savefig(dpi = 1000,bbox_inches= 'tight')
        plt.close()
        
        display=pd.DataFrame()
        display["train"]=[round((confusion_matrix.iloc[0][0]+confusion_matrix.iloc[1][1])/len(train_y),2),
        round(confusion_matrix.iloc[1][1]/(confusion_matrix.iloc[1][1]+confusion_matrix.iloc[0][1]),2),
        round(confusion_matrix.iloc[1][1]/(confusion_matrix.iloc[1][1]+confusion_matrix.iloc[1][0]),2),
        round(confusion_matrix.iloc[0][0]/(confusion_matrix.iloc[0][0]+confusion_matrix.iloc[0][1]),2),
        round(concordance,2),round(discordance,2),round(ties_perc,2),round(Somers_D,2),round(Gamma,2),KS_Statistic.iloc[0][0]]
        display.index=["Accuracy","Precision","Sensitivity","Specificity","Concordance","Discordance","Ties","Somers'D","Gamma","KS-Statistic"]
        
        
        
        
        
        
        #############################################################
        probs_test = model_test.predict_proba(X_test)

        ##Model Scoring
        test_y = pd.DataFrame(y_test)
        test_y=test_y.reset_index(drop=True)
        test_y.columns=[dep_var]
        probs_test=pd.DataFrame(model_test.predict_proba(X_test))
        probs_test.columns=["prob_1"]
        test_y=pd.concat([probs_test.reset_index(drop=True), test_y.reset_index(drop=True)], axis=1)
        test_y['decile'] = pd.qcut(test_y['prob_1'].rank(method='first'), 10,labels=np.arange(10, 0, -1))
        a = pd.DataFrame()
        a["Min_Prob"]=test_y.groupby(["decile"]).prob_1.agg(['min']).reset_index(drop=True)
        a["Max_Prob"] = test_y.groupby(["decile"]).prob_1.agg(['max']).reset_index(drop=True)
        a["Decile"] = [10,9,8,7,6,5,4,3,2,1]
        non_churn_count =test_y[test_y[dep_var] == 0]
        a["Non_churn_count"]=non_churn_count.groupby(["decile"]).size().reset_index(drop=True)
        churn_count =test_y[test_y[dep_var] == 1]
        a["Churn_count"]=churn_count.groupby(["decile"]).size().reset_index(drop=True)
        a=a.sort_values("Decile")
        a["CumSum_Non_Churn"]=a.loc[:,"Non_churn_count"].cumsum()
        a["CumSum_Churn"]=a.loc[:,"Churn_count"].cumsum()
        a["Perc_cumsum_Churn"]=100*a["CumSum_Churn"]/a['Churn_count'].sum()
        a["Perc_cumsum_non_Churn"]=100*a["CumSum_Non_Churn"]/a['Non_churn_count'].sum()
        a["Perc_diff"]=a["Perc_cumsum_Churn"]-a["Perc_cumsum_non_Churn"]
        
        
        ##To be displayed##
        Input_probability=pd.DataFrame(a.Min_Prob[a.Perc_diff == max(a.Perc_diff)]).reset_index(drop=True)
        
        
        test_y["Prediction"]=np.where(test_y["prob_1"]>Input_probability.iloc[0]['Min_Prob'],1,0)
        
        ##Confusion matrix
        from sklearn.metrics import confusion_matrix
        from pandas.tools.plotting import table
        confusion_matrix=pd.DataFrame(confusion_matrix(test_y[dep_var], test_y["Prediction"]))
        confusion_matrix.columns=["Predicted:False","Predicted:True"]
        confusion_matrix.index=["Actual:False","Actual:True"]
        plt.figure(figsize=(8, 6))
        ax = plt.subplot(frame_on=False) # no visible frame
        ax.xaxis.set_visible(False)  # hide the x axis
        ax.yaxis.set_visible(False)  # hide the y axis
        plt.title('Confusion Matrix Test')
        table(ax, confusion_matrix, loc = 'center')  # where df is your data frame
        pdf.savefig(dpi = 1000,bbox_inches= 'tight')
        plt.close()
        
        ##ROC Curve with AUC
        fpr, tpr, threshold = roc_curve(test_y[dep_var], test_y["prob_1"])
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
        plt.title("ROC Curve Test")
        plt.legend(loc = 'lower right')
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        pdf.savefig()
        plt.close()
        
        ##Lift Chart
        probs1=model_test.predict_proba(X_test)
        probs1=pd.DataFrame(probs1)
        probs1.columns=["1"]
        probs1["0"]= 1-probs1["1"]
        probs1 = probs1[['0', '1']]
        probs1=probs1.values
        plt.figure(figsize=(8, 6))
        skplt.metrics.plot_lift_curve(test_y[dep_var], probs1[:,])
        plt.title("Lift Chart Test")
        pdf.savefig()
        plt.close()
        
        ##KS Statistic
        KS_Statistic=pd.DataFrame()
        KS_Statistic.loc[0, 0]= round(max(a["Perc_diff"]),2)
        keshav =  "KS Statistic Test = " + str(KS_Statistic.iloc[0, 0])
        
        ##Concordance Discordance 
        churn_0=test_y.loc[test_y[dep_var]==0, [dep_var,'prob_1']].reset_index(drop=True)
        churn_1=test_y.loc[test_y[dep_var]==1, [dep_var,'prob_1']].reset_index(drop=True)
        ones_list = sorted([churn_1.iloc[j,1] for j in churn_1.index])
        ones_length = len(ones_list)
        conc=ties=disc=0
        for i in churn_0.index:
            cur_disc = bisect_left(ones_list, churn_0.iloc[i,1])
            cur_ties = bisect_right(ones_list, churn_0.iloc[i,1]) - cur_disc
            disc += cur_disc
            ties += cur_ties
            conc += ones_length - cur_ties - cur_disc

        pairs_tested = round(ones_length * len(churn_0.index),2)
        concordance = round(conc/pairs_tested,2)
        discordance = round(disc/pairs_tested,2)
        ties_perc = round(ties/pairs_tested,2)
        Somers_D=round(concordance-discordance,2)
        Gamma=round((concordance-discordance)/(concordance+discordance+ties_perc),2)
        
        conc_disc=pd.DataFrame()
        conc_disc.loc[0,"Pairs_tested"]=pairs_tested
        conc_disc.loc[0,"Concordance"]=concordance
        conc_disc.loc[0,"Discordance"]=discordance
        conc_disc.loc[0,"Ties_Perc"]=ties_perc
        conc_disc.loc[0,"Somers'D"]=Somers_D
        conc_disc.loc[0,"Gamma"]=Gamma
        conc_disc.index=["Values:"]
        plt.figure(figsize=(8, 6))
        ax = plt.subplot(frame_on=False) # no visible frame
        ax.xaxis.set_visible(False)  # hide the x axis
        ax.yaxis.set_visible(False)  # hide the y axis
        plt.title(keshav + '\n \n \n Concordance Discordance Test')
        table(ax, conc_disc,loc="center")  # where df is your data frame
        pdf.savefig(dpi = 1000,bbox_inches= 'tight')
        plt.close()
        
        display["test"]=[round((confusion_matrix.iloc[0][0]+confusion_matrix.iloc[1][1])/len(test_y),2),
        round(confusion_matrix.iloc[1][1]/(confusion_matrix.iloc[1][1]+confusion_matrix.iloc[0][1]),2),
        round(confusion_matrix.iloc[1][1]/(confusion_matrix.iloc[1][1]+confusion_matrix.iloc[1][0]),2),
        round(confusion_matrix.iloc[0][0]/(confusion_matrix.iloc[0][0]+confusion_matrix.iloc[0][1]),2),
        round(concordance,2),round(discordance,2),round(ties_perc,2),round(Somers_D,2),round(Gamma,2),KS_Statistic.iloc[0][0]]
        
        
        
        ######################################################
        probs_validate = model_train.predict_proba(X_validate)
        
        
        
        
        
        validate_y=pd.DataFrame(y_validate)
        validate_y=validate_y.reset_index(drop=True)
        probs_validate=pd.DataFrame(model_train.predict_proba(X_validate))
        probs_validate.columns=["prob_1"]
        validate_y=pd.concat([probs_validate.reset_index(drop=True), validate_y.reset_index(drop=True)], axis=1)
        validate_y['decile'] = pd.qcut(validate_y['prob_1'].rank(method='first'), 10,labels=np.arange(10, 0, -1))
        a = pd.DataFrame()
        a["Min_Prob"]=validate_y.groupby(["decile"]).prob_1.agg(['min']).reset_index(drop=True)
        a["Max_Prob"] = validate_y.groupby(["decile"]).prob_1.agg(['max']).reset_index(drop=True)
        a["Decile"] = [10,9,8,7,6,5,4,3,2,1]
        non_churn_count =validate_y[validate_y[dep_var] == 0]
        a["Non_churn_count"]=non_churn_count.groupby(["decile"]).size().reset_index(drop=True)
        churn_count =validate_y[validate_y[dep_var] == 1]
        a["Churn_count"]=churn_count.groupby(["decile"]).size().reset_index(drop=True)
        a=a.sort_values("Decile")
        a["CumSum_Non_Churn"]=a.loc[:,"Non_churn_count"].cumsum()
        a["CumSum_Churn"]=a.loc[:,"Churn_count"].cumsum()
        a["Perc_cumsum_Churn"]=100*a["CumSum_Churn"]/a['Churn_count'].sum()
        a["Perc_cumsum_non_Churn"]=100*a["CumSum_Non_Churn"]/a['Non_churn_count'].sum()
        a["Perc_diff"]=a["Perc_cumsum_Churn"]-a["Perc_cumsum_non_Churn"]
        
        
        ##To be displayed##
        Input_probability=pd.DataFrame(a.Min_Prob[a.Perc_diff == max(a.Perc_diff)]).reset_index(drop=True)
        
        
        validate_y["Prediction"]=np.where(validate_y["prob_1"]>Input_probability.iloc[0]['Min_Prob'],1,0)
        
        ##Confusion matrix
        from sklearn.metrics import confusion_matrix
        from pandas.tools.plotting import table
        confusion_matrix=pd.DataFrame(confusion_matrix(validate_y[dep_var], validate_y["Prediction"]))
        confusion_matrix.columns=["Predicted:False","Predicted:True"]
        confusion_matrix.index=["Actual:False","Actual:True"]
        plt.figure(figsize=(8, 6))
        ax = plt.subplot(frame_on=False) # no visible frame
        ax.xaxis.set_visible(False)  # hide the x axis
        ax.yaxis.set_visible(False)  # hide the y axis
        plt.title('Confusion Matrix Validate')
        table(ax, confusion_matrix, loc = 'center')  # where df is your data frame
        pdf.savefig(dpi = 1000,bbox_inches= 'tight')
        plt.close()
        
        ##ROC Curve with AUC
        fpr, tpr, threshold = roc_curve(validate_y[dep_var], validate_y["prob_1"])
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
        plt.title("ROC Curve Validate")
        plt.legend(loc = 'lower right')
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        pdf.savefig()
        plt.close()
        
        ##Lift Chart
        probs1=model_train.predict_proba(X_validate)
        probs1=pd.DataFrame(probs1)
        probs1.columns=["1"]
        probs1["0"]= 1-probs1["1"]
        probs1 = probs1[['0', '1']]
        probs1=probs1.values
        plt.figure(figsize=(8, 6))
        skplt.metrics.plot_lift_curve(validate_y[dep_var], probs1[:,])
        plt.title("Lift Chart Validate")
        pdf.savefig()
        plt.close()
        
        ##KS Statistic
        KS_Statistic=pd.DataFrame()
        KS_Statistic.loc[0, 0]= round(max(a["Perc_diff"]),2)
        keshav =  "KS Statistic Validate = " + str(KS_Statistic.iloc[0, 0])
        
        ##Concordance Discordance 
        churn_0=validate_y.loc[validate_y[dep_var]==0, [dep_var,'prob_1']].reset_index(drop=True)
        churn_1=validate_y.loc[validate_y[dep_var]==1, [dep_var,'prob_1']].reset_index(drop=True)
        ones_list = sorted([churn_1.iloc[j,1] for j in churn_1.index])
        ones_length = len(ones_list)
        conc=ties=disc=0
        for i in churn_0.index:
            cur_disc = bisect_left(ones_list, churn_0.iloc[i,1])
            cur_ties = bisect_right(ones_list, churn_0.iloc[i,1]) - cur_disc
            disc += cur_disc
            ties += cur_ties
            conc += ones_length - cur_ties - cur_disc
        
        pairs_tested = round(ones_length * len(churn_0.index),2)
        concordance = round(conc/pairs_tested,2)
        discordance = round(disc/pairs_tested,2)
        ties_perc = round(ties/pairs_tested,2)
        Somers_D=round(concordance-discordance,2)
        Gamma=round((concordance-discordance)/(concordance+discordance+ties_perc),2)
        conc_disc=pd.DataFrame()
        conc_disc.loc[0,"Pairs_tested"]=pairs_tested
        conc_disc.loc[0,"Concordance"]=concordance
        conc_disc.loc[0,"Discordance"]=discordance
        conc_disc.loc[0,"Ties_Perc"]=ties_perc
        conc_disc.loc[0,"Somers'D"]=Somers_D
        conc_disc.loc[0,"Gamma"]=Gamma
        conc_disc.index=["Values:"]
        plt.figure(figsize=(8, 6))
        ax = plt.subplot(frame_on=False) # no visible frame
        ax.xaxis.set_visible(False)  # hide the x axis
        ax.yaxis.set_visible(False)  # hide the y axis
        plt.title(keshav + '\n \n \n Concordance Discordance Validate')
        table(ax, conc_disc,loc="center")  # where df is your data frame
        pdf.savefig(dpi = 1000,bbox_inches= 'tight')
        plt.close()

        display["validate"]=[round((confusion_matrix.iloc[0][0]+confusion_matrix.iloc[1][1])/len(validate_y),2),
        round(confusion_matrix.iloc[1][1]/(confusion_matrix.iloc[1][1]+confusion_matrix.iloc[0][1]),2),
        round(confusion_matrix.iloc[1][1]/(confusion_matrix.iloc[1][1]+confusion_matrix.iloc[1][0]),2),
        round(confusion_matrix.iloc[0][0]/(confusion_matrix.iloc[0][0]+confusion_matrix.iloc[0][1]),2),
        round(concordance,2),round(discordance,2),round(ties_perc,2),round(Somers_D,2),round(Gamma,2),KS_Statistic.iloc[0][0]]
        table_dict = Table_json(display)
        result.append({"table" : table_dict})
    
    data=dict()
    
    data={"train":train_y,
          "test":test_y,
          "validate":validate_y
        }
    graphs_dict = validation(data,dep_var)
    json.loads(graphs_dict)
    result.append(graphs_dict)
    return result
    import json
########################## Graphs in JSON ###############################


def validation(data,dep_var):
    fpr = dict()
    tpr = dict()
    percentages = dict()
    gains = dict()
    roc_auc = []
    
    import sklearn.metrics as m
    from scikitplot.helpers import cumulative_gain_curve
        
    #Calculating the accuracy and precision using the sklear.metrics of Python
    for key in data:
        fpr[key], tpr[key], _ = m.roc_curve(data[key][dep_var],data[key]['prob_1'])
        roc_auc.append(round((m.auc(fpr[key], tpr[key])),4))
        percentages[key], gains[key] = cumulative_gain_curve(data[key][dep_var].values,data[key]['prob_1'])
        
        
    J_ROC=jsonROC('Reciever Operating Curve','False Positive Rate','True Positive Rate',MultiRoc(tpr,fpr,roc_auc))
    J_Gains=jsonROC('Cumulative Gains Chart','Percentage of Sample','Gain',MultiGain(gains,percentages))    

    chart_types=["thermometer","mscombi2d","scatter"]
    
    result=ResultJson(J_ROC,J_Gains,chart_types)

    return result

#Function to calculate the average health of the model as per the standard logic which can be configured as well
def ReturnROC(tpr,fpr): 
    ls=[]
    for i in range(len(tpr)):
        data=dict()
        data["x"] = fpr[i]
        data["y"] = tpr[i]
        ls.append(data)
    return ls


def jsonROC(title,xaxis,yaxis,charts):
    dataSource = {
      "chart": {
        "theme" : "fint",
        "caption": title,
        "baseFont": "Helvetica Neue,Arial",
        "captionFontSize": "14",
        "subcaptionFontSize": "14",
        "subcaptionFontBold": "0",
        "xAxisName": xaxis,
        "yAxisName": yaxis,
        "xAxisMinValue": "0",
        "xAxisMaxValue": "1",
        "yAxisMinValue": "0",
        "yAxisMaxValue": "1",
        "showBorder": "1",
        "showCanvasBorder": "1",
        "showAxisLines": "1",
        "use3dlighting": "1",
        "drawAnchors": "0",
        "showYAxisLine": "1",
        "yAxisLineThickness": "1",
        "showLegend" : "1",
        "legendItemFontBold" : "1",
        "legendCaptionFont" : "Verdana",
        "numDivLines" : 10,
        "numVDivLines" : 10
      },
     "trendlines": [{
        "line": [{
          	"startValue": 0,
            "endValue" : 1,
            "dashed" : "1",
            "thickness" : 2,
          }]
          }],
      "dataset": charts
      }
    
    return dataSource

def jsonChart(SeriesName,data):
    ChartData = {
          "seriesname": SeriesName,
          "drawLine": "1",
          "data": data
        }
    return ChartData
    
def MultiRoc(tpr,fpr,Auc):
    ls=[]
    for i,key in enumerate(tpr):        
        SeriesName=str(key)+" AUC "+str(Auc[i])
        ls.append(jsonChart(SeriesName,ReturnROC(tpr[key].tolist(),fpr[key].tolist())))
    return ls
    

def MultiGain(tpr,fpr):
    ls=[]
    for key in tpr:        
        SeriesName=key
        ls.append(jsonChart(SeriesName,ReturnROC(tpr[key].tolist(),fpr[key].tolist())))
    return ls
    


def jsonCreate(Metric,data,label,model_base):
    dataSource = {
            "chart": {
                "theme" : "fint",
                "caption": Metric,
                "xAxisname": "Model",
                "showplotborder": "1",
                "showXAxisLine": "1",
                "showLegend" : "0",
                "xAxisLineThickness": "1",
                "xAxisLineColor": "#999999",
                "showBorder": "1",
          		 "showCanvasBorder": "1",
                   "showAxisLines": "1",
                   "use3dlighting": "1",
                   "showYAxisLine": "1",
                   "yAxisLineThickness": "1",
                   "numDivLines" : 9
            },
            "categories": [{
                "category": label
            }
                          ],
            "trendlines": [{
        "line": [{
          	"startValue": model_base,
            "dashed" : "1",
            "thickness" : 2,
          }]
          }],
            "dataset": [
                {
                    "seriesName": Metric,
                    "renderAs": "column",
                    "data": data
                }
                
            ]
    }
    
    return dataSource




def ResultJson(J_ROC,J_Gains,chart_types):
    import json
    result={
            "charts": 
                    [
               {
            "chartname": "ROC Chart",
            "displayName": "ROC",
            "chartType":chart_types[2],
            "chartObject": J_ROC
                 },
               {
            "chartname": "Gains Chart",
            "displayName": "Gains Chart",
            "chartType":chart_types[2],
            "chartObject": J_Gains
                 },
     
                    ]
                
            }
    result=json.dumps(result)
    return result

############################## Table in JSON #######################
def Table_json(table_df):
    tabledata=[]
    for i in table_df.index:
        a={}
        for j in table_df.columns:
            a[j] = table_df.loc[i,j]
        tabledata.append({i:a})
    return(tabledata)
