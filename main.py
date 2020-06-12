
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import LocalOutlierFactor
from sklearn import metrics
from sklearn.model_selection import GridSearchCV,cross_val_score
from sklearn.linear_model import LogisticRegression as log_rec
from sklearn.neighbors import KNeighborsClassifier
warnings.simplefilter("ignore")
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from keras.models import Sequential
from keras.layers import Dense,Dropout
# PLEASE CLOSE FIGURE TO SEE OTHERS.
########################################################
def logistic_regression(x_train,x_test,y_train,y_test):
    logistic_model=log_rec(C=0.0001,solver='newton-cg')
    logistic_model.fit(x_train,y_train)
    logistic_pred=logistic_model.predict(x_test)
    print("Logistic Regression Accuracy Score Before Tuning %.5f"% metrics.accuracy_score(logistic_pred,y_test))
    print("Logistic Regression F1 Score Before Tuning %.5f"%metrics.f1_score(logistic_pred,y_test))
    ### Logistic Regression Parameter Tuning ###
    logistic_params={'C':np.logspace(-4, 4, 20),'solver':['liblinear'],'penalty':['l1','l2']}
    grid=GridSearchCV(log_rec(),logistic_params,scoring='accuracy',cv=3)
    grid.fit(x_train,y_train)
    ### Training Model with best parameters ###

    logistic_model=log_rec(C=grid.best_params_['C'],solver=grid.best_params_['solver'])
    logistic_model.fit(x_train,y_train)
    logistic_pred=logistic_model.predict(x_test)
    print("Logistic Regression Accuracy Score After Tuning %.5f"%metrics.accuracy_score(logistic_pred,y_test))
    print("Logistic Regression F1 Score After Tuning %.5f"%metrics.f1_score(logistic_pred,y_test))
    return logistic_pred


########################################################
def knn_algorithm(x_train,x_test,y_train,y_test):
    knn_model=KNeighborsClassifier(n_neighbors=3)
    knn_model.fit(x_train,y_train)
    knn_pred=knn_model.predict(x_test)
    print("KNN Accuracy Score Before Tuning %.5f"% metrics.accuracy_score(knn_pred,y_test))
    print("KNN F1 Score Before Tuning %.5f"% metrics.f1_score(knn_pred,y_test))
    knn_params={'n_neighbors':np.arange(3,90,2)}
    grid=GridSearchCV(KNeighborsClassifier(),knn_params,scoring='accuracy',cv=3)
    grid.fit(x_train,y_train)
    knn_model=KNeighborsClassifier(n_neighbors=grid.best_params_['n_neighbors'])
    knn_model.fit(x_train,y_train)
    knn_pred=knn_model.predict(x_test)
    print("KNN Accuracy Score After Tuning %.5f"% metrics.accuracy_score(knn_pred,y_test))
    print("KNN F1 Score After Tuning %.5f" % metrics.f1_score(knn_pred,y_test))
    #print(knn_model.predict_proba(x_test[3:5]))
    return knn_pred
########################################################
def adaboost_algorithm(x_train,x_test,y_train,y_test):
    model = AdaBoostClassifier(base_estimator = DecisionTreeClassifier(max_depth=1), n_estimators=3)
    model.fit(x_train, y_train)
    adaboost_pred=model.predict(x_test)
    print("Adaboost Accuracy Score Before Tuning %.5f"%metrics.accuracy_score(adaboost_pred,y_test))
    print("Adaboost F1 Score Before Tuning %.5f"%metrics.f1_score(adaboost_pred,y_test))
    ada_params={'base_estimator':[DecisionTreeClassifier(),RandomForestClassifier()],'n_estimators':[1,2,3,5,10,15,20]}
    grid=GridSearchCV(AdaBoostClassifier(),ada_params,scoring='accuracy',cv=3)
    grid.fit(x_train,y_train)
    ada_model=AdaBoostClassifier(base_estimator=grid.best_params_['base_estimator'],n_estimators=grid.best_params_['n_estimators'])
    ada_model.fit(x_train,y_train)
    adaboost_pred=ada_model.predict(x_test)
    print("Adaboost Accuracy Score After Tuning %.5f"%metrics.accuracy_score(adaboost_pred,y_test))
    print("Adaboost F1 Score After Tuning %.5f"%metrics.f1_score(adaboost_pred,y_test))
    return adaboost_pred

########################################################
def xgboost_algorithm(x_train,x_test,y_train,y_test):
    xg_model=XGBClassifier()
    xg_model.fit(x_train,y_train)
    xg_pred=xg_model.predict(x_test)
    print("XGBoost Accuracy Score Before Tuning %.5f"% metrics.accuracy_score(xg_pred,y_test))
    print("XGBoost F1 Score Before Tuning %.5f"%metrics.f1_score(xg_pred,y_test))
    xg_model= XGBClassifier(silent=False, 
                      scale_pos_weight=1,
                      learning_rate=0.01,  
                      colsample_bytree = 0.4,
                      subsample = 0.8,
                      objective='binary:logistic', 
                      n_estimators=1000, 
                      reg_alpha = 0.3,
                      max_depth=4, 
                      gamma=10)
    xg_model.fit(x_train,y_train)
    xg_pred=xg_model.predict(x_test)
    print("XGBoost Accuracy Score After Tuning %.5f"% metrics.accuracy_score(xg_pred,y_test))
    print("XGBoost F1 Score After Tuning %.5f"%metrics.f1_score(xg_pred,y_test))
    return xg_pred
########################################################
def decision_tree_algorithm(x_train,x_test,y_train,y_test):
    dt_model=DecisionTreeClassifier()
    dt_model.fit(x_train,y_train)
    dt_pred=dt_model.predict(x_test)
    print("Decision Tree Accuracy Score Before Tuning %.5f"% metrics.accuracy_score(dt_pred,y_test))
    print("Decision Tree F1 Score Before Tuning %.5f"% metrics.f1_score(dt_pred,y_test))
    dt_params={'criterion':['gini','entropy'],'max_depth':(2,4,6,8,10,12,16,18,20)}
    grid=GridSearchCV(DecisionTreeClassifier(),dt_params,scoring='accuracy')
    grid.fit(x_train,y_train)
    dt_model=DecisionTreeClassifier(criterion=grid.best_params_['criterion'],max_depth=4)
    dt_model.fit(x_train,y_train)
    dt_pred=dt_model.predict(x_test)
    print("Decision Tree Accuracy Score After Tuning %.5f"% metrics.accuracy_score(dt_pred,y_test))
    print("Decision Tree F1 Score After Tuning %.5f"% metrics.f1_score(dt_pred,y_test))

   
   #Visualization of tree. In folder there is tree's visual figure.
   
   # from sklearn import tree
   # plt.figure(figsize=(60,40),dpi=400)
    #tree.plot_tree(dt_model,filled=True,rounded=True,qclass_names=['Diabetes','No Diabetes'])
    # plt.show()
    #plt.savefig("tree_visual.png")
    return dt_pred
########################################################
def mlp_backprop(x_train,x_test,y_train,y_test):
    keras_model=Sequential()
    keras_model.add(Dense(units=6,init='uniform',activation='relu',input_dim=x_train.shape[1]))
    keras_model.add(Dense(units=3,init='uniform',activation='relu'))
    keras_model.add(Dense(units=3,init='uniform',activation='relu'))
    keras_model.add(Dense(1,activation='sigmoid'))
    keras_model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    keras_model.fit(x_train, y_train, epochs=150, batch_size=10,verbose=0)
    keras_pred=keras_model.predict(x_test)
    
    
    
    keras_pred[np.where(keras_pred>0.5)]=1
    keras_pred[np.where(keras_pred<0.5)]=0
    print("MLayer Perceptron Accuracy Score  %.5f"% metrics.accuracy_score(keras_pred,y_test))
    print("MLayer Perceptron F1 Score  %.5f"%metrics.f1_score(keras_pred,y_test))
    
    return keras_pred



########################################################

def svm_algorithm(x_train,x_test,y_train,y_test):
    svc_model=SVC(kernel='linear')
    svc_model.fit(x_train,y_train)
    svc_pred=svc_model.predict(x_test)
    print("SVM Accuracy Score Before Tuning %.5f"% metrics.accuracy_score(svc_pred,y_test))
    print("SVM F1 Score Before Tuning %.5f"%metrics.f1_score(svc_pred,y_test))
    svc_params=({'kernel':['rbf'],'C':[0.001,0.1,1,10,100],'gamma':['auto','scale']})
    grid=GridSearchCV(SVC(),param_grid=svc_params,scoring="accuracy",cv=3)
    grid.fit(x_train,y_train)
    svc_model=SVC(C=grid.best_params_['C'],kernel=grid.best_params_['kernel'],gamma=grid.best_params_['gamma'])
    svc_model.fit(x_train,y_train)
    svc_pred=svc_model.predict(x_test)
    print("SVM Accuracy Score After Tuning %.5f"% metrics.accuracy_score(svc_pred,y_test))
    print("SVM  F1 Score After Tuning %.5f"%metrics.f1_score(svc_pred,y_test))
    return svc_pred
#######################################################
def naive_bayes_algorithm(x_train,x_test,y_train,y_test):
    from sklearn.naive_bayes import GaussianNB
    nb_model=GaussianNB()
    nb_model.fit(x_train,y_train)
    nb_pred=nb_model.predict(x_test)
    print("NaiveBayes Accuracy Score  %.5f"% metrics.accuracy_score(nb_pred,y_test))
    print("NaiveBayes F1 Score  %.5f"%metrics.f1_score(nb_pred,y_test))
    return nb_pred
########################################################
def kmeans_algorithm(x_train,x_test,y_test):
    
    km_model=KMeans(n_clusters=2,init='random')
    km_model.fit(x_train)
    km_pred=km_model.predict(x_test)
    if metrics.accuracy_score(km_pred,y_test)<0.5:
        zeros=np.where(km_pred==0)
        ones=np.where(km_pred==1)
        km_pred[zeros]=1
        km_pred[ones]=0
    print("Kmeans Accuracy Score  %.5f"% metrics.accuracy_score(km_pred,y_test))
    print("Kmeans F1 Score  %.5f"% metrics.f1_score(km_pred,y_test))
    

########################################################
def random_forest_algorithm(x_train,x_test,y_train,y_test):
    rf_model=RandomForestClassifier()
    rf_model.fit(x_train,y_train)
    rf_pred=rf_model.predict(x_test)
    print("RandomForest Accuracy Score Before Tuning %.5f"% metrics.accuracy_score(rf_pred,y_test))
    print("RandomForest F1 Score Before Tuning %.5f"%metrics.f1_score(rf_pred,y_test))
    rf_params={'n_estimators':range(10,110,10),'criterion':['gini','entropy']}
    grid=GridSearchCV(RandomForestClassifier(),rf_params,cv=3,scoring='accuracy')
    grid.fit(x_train,y_train)
    rf_model=RandomForestClassifier(n_estimators=grid.best_params_['n_estimators'],criterion=grid.best_params_['criterion'],max_depth=4)
    rf_model.fit(x_train,y_train)
    rf_pred=rf_model.predict(x_test)
    print("RandomForest Accuracy Score After Tuning %.5f"% metrics.accuracy_score(rf_pred,y_test))
    print("RandomForest F1 Score After Tuning %.5f"%metrics.f1_score(rf_pred,y_test))
    return rf_pred
########################################################
def lof(data):
    clf=LocalOutlierFactor(n_neighbors=20,contamination=0.1)
    outlier_pred=clf.fit_predict(data)
    x_score=clf.negative_outlier_factor_
    x_score=np.abs(x_score)
    xscr_mean=x_score.mean()
    xscr_std=np.std(x_score)
    lower=xscr_mean-(xscr_std)
    upper=xscr_mean+(xscr_std)
    inliers=data[~((x_score>upper)| (x_score<lower))]
    print("Local OutlierFactor",len(data)-len(inliers),"Row Affected")
    return inliers
    
########################################################
def spliting_data(data):
    columns=set(data.columns)
    columns.remove('Outcome')
    x_reduced=data[columns]
    y=data['Outcome']
    from sklearn.model_selection import train_test_split
    x_train,x_test,y_train,y_test=train_test_split(x_reduced,y,test_size=0.33)
    return x_train,x_test,y_train,y_test

########################################################
def iqr4outliers(data):
    before_x=[]
    after_x=[]
    data4iqr=data.copy()
    for i in range(len(data4iqr.columns)-1):
        col=data4iqr.iloc[:,i:i+1]
        
        Q1=col.quantile(0.25)
        Q3=col.quantile(0.75)
        IQR=Q3-Q1
        lower=Q1-1.5*IQR
        upper=Q3+1.5*IQR
        new_col=col[~((col<lower)|(col>upper)).any(axis=1)]
        ex_col=col[((col<lower)|(col>upper)).any(axis=1)]
        before_x.append(col)
        data4iqr.drop(index=ex_col.index,axis=0,inplace=True)
        after_x.append(data4iqr.iloc[:,i:i+1])
    data4iqr.reset_index(inplace=True)
    print("IQR METHOD",len(data)-len(data4iqr)," Row Effected")
    ####IQR Visualization####
    f, axes = plt.subplots(2,5, figsize=(22, 7))
    j=0
    for i in range(6):
        if data4iqr.columns[i+1]=="Outcome":
            continue
        sns.boxplot(before_x[i],ax=axes[0,j]).set_title(data4iqr.columns[i+1]+" Before IQR")
        sns.boxplot(after_x[i],ax=axes[1,j]).set_title(data4iqr.columns[i+1]+" After IQR")
        j+=1
    plt.show()
    return data4iqr




########################################################
def pcanalysis(x,y):
    pca=PCA().fit(x)
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.title("How many variable represents our model with acceptable error PCA")
    plt.xlabel("Number of variable")
    plt.ylabel("Variance")
    plt.grid()
    plt.show()
    for i in range(len(x)):
       
        if np.cumsum(pca.explained_variance_ratio_)[i]>=0.8: #This cumulative sum operations give us how many variable represents our data how much.
            print(f" PCA ---> We can represent all data %20 acceptable error with {i} features")
            break
    
    #creating pca again with best params.
    pca=PCA(n_components=5)
    x_reduced=pca.fit_transform(data_scaled.iloc[:,:-1])
    data_reduced=pd.concat([pd.DataFrame(x_reduced),y],axis=1)
    return data_reduced
########################################################
def age_visualization(data):
    age_data=pd.DataFrame(data.groupby(['Age'],as_index=False)['Outcome'].count())
    interval={}
    temp_sum=0
    for i in range(len(age_data)):
        temp_sum+=int(age_data.iloc[i,1])
        if age_data.iloc[i,0]==35:
            interval.update({"20-35":temp_sum})
            temp_sum=0
        elif age_data.iloc[i,0]==50:
            interval.update({"35-50":temp_sum})
            temp_sum=0
        elif age_data.iloc[i,0]==81:
            interval.update({"+50":temp_sum})
    plt.bar(interval.keys(),interval.values(),color=['#cc6699','#339933','#006666'])
    plt.xlabel("Age")
    plt.ylabel("Count")
    plt.title("Counts of Age Interval")
    plt.show()
        
        
########################################################
def scaling_data(dataframe):
    scaler=StandardScaler()
    x_scaled=scaler.fit_transform(X)
    data_scaled=pd.concat([pd.DataFrame(x_scaled,columns=dataframe.iloc[:,:-1].columns),y],axis=1)
    return x_scaled,data_scaled
########################################################
def visualize(data):
    bmi=data[(data['BMI'].notnull()) & (data['Outcome']==1)]
    plt.scatter(bmi['BMI'],bmi['Age'])
    bmineg=data[(data['BMI'].notnull()) & (data['Outcome']==0)]
    plt.scatter(bmineg['BMI'],bmineg['Age'])
    plt.title("BMI Visualization (How Distrubuted Data)")
    plt.legend(["Positive","Negative"])
    plt.show()

########################################################
def homemade_ensembling(xg_pred,keras_pred,svc_pred,rf_pred,nb_pred,logistic_pred):
    
    xg=pd.DataFrame(xg_pred.reshape(-1,1))
    keras=pd.DataFrame(keras_pred.reshape(-1,1))
    svm=pd.DataFrame(svc_pred.reshape(-1,1))
    rf=pd.DataFrame(rf_pred.reshape(-1,1))
    nb=pd.DataFrame(nb_pred.reshape(-1,1))
    logistic=pd.DataFrame(logistic_pred.reshape(-1,1))
    out_df=pd.concat([rf,xg,keras,keras,svm,svm,nb,logistic],axis=1)
    out_df.columns=['RandomForest','XGBoost','Keras','Keras','SVM','SVM','NaiveBayes','LogisticRegression']
    out_df=out_df.astype('int')
    return out_df



########################################################

total_pred=[]
def freq_ones(row):
    zero_temp=0
    one_temp=0
    for i in range(7):
        if row.iloc[i]==0:
            zero_temp+=1
        else:
            one_temp+=1
    if zero_temp>one_temp:
        total_pred.append(0)
    else:
        total_pred.append(1)
########################################################
if __name__=='__main__':
    data=pd.read_csv('diabetes.csv')
   # print("Correlation RAW DATA",data.corr())
    X=set(data.columns)
    X.remove("BloodPressure")
    X.remove("SkinThickness")
    data=data[X] #remove bloodpressure and skinthickness on dataframe.
    X.remove('Outcome') # This is for train and test data.
    X=data[X]
    y=data['Outcome']
    visualize(data)
    x_scaled,data_scaled=scaling_data(data) # labels doesnt need scaling or encoding.Already has 2 classes:0 and 1
    pcanalysis(x_scaled,y)
    age_visualization(data)
    data_iqr=iqr4outliers(data)
    data_lof=lof(data)
    x_train,x_test,y_train,y_test=spliting_data(data_lof)
    logistic_pred=logistic_regression(x_train,x_test,y_train,y_test)
    knn_pred=knn_algorithm(x_train,x_test,y_train,y_test)
    dt_pred=decision_tree_algorithm(x_train,x_test,y_train,y_test)
    svc_pred=svm_algorithm(x_train,x_test,y_train,y_test)
    nb_pred=naive_bayes_algorithm(x_train,x_test,y_train,y_test)
    rf_pred=random_forest_algorithm(x_train,x_test,y_train,y_test)
    ada_pred=adaboost_algorithm(x_train,x_test,y_train,y_test)
    xg_pred=xgboost_algorithm(x_train,x_test,y_train,y_test)
    kmeans_algorithm(x_train,x_test,y_test)
    keras_pred=mlp_backprop(x_train,x_test,y_train,y_test)
    out_df=homemade_ensembling(xg_pred,keras_pred,svc_pred,rf_pred,nb_pred,logistic_pred)
    for i in range(len(out_df)):
        freq_ones(out_df.iloc[i,:])
    print("Home-Made Ensembling Accuracy Score",metrics.accuracy_score(total_pred,y_test))
    print("Home-Made Ensembling F1 Score",metrics.f1_score(total_pred,y_test))
   






  # print("Correlation IQR Inliners",data_iqr.corr())
    




