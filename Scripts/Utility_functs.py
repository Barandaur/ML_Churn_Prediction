# -*- coding: utf-8 -*-
"""
Created on Sat May  8 23:44:03 2021

@author: Eugen
"""
import pandas as pd
import numpy as np


from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix


import seaborn as sns
from matplotlib import pyplot as plt
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

def imputer(df):
    return 0 if  df.time==0  else  df.chargesTotal

def reset_dataset():
    train = pd.read_csv("train.csv")
    try:
        train.drop(columns=["w"], inplace = True)
    except:
        pass
    test = pd.read_csv("test.csv")

    print(f"train shape = {train.shape}, test shape = {test.shape}")
    
    dataset = pd.concat([train,test])
    dataset["chargesTotal"] = dataset.apply(imputer, axis=1)

    
    return train, test, dataset


def split_dset(dset, split_val):
    train = dset.iloc[:split_val]
    test  = dset.iloc[split_val:].drop("y", axis=1)
    
    return train, test

# -------------------------------------------------------------------------------

def plot_df_cols(df, cols, larg=20, alt=20):
    
    fig = plt.figure(figsize=(larg, alt)) # create a figure object
    # aggounge spazi bianchi tra un plot e quelli sopra\sotto e a dx\sx
    fig.subplots_adjust(hspace=0.4, wspace=0.3)

    nrows = np.ceil(len(cols)/4)
    ncols = 4

    for i, col in enumerate(cols):
        ax = fig.add_subplot(nrows,ncols,i+1) 
        ax.hist(df[f"{col}"]) 
        ax.set_title(f"{col}")

    plt.show()
    return

# -------------------------------------------------------------------------------

def tt_split(df, test_size=0.35, r_state= 42):
    """ se usi cross-validation imho non ti serve questo split,
        perché il tuo 'validation' sarà dato dai tuoi k folds """
     
    X = df.drop('y', axis=1).to_numpy()
    y = df[["y"]].to_numpy().ravel()


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state= r_state)
    
    return X_train, X_test, y_train, y_test

def create_X_y(train_df, random_state = 42):
    # shuffle
    df = train_df.sample(frac=1, random_state = random_state)
    
    X = df.drop("y", axis=1).to_numpy()
    y = df[["y"]].to_numpy().ravel()
    return X, y

# -------------------------------------------------------------------------------

def custom_scorer(y_true, y_predict):
    """ Crea un custom scorer da passare al cv_scores di SkLearn """
    
    # sottrai, così da avere zero se la predizione è corretta, 1 se FN e -1 se FP
    score_arr = y_predict - y_true
    score_arr[score_arr==-1] = 5
    #print("errori da 5 su errori totali: ", round(len(score_arr[score_arr==5])/len(score_arr) ,2) )
    
    return np.sum(score_arr)

loss_function = make_scorer(custom_scorer, 
                              greater_is_better=False)


def compute_cv(model, X_train, y_train, k, scorer, silence=False):
    
    if scorer:
        cv_scores = cross_val_score(model, 
                                    X_train,
                                    y_train,
                                    cv=k, 
                                    scoring=scorer)  #mean_abs_scorer, 'neg_mean_absolute_error', sono effettivamente equivalenti
    else:
        cv_scores = cross_val_score(model, 
                                    X_train, 
                                    y_train,
                                    cv=k) 
    if not silence:    
    	print("mean = ", cv_scores.mean(), "std = ", cv_scores.std())
    
    return cv_scores


def custom_k_folds(model, train_df, k=10, thr=0.5):
    """ requires model.predict_proba, assumes l=2 with y={1,2} """
    
    X, y = create_X_y(train_df)
    kf = KFold(n_splits=k)
    
    cv_scores = []
    y_tr_y_pr = []
    y_probs   = []
    for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            model.fit(X_train, y_train)
            proba = model.predict_proba(X_test)
            y_pred = np.array([1 if i[0]> thr else 2 for i in proba ])

            score = custom_scorer(y_test, y_pred)
            cv_scores.append(score)
            y_tr_y_pr.append([y_test, y_pred])
            y_probs.append(proba)
    
    cv_scores = np.array(cv_scores)
    print("mean = ", cv_scores.mean(), "std =", cv_scores.std())
    return cv_scores, y_tr_y_pr, y_probs

# -------------------------------------------------------------------------------
def find_ROC_Score(df, model):
    X_train, X_test, y_train, y_test = tt_split(df, test_size=0.2)
    
    # try-except in case the model doesn't have .decision_function()
    try:
        y_score = model.fit(X_train, y_train).decision_function(X_test)
        fpr, tpr, _ = roc_curve(y_test, y_score, pos_label=2)
    except:
        y_score = model.fit(X_train, y_train).predict_proba(X_test) 
        fpr, tpr, _ = roc_curve(y_test, y_score[:,1] , pos_label=2)

    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc

def plot_ROC(fpr, tpr, roc_auc):
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()


def Feature_Importance_plot(feature_dataframe, column="Random_Forest" ):
    trace = go.Scatter(
        y = feature_dataframe[column].values,
        x = feature_dataframe['features'].values,
        mode='markers',
        marker=dict(
            sizemode = 'diameter',
            sizeref = 1,
            size = 25,
    #       size= feature_dataframe['AdaBoost feature importances'].values,
            #color = np.random.randn(500), #set color equal to a variable
            color = feature_dataframe[column].values,
            colorscale='Portland',
            showscale=True
        ),
        text = feature_dataframe['features'].values
    )
    data = [trace]

    layout= go.Layout(
        autosize= True,
        title= column + " feature importance",
        hovermode= 'closest',
    #     xaxis= dict(
    #         title= 'Pop',
    #         ticklen= 5,
    #         zeroline= False,
    #         gridwidth= 2,
    #     ),
        yaxis=dict(
            title= 'Feature Importance',
            ticklen= 5,
            gridwidth= 2
        ),
        showlegend= False
    )
    fig = go.Figure(data=data, layout=layout)
    py.iplot(fig,filename='scatter2010')
    return


def mean_conf_matrix(y_tr_y_pr):
    conf_matrices = np.zeros((2,2))
    for i in range(len(y_tr_y_pr)):
        y_true = y_tr_y_pr[i][0]
        y_pred = y_tr_y_pr[i][1]
        conf_matrix = confusion_matrix(y_true, y_pred)
        conf_matrices += conf_matrix     

    return conf_matrices/len(y_tr_y_pr)

# -------------------------------------------------------------------------------
def get_txt(final_pred, filename = "Predictions.txt"):
    # non hanno senso predizioni negative
    #final_pred[final_pred<0] = 0

    
    np.savetxt(filename, final_pred, fmt="%s")
    return