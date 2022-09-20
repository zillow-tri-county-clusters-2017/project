import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, explained_variance_score
from sklearn.metrics import r2_score
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression, TweedieRegressor, LassoLars
from sklearn.preprocessing import PolynomialFeatures

import matplotlib.pyplot as plt
import seaborn as sns



def x_train_feats(df, y):
    
    X_train = df
    
    lm = LinearRegression()
    
    model = lm.fit(X_train, y)
    
    return X_train, model


def plot_residuals(y, yhat):
    
    residuals = y - yhat
    
    plt.scatter(y, residuals)

    plt.xlabel('x = Home Value')
    plt.ylabel('y = Residuals')
    plt.title('Residuals vs Home Value')
    plt.show()
    return

def regression_errors(y, yhat):
    
    df = pd.DataFrame(y, yhat)
    
    baseline = y.mean()
    
    df['baseline'] = baseline
    
    
    MSE = mean_squared_error(y, yhat)
    
    SSE = MSE * len(df)
    
    RMSE = mean_squared_error(y, yhat, squared=False)
    
    TSS = (mean_squared_error(y, df.baseline) *len(df))
    
    ESS = TSS - SSE
    
    return SSE, ESS, TSS, MSE, RMSE

def baseline_mean_errors(y):
    
    
    baseline = np.repeat(y.mean(), len(y))
    
    
    MSE_baseline = mean_squared_error(y, baseline)
    
    SSE_baseline = MSE_baseline * len(y)
    
    RMSE_baseline = mean_squared_error(y, baseline, squared=False)
    
    return SSE_baseline, MSE_baseline, RMSE_baseline

def better_than_baseline(y, yhat):
    
    baseline = np.repeat(y.mean(), len(y))
    
    r2 = r2_score(y, yhat)
    
    r2_baseline = r2_score(y, baseline)
    
    if r2 > r2_baseline:
        
        return True
    else:
        return False
    
def select_kbest(X, y, K):
    
    kbest = SelectKBest(f_regression, k=K)

    kbest.fit(X, y)
    
    return X.columns[kbest.get_support()]

def rfe(X, y, K):
    
    model = LinearRegression()
    
    rfe = RFE(model, n_features_to_select=K)

    rfe.fit(X, y)

    return pd.DataFrame({'rfe_ranking':rfe.ranking_}, index = X.columns)
    

def model_metrics(model, 
                  X_train, 
                  y_train, 
                  X_validate, 
                  y_validate, 
                  metric_df):
    '''
    model_metrics will use an sklearn model object to 
    create predictions after fitting on our training set, and add
    the model scores to a pre-established metric_df
    returns: metric_df
    **TODO: create a check to see if metric_df exists.  
    Create it if not
    '''
    
    
    # fit our model object
    model.fit(X_train, y_train['tax_value'])
    in_sample_pred = model.predict(X_train)
    out_sample_pred = model.predict(X_validate)
    model_name = input('Name for model?')
    y_train[model_name] = in_sample_pred
    y_validate[model_name] = out_sample_pred
 
    rmse_val = mean_squared_error(
    y_validate['tax_value'], out_sample_pred, squared=False)
    r_squared_val = explained_variance_score(
        y_validate['tax_value'], out_sample_pred)
    metric_df = metric_df.append({
        'model': model_name,
        'rmse': rmse_val,
        'r^2': r_squared_val
    }, ignore_index=True)
    
    return metric_df
    



