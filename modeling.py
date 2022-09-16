import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import explained_variance_score

from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

from statistics import harmonic_mean

# <  scale_data     >
# <  get_baselines  > 
# <  model_sets     >
# <  make_metrics   >
# <  make_models    >
# <  get_models     >
# <  evaluate       >
# 

def get_modeling():
    print(
        '''
from modeling import scale_data, get_baselines, model_sets, make_metrics,\
    get_models, evaluate
        '''
    )

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~<  SCALE_DATA  >~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

def scale_data(train, validate, test):
    '''
    scales the data using MinMaxScaler from SKlearn
    should only be the X_train, X_validate, and X_test
    '''
#     Remember incoming columns and index numbers to output DataFrames
    cols = train.columns
    train_index = train.index
    validate_index = validate.index
    test_index = test.index
    
#     Make the scaler
    scaler = MinMaxScaler()
    
#     Use the scaler
    train = scaler.fit_transform(train)
    validate = scaler.transform(validate)
    test = scaler.transform(test)
    
#     Reset the transformed datasets into DataFrames
    train = pd.DataFrame(train, columns= cols, index= train_index)

    validate = pd.DataFrame(validate, columns= cols, index= validate_index)

    test = pd.DataFrame(test, columns= cols, index= test_index)
    
    return train, validate, test

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~<  GET_BASELINES  >~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def get_baselines(train, validate, test, y='tax_value'):
    '''
    Parameters:
    --------------------
    
    train           :       your training set
    validate        :       your validation set
    test            :       your test set
    y=  (tax_value) :       the target variable
    '''
    # Various methods for baseline predictions
    # We'll make new columns for each, and stick them in our training set

    train['mean_preds'] = \
    train[y].mean()

    train['median_preds'] = \
    train[y].median()

    train['mode_preds'] = \
    train[y].round(1).mode()[0]

    train['m_mmm_preds'] = \
    sum([train[y].mean(), train[y].median(), train[y].round(1).mode()[0]])/3

    train['hm_mmm_preds'] = \
    harmonic_mean([train[y].mean(), train[y].median(), train[y].round(1).mode()[0]])

    train['h_m_total_preds'] = \
    harmonic_mean(train[y])

    train_index = train.index.tolist()
    #  broke out the number ... damn, i need to rewrite all of this to use enumerate SMH
    one = train_index[0]

    baselines = ['mean_preds',
    'median_preds',
    'mode_preds',
    'm_mmm_preds',
    'hm_mmm_preds',
    'h_m_total_preds']

    for i in baselines:
        validate[i] = train[i][one]
        test[i] = train[i][one]
    
    return train, validate, test

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~<  MODEL_SETS  >~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

def model_sets(train, validate, test, scale_X_cols=True, target='tax_value', with_baseline= True):
    '''
    Takes in the train, validate, and test sets and returns the X_train, y_train, etc. subsets

    Should look like: 
            
            X_train, y_train, x_validate, y_validate, X_test, y_test = model_sets(train, validate, test)

    Parameters:
    ------------------
    train     :  your split        train data
    
    validate  :  your split   validation data
    
    test      :  your split         test data

    scale_X_cols  :  (=True) this will invoke the scale_data function to scale the data using MinMaxScaler
                        False will skip the scaling and return the unscaled version

    target        :  (='tax_value') is your target variable, set to tax_value, cause that's what I was working on ;)


    Returns:
    ------------------
    X_train, y_train,
    X_validate, y_validate,
    X_test, y_test

    These can be used to train and evaluate model performance!

    '''

    # use forloop to get columns for X_cols exckuding the target and the baseline
    X_cols = []
    for i in train.columns:
        if i not in [target, 'baseline']:
            X_cols.append(i)
    if with_baseline:
        y_cols = [target, 'baseline']
    else:
        y_cols = [target]

    # print what they are for the users reference
    print(f'\nX_cols = {X_cols}\n\ny_cols = {y_cols}\n\n')

    # set the X_ and y_ for train, validate and test
    X_train, y_train = train[X_cols], train[y_cols]
    X_validate, y_validate = validate[X_cols], validate[y_cols]
    X_test, y_test = test[X_cols], test[y_cols]

    # if scale_X_cols is true then we send all of our X_ columns trhough the scale_data function
    if scale_X_cols:
        X_train, X_validate, X_test = scale_data(X_train, X_validate, X_test)

    # 
    return X_train, y_train, X_validate, y_validate, X_test, y_test



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~<  MAKE_METRICS  >~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

def make_metrics(zillow, target='tax_value'):
    '''
    takes in a list of DataFrames:
    -------------------
            zillow = [df, X_train, y_train, X_validate, y_validate, X_test, y_test]

    and a target variable
    '''

    # Make metrics
    rmse = mean_squared_error(zillow[2][target], zillow[2].baseline) ** (1/2)
    r2 = explained_variance_score(zillow[2][target], zillow[2].baseline)

    rmse_v = mean_squared_error(zillow[4][target], zillow[4].baseline) ** (1/2)
    r2_v = explained_variance_score(zillow[4].tax_value, zillow[4].baseline)
# Setup the metric dataframe
    metric_df = pd.DataFrame(data=[{
        'model': 'baseline',
        'rmse_train': hr(rmse),
        'r^2': r2,
        'rmse_validate': hr(rmse_v),
        'r^2_validate': r2_v}])

    return metric_df


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~<  MAKE_MODELS  >~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def make_models(zillow, target='tax_value'):
    '''
    Makes trained models from the X_train and y_train sets and evaluates them on validation sets

    models include:
                    Linear Regression
                    Lasso Lars
                    GLM (Tweedie Reg)

    Returns:
    -----------------
    a list of the X_ y_ dataframes, the models in a dataframe, and the metrics for the models in a dataframe
    zillow, models, metric_df

    '''
    
    # Name and make models
    models = pd.DataFrame(\
    {'model_name':['Linear Regression',
                'Lasso Lars',
                'GLM (Tweedie Reg)',
                ],
    'made_model': [LinearRegression(normalize=True),
                LassoLars(alpha=1, random_state=123),
                TweedieRegressor(power=(1), alpha=0)
                ],}
    )

    # Fit the models
    models['fit_model'] = models.model_name
    for i, j in enumerate(models.made_model):
        models['fit_model'][i] = j.fit(zillow[1], zillow[2].tax_value)

    # Make Model Predictors
    models['predict_model'] = models.model_name
    for i, j in enumerate(models.fit_model):
        models.predict_model[i] = j.predict

    # Make metrics_df
    metric_df = make_metrics(zillow)

    # Fill metrics_df with predictions
    for i, j in enumerate(models.predict_model):
        
    #     Make prediction: zillow[2] is y_train, [4] is y_validate, j is the .predict
        zillow[2][models.model_name[i]] = j(zillow[1])
        zillow[4][models.model_name[i]] = j(zillow[3])
        
    # Make metrics
            
        rmse = mean_squared_error(zillow[2][target], j(zillow[1])) ** (1/2)
        r2 = explained_variance_score(zillow[2][target], j(zillow[1])) 

        rmse_v = mean_squared_error(zillow[4][target], j(zillow[3])) ** (1/2)
        r2_v = explained_variance_score(zillow[4][target], j(zillow[3]))

        metric_df = metric_df.append([{
            'model': models.model_name[i],
                'rmse_train': hr(rmse),
                'r^2': r2,
                'rmse_validate': hr(rmse_v),
                'r^2_validate': r2_v}])

    return zillow, models, metric_df

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~<  GET_MODELS  >~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

def get_models():
    '''
    No Parameters!
    -----------------
    just run the function saving the three outputs

    outputs a list of the X_ y_ dataframes, the models in a dataframe, 
    and the metrics for the models in a dataframe

    zillow, models, metric_df 
    '''
    # grab the clean data
    df = clean_zillow()
    # split it
    train, validate, test = split_data_continuous(df, with_baseline=True)
    # get the model sets
    X_train, y_train, X_validate, y_validate, X_test, y_test = model_sets(train, validate, test)
    # make the list of sets to put into the models
    zillow = [df, X_train, y_train, X_validate, y_validate, X_test, y_test]
    # Use the make_models function
    zillow, models, metric_df = make_models(zillow)
    # output the results
    return zillow, models, metric_df

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~<  evaluate  >~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def evaluate(y_validate):
    '''
    evaluate the models on the y_validate set 
    '''
    # y_validate.head()
    plt.figure(figsize=(16,8))
    plt.plot(y_validate.tax_value, y_validate.baseline, alpha=1, color="gray", label='_nolegend_')
    plt.annotate("Baseline: Predict Using Mean", (16, 9.5))
    plt.plot(y_validate.tax_value, y_validate.tax_value, alpha=.5, color="blue", label='_nolegend_')
    plt.annotate("The Ideal Line: Predicted = Actual", (.5, 3.5), rotation=15.5)

    plt.scatter(y_validate.tax_value, y_validate['Linear Regression'], 
                alpha=.2, color="red", s=100, label="Model: LinearRegression")
    plt.scatter(y_validate.tax_value, y_validate['GLM (Tweedie Reg)'], 
                alpha=.2, color="yellow", s=100, label="Model: TweedieRegressor")
    plt.scatter(y_validate.tax_value, y_validate['Lasso Lars'], 
                alpha=.2, color="green", s=100, label="Model: Lasso Lars")
    plt.legend()
    plt.xlabel("Actual Tax Value")
    plt.ylabel("Predicted Tax Value")
    plt.title("Where are predictions more extreme? More modest?")
    plt.annotate("The polynomial model appears to overreact to noise", (2.0, -10))
    plt.annotate("The OLS model (LinearRegression)\n appears to be most consistent", (15.5, 3))
    plt.show()