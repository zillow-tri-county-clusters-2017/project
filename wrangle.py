import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statistics import harmonic_mean

# modeling methods

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import explained_variance_score

from IPython.display import display, Markdown, Latex

import warnings
warnings.filterwarnings("ignore")

def get_wrangled():
    print('''
# this will pull in the DB from the RDBMS or cache
# utilizes get_db_url, gdb, and is dependent on an env.py file with credentials
# summarize will give a summary of a DataFrame
from wrangle import get_zillow, summarize, sfr, remove_outliers,\
     handle_missing_values, split_data_continuous


    ''')

# <  get_db_url             >
# <  gdb (Get DataBase)    >
# <  get_zillow             >
# <  nulls_by_row           >
# <  nulls_by_col           >
# <  summarize v.1.1        >
# <  column_value_counts    >
##<  SUMMARIZE V.1.0        >##
# <  sfr                    >
# <  clean_zillow           >
# <  split_data_continuous  >
# <  remove_outliers        >
# <  handle_missing_values  >


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~<  GET_DB_URL  >~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

def get_db_url(schema):
    import env
    user = env.username
    password = env.password
    host = env.host
    conn = f'mysql+pymysql://{user}:{password}@{host}/{schema}'
    return conn

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~<  GDB! (Get DataBase) >~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

def gdb(db_name, query):
    '''
    gdb(db_name, query):
    
        takes in    a (db_name) schema name from the codeup database ;dtype int
        and         a (query) to the MySQL server ;dtype int

        and         returns the query using pd.read_sql(query, url)
        having      created the url from my environment file
    '''
    from pandas import read_sql
    url = get_db_url(db_name)
    return read_sql(query, url)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~<  GET_ZILLOW  >~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

def get_zillow():
    '''
    This is basically the aqcuire function for the zillow data:
    -----------------------------------------------------------

    A DataFrame is returned with df. shape: (52_279, 62)

    The query used on the zillow schema on the codeup MySQL database:
       
    If pickled and available locally as filename: 'zillow_2017_transactions':

        The DataFrame is pulled from there instead.

    else:

        This will pickle the DataFrame and store it locally for next time
    '''
    # Set the filename for caching
    filename= 'zillow_2017_transactions'
    
    # if the file is available locally, read it
    if os.path.isfile(filename):
        df = pd.read_pickle(filename)
        
    else:
        # Read the SQL query into a DataFrame
        df = gdb('zillow',
        '''
            SELECT 	ps.id id,
			ps.parcelid parcel,
            ps.logerror logerror,
            ps.transactiondate trans_date,
            air.airconditioningdesc ac_type,
			arc.architecturalstyledesc arch_type,
			bui.buildingclassdesc build_type,
			hea.heatingorsystemdesc heat_type,
			pro.propertylandusedesc land_use_type,
			sto.storydesc story_type,
			typ.typeconstructiondesc construction_type,
            p.basementsqft basementsqft,
            p.bathroomcnt bathrooms,
            p.bedroomcnt bedrooms,
            p.buildingqualitytypeid quality_type,
            p.calculatedbathnbr calc_bath_n_bed,
            p.decktypeid deck_type,
            p.finishedfloor1squarefeet floor_1_sqft,
            p.calculatedfinishedsquarefeet tot_sqft,
            p.finishedsquarefeet12 sqft_12,
            p.finishedsquarefeet13 sqft_13,
            p.finishedsquarefeet15 sqft_15,
            p.finishedsquarefeet50 sqft_50,
            p.finishedsquarefeet6 sqft_6,
            p.fips fips,
            p.fireplacecnt fireplaces,
            p.fullbathcnt full_baths,
            p.garagecarcnt garages,
            p.garagetotalsqft garage_sqft,
            p.hashottuborspa hot_tub,
            p.latitude lat,
            p.longitude lon,
            p.lotsizesquarefeet lot_sqft,
            p.poolcnt pools,
            p.poolsizesum pool_sqft,
            p.pooltypeid10 pool_id10,
            p.pooltypeid2 pool_id2,
            p.pooltypeid7 pool_id7,
            p.propertycountylandusecode county_landuse,
            p.propertyzoningdesc zoning,
            p.rawcensustractandblock raw_tract_and_block,
            p.regionidcity city_id,
            p.regionidcounty county_id,
            p.regionidneighborhood neighborhood,
            p.regionidzip zip_code,
            p.roomcnt num_rooms,
            p.storytypeid stories_type,
            p.threequarterbathnbr three_quarter_baths,
            p.unitcnt units,
            p.yardbuildingsqft17 yard_sqft_17,
            p.yardbuildingsqft26 yard_sqft_26,
            p.yearbuilt year_built,
            p.numberofstories num_stories,
            p.fireplaceflag fireplace_flag,
            p.structuretaxvaluedollarcnt building_tax_value,
            p.taxvaluedollarcnt tax_value,
            p.assessmentyear year_assesed,
            p.landtaxvaluedollarcnt land_tax_value,
            p.taxamount tax_amount,
            p.taxdelinquencyflag tax_delinquency_flag,
            p.taxdelinquencyyear tax_delinquency_year,
            p.censustractandblock tract_and_block
            
		FROM predictions_2017 ps
LEFT JOIN properties_2017 p
ON ps.parcelid = p.parcelid

LEFT JOIN airconditioningtype air
ON p.airconditioningtypeid = air.airconditioningtypeid

LEFT JOIN architecturalstyletype arc
ON p.architecturalstyletypeid = arc.architecturalstyletypeid

LEFT JOIN buildingclasstype bui
ON p.buildingclasstypeid = bui.buildingclasstypeid

LEFT JOIN heatingorsystemtype hea
ON p.heatingorsystemtypeid = hea.heatingorsystemtypeid

LEFT JOIN propertylandusetype pro
ON p.propertylandusetypeid = pro.propertylandusetypeid

LEFT JOIN storytype sto
ON p.storytypeid = sto.storytypeid

LEFT JOIN typeconstructiontype typ
ON p.typeconstructiontypeid = typ.typeconstructiontypeid
    WHERE transactiondate >= "2017-01-01" 
		AND transactiondate < "2018-01-01"
        AND transactiondate < "2018-01-01"
        AND p.bedroomcnt > 0
        AND p.bathroomcnt > 0 
        AND p.calculatedfinishedsquarefeet > 0
        AND p.taxvaluedollarcnt > 0;
        ''')

        df['trans_date'] = pd.to_datetime(df.trans_date)
        drop_duplicate_predictions_rows = \
        df\
        [df.duplicated\
        (subset=['parcel'], keep='last')\
        == True]
        df = df.drop(index= drop_duplicate_predictions_rows.index)
        df.fips = df.fips.astype(object)
        df.fips = df.fips.map({6037.0: 'Los Angeles County',
                6059.0: 'Orange County',
                6111.0: 'Ventura County'
               })
        df.year_built = df.year_built.astype(object)
            
    # Pickle the DataFrame for caching (pickling is much faster than using .csv)
    df.to_pickle(filename)
    return df

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~<  NULLS_BY_ROW  >~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

def nulls_by_row(df):
    '''
    Takes in a DataFrame and tells us the number of rows with missing values
    '''
    num_missing = df.isnull().sum(axis=1)
    prnt_missing = num_missing / df.shape[1] * 100
    rows_missing = pd.DataFrame({'num_cols_missing': num_missing,
                             'percent_missing': prnt_missing,
                            })\
    .reset_index()\
    .groupby(['num_cols_missing', 'percent_missing'])\
    .count().reset_index().rename(columns={'index': 'count'})
    return rows_missing

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~<  NULLS_BY_COL  >~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

def nulls_by_col(df):
    '''
    
    '''
#     
    num_missing = df.isnull().sum(axis=0)
    prnt_missing = num_missing / df.shape[0] * 100
    cols_missing = pd.DataFrame({'num_rows_missing': num_missing,
                             'percent_missing': prnt_missing,
                            })
    null_cols = cols_missing[cols_missing['num_rows_missing'] > 0]
    print(f'Number of Columns with nulls: {len(null_cols)}')
    return null_cols


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~<  SUMMARIZE V.1.1 >~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

def summarize(df, cat_cols= None, too_long= 50, show_all= False, q= 10):
    '''
    Takes in a DataFrame and provides a summary of whats going on

    Parameters:
    ------------
    a Dataframe :
    
    cat_cols= None  : if you have them, otherwise they will be taken from object column types
    
    show_all= False :  if set to true:
                       it will print out the longest of the 
                       long markdown dataframes...BEWARE
    
    too_long= 50    : how many rows are too many rows in column.value_counts()??
    
    q = 10          : how many quartiles should numerical data be divided into?

    no return: just prints the:
    --------------
    df.head()
    df.info()
    df.describe()
    
    Null Values:
        By Column:
        By Row:

    df[each_separate_column].value_counts()
        Note: The object variables are not binned, the numerical variables are
        v1.1 -- Now you can control how much is displayed in long Markdown
    '''
#     print('test')
    display(Markdown(
    f'''
    DataFrame .head():
    -----------------
{df.head().to_markdown()}
    
    DataFrame .info():
    -----------------\n'''))
    print(df.info())
    display(Markdown(f'''\n    
    DataFrame .describe():
    -----------------
{df.describe().T.to_markdown()}
    
    Null Value Assessments:
    -----------------
        
        Nulls By Column:
{nulls_by_col(df).to_markdown()}
    -----------------
        
        Nulls By Row:
{nulls_by_row(df).to_markdown()}
    
    DataFrame .value_counts():
    -----------------
    
    '''))
    column_value_counts(df,
                        cat_cols, 
                        too_long, 
                        show_all, 
                        q
                       )                    
                    
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~<  COLUMN_VALUE_COUNTS >~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


def column_value_counts(df, cat_cols=None, too_long=50, show_all=False, q= 10):
    
    if cat_cols == None:
        num_cols = [col for col in df.columns if df[col].dtype != 'O']
        cat_cols = [col for col in df.columns if col not in num_cols]

    for col in df.columns:
        
        print('Column Name: ', col,'\n--------------')
        
        if col in cat_cols:
            print_this = df[col].value_counts(dropna=False)
            
            print('Categorical:\n - ', len(print_this), 'Categories')
            
            if (len(print_this) < too_long) | (show_all == True):
                
                display(Markdown(print_this.to_markdown()))
            
            else:
                print('\n',print_this)
            
            print('\n\n-------------')
        else: 
            print_this = df[col].value_counts(bins= q, sort=False, dropna=False)
            
            print('Numerical: Divided by Quartile\n - ', len(print_this), 'bins')
            
            if (len(print_this) < too_long) | (show_all == True):
                
                display(Markdown(print_this.to_markdown()))
            
            else:
                print(print_this)
                
            print('\n\n-------------')
    print('-----------------\n---End of Line---')

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~<  SUMMARIZE V.1.0 >~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

# def summarize(df, cat_cols= None):
#     '''
#     Takes in a DataFrame and provides a summary of whats going on

#     Parameters:
#     ------------
#     a Dataframe
#     cat_cols if you have them, otherwise they will be taken from object column types

#     no return: just prints the:
#     --------------
#     df.head()
#     df.info()
#     df.describe()
    
#     Null Values:
#         By Column:
#         By Row:

#     df[each_separate_column].value(counts)
#         Note: The object variables are not binned, the numerical variables are
#     '''
# #     print('test')
#     print(
#     f'''
#     DataFrame .head():
#     -----------------
# {df.head()}
    
#     DataFrame .info():
#     -----------------\n''')
#     print(df.info())
#     print(f'''\n    
#     DataFrame .describe():
#     -----------------
# {df.describe()}
    
#     Null Value Assessments:
#     -----------------
        
#         Nulls By Column:
# {nulls_by_col(df)}
#     -----------------
        
#         Nulls By Row:
# {nulls_by_row(df)}
    
#     DataFrame .value_counts():
#     -----------------
    
#     ''')
# #     print('test2')
#     if cat_cols == None:
#         num_cols = [col for col in df.columns if df[col].dtype != 'O']
#         cat_cols = [col for col in df.columns if col not in num_cols]


#     for col in df.columns:
#         print('Column Name: ', col,'\n--------------')
#         if col in cat_cols:
#             print(df[col].value_counts(dropna=False),'\n\n-------------')
#         else: 
#             print(df[col].value_counts(bins=10, sort=False, dropna=False),'\n\n-------------')
#     print('-----------------\n---End of Line---')
        
   

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~<  SFR  >~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

def SFR(df):
    '''
    Takes in a DataFrame from the zillow dataset 
    and returns only the Single family residential properties
    aslo as a DataFrame
    '''
    sfr = df[df.land_use_type == 'Single Family Residential']
    sfr = sfr[sfr.stories_type != 7]

    multi_unit = sfr[sfr.units > 1]

    sfr = sfr.drop(index=multi_unit.index)
    
    return sfr


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~<  CLEAN_ZILLOW  >~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

def clean_zillow():
    '''
    No parameters!
    ----------------

    This will output a clean dataframe of zillow information with outliers removed
    also, why model based on houses less than $100,000.00 or more than $1M 
    and we will reset the index, which will allow us to use the boxplots function for explore
    '''
# Set the filename for caching
    filename= 'clean_zillow'
    
# if the file is available locally, read it
    if os.path.isfile(filename):
        df = pd.read_pickle(filename)
        
    else:

        df= get_zillow()

        df = df.drop(columns= ['garages', 'garage_sqft', 'sqft_12'])

        outlier_cols = ['bathrooms',
        'bedrooms',
        'tot_sqft',
        'fireplaces',
        'full_baths',
        # 'garages',
        # 'garage_sqft',
        'lot_sqft',
        'tax_value',
        'land_tax_value',
        ]

        df = SFR(df)
        
# Set the cols to outlier_cols 
        cols = outlier_cols

# Use remove_outliers
        df = remove_outliers(df, 1.5, cols)
# fillna(0)
        # df.garage_sqft = df.garage_sqft.fillna(0)
        
        # df.garages = df. garages.fillna(0)

        df.hot_tub = df.hot_tub.fillna(0)

        df.pools = df.pools.fillna(0)

        df.tax_delinquency_flag = df.tax_delinquency_flag.map({'Y':1}).fillna(0)

        df= handle_missing_values(df)

        df.bedrooms = df.bedrooms.map({2: '_2_',
                    3: '_3_',
                    4: '_4_',
                    5: '_5_'
                    })

        df.bathrooms = pd.cut(df.bathrooms,
                        bins = (0,1,2,3,4,5), 
                        labels=('_1_', '_2_', '_3_', '_4_', '_5_')
                        )
                        
        df.bathrooms = df.bathrooms.astype('O')

        df = df.drop(columns= ['units', 'year_assesed', 'land_use_type', 'county_id'])

# Reset the index
        df = df.reset_index()
# drop the resulting 'index' column
        df = df.drop(columns='index')

        df.zip_code[19709] = 99675

        df = df.drop(index= df[df.zip_code.isnull()].index)
        
        df = df.drop(index= df[df.city_id.isnull()].index)

        df = df.drop(columns= ['heat_type', 'quality_type', 'zoning'])

        df = df.drop(index= df[df.isnull().sum(axis=1) > 0].index)

        df = df.drop(index= df[df.tract_and_block == df.tract_and_block.max()].index)

        df = df.drop(index= df[df.zip_code == df.zip_code.max()].index)

        df = df.drop(index= df[df.parcel == df.parcel.max()].index)

        df = df.drop(index= df[df.parcel == df.parcel.max()].index)

        df = df.drop(index= df[df.parcel == df.parcel.max()].index)

        df.full_baths = df.full_baths.map({1:'_1_',
                    2:'_2_',
                    3:'_3_',
                    4:'_4_'
                    })

        df['half_baths'] = 0

        for i in df[df.full_baths != df.bathrooms].index:
#     print(i)
            df.half_baths[i] += 1

        df = df.drop(columns= 'full_baths')
        

# Reset the index
        df = df.reset_index()
# drop the resulting 'index' column
        df = df.drop(columns='index')
        df = df.drop(columns= 'calc_bath_n_bed')
        df['house_age'] = (df.year_built.astype('int')-2017)*-1
        df = df.drop(columns='year_built')
        df = df.drop(columns='num_rooms')
        df.tax_delinquency_flag = df.tax_delinquency_flag.astype('int')
        df.tract_and_block = df.tract_and_block.astype('int').astype('str').str[4:].astype('int')
        df.pools = df.pools.astype('int')
        df.city_id = df.city_id.astype('int').astype('str')
        df.zip_code = df.zip_code.astype('int').astype('str')

        df.to_pickle(filename)
        return df


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~<  SPLIT_DATA_CONTINUOUS  >~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


def split_data_continuous(df, rand_st=123, with_baseline=False):
    '''
    Takes in: a pd.DataFrame()
          and a random state           ;if no random state is specifed defaults to [123]
          and a boolean value for with_baseline (=False) 
            if True: the baselines are computed for mean, median, mode, 
            the mean of those, the harmonic mean of those, and the harmonic mean of all predictions
          
      return: train, validate, test    ;subset dataframes
    '''
    from sklearn.model_selection import train_test_split
    train, test = train_test_split(df, test_size=.2, 
                               random_state=rand_st)
    train, validate = train_test_split(train, test_size=.25, 
                 random_state=rand_st)
    print(f'Prepared df: {df.shape}')
    print()
    print(f'Train: {train.shape}')
    print(f'Validate: {validate.shape}')
    print(f'Test: {test.shape}')
# here we add the if with_baselines to add all the possible baselines
    if with_baseline:
        baselines = ['mean_preds',
        'median_preds',
        'mode_preds',
        'm_mmm_preds',
        'hm_mmm_preds',
        'h_m_total_preds']
        # Use the get_baselines function
        train, validate, test = get_baselines(train, validate, test)
# Set the best basline RMSE to 1M so that it's above any reasonable baseline and baseline to none
        best_rmse = 1_000_000_000
        best_baseline = None
# test the RSME of each baseline and compare; sticking with only the lowest RSME
        for i in baselines:
            rmse_train = mean_squared_error(train.tax_value, train[i]) ** 0.5
            rmse_validate = mean_squared_error(validate.tax_value, validate[i]) ** 0.5

            if rmse_train < best_rmse:
                best_rmse = rmse_train
                best_baseline = i
                in_out = rmse_train/rmse_validate
# round the baseline values for human readability
        our_baseline = round(train[best_baseline].values[0])
# add to our dataframe
        train['baseline'] = our_baseline
# drop all the baselines we tested
        train = train.drop(columns= baselines)
# set the validate set with baseline and drop the others
        validate['baseline'] = our_baseline

        validate = validate.drop(columns= baselines)
# And the same for test
        test['baseline'] = our_baseline

        test = test.drop(columns= baselines)
            
        print(f'The {best_baseline} had the lowest RMSE: {round(best_rmse)} with an in/out of: {round(in_out,3)}')

    return train, validate, test

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~<  REMOVE_OUTLIERS  >~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

def remove_outliers(df, k=1.5, col_list=[]):
    ''' 
    Removes outliers from a list of columns in a dataframe 
    and return that dataframe
    
    PARAMETERS:
    ------------
    
    df    :   DataFrame that you want outliers removed from
    
    k     :   The scaler of IQR you want to use for tromming outliers
                 k = 1.5 gives a 8σ total range
    col_list : The columns to have outliers removed using
    '''
    # Create a column that will label our rows as containing an outlier value or not
    num_obs = df.shape[0]
    df['outlier'] = False
    for col in col_list:

        q1, q3 = df[col].quantile([.25, .75])  # get quartiles
        
        iqr = q3 - q1   # calculate interquartile range
        
        upper_bound = q3 + k * iqr   # get upper bound
        lower_bound = q1 - k * iqr   # get lower bound

        # update the outlier label any time that the value is outside of boundaries
        df['outlier'] = np.where(((df[col] < lower_bound) | (df[col] > upper_bound)) & (df.outlier == False), True, df.outlier)
    
    df = df[df.outlier == False]
    df = df.drop(columns=['outlier'])
    print(f"Number of observations removed: {num_obs - df.shape[0]}")
        
    return df


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~<  HANDLE_MISSING_VALUES  >~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

def handle_missing_values(df,
                          prop_req_cols = .5,
                          prop_req_rows = .75
                         ):
    threshold = int(round(prop_req_cols * len(df.index), 0))
    df = df.dropna(axis=1, thresh= threshold)
    threshold= int(round(prop_req_rows * len(df.columns), 0))
    df = df.dropna(axis=0, thresh = threshold)
    return df






