import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


# <  boxplots  >
# <  hists  >
# <  heatmaps  >
# <  slicer  >
# <  haversine  >

def get_viz():
    print(
        '''
from vis import boxplots, hists, heatmaps, slicer, haversine
        '''
    )
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~<  boxplots  >~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

def boxplots(df, excluding=False):
    '''
    make boxplots from all the columns in a dataframe, excluding anything you want to exclude
    '''
    # Set the cols for use in creating the boxplots
    cols = [col for col in df.columns if col not in [excluding]]
# set the figure and for loop to plot each column
    plt.figure(figsize=(16, 20))
    for i, col in enumerate(cols):

        # i starts at 0, but plot nos should start at 1
        plot_number = i + 1 

        # Create subplot.
        plt.subplot(1, len(cols), plot_number)

        # Title with column name.
        plt.title(col)

        # Display boxplot for column.
        sns.boxplot(data=df[col])

        # Hide gridlines.
        plt.grid(False)

    plt.show()


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~<  hists  >~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


def hists(df, exclude='', granularity=5): 

    '''
    Make histograms of all the columns in a dataframe, except any you want to exclude, and you can set the number of bins with granularity
    '''   
    # Set figure size. Went with 4x for the width:height to display 4 graphs... future version could have these set be the DataFrame columns used
    plt.figure(figsize=(16, 4))

    # List of columns
    cols = [col for col in df.columns if col not in [exclude]]

    for i, col in enumerate(cols):

        # i starts at 0, but plot nos should start at 1
        plot_number = i + 1 

        # Create subplot.
        plt.subplot(1, len(cols), plot_number)

        # Title with column name.
        plt.title(col)

        # Display histogram for column.
        df[col].hist(bins=granularity)

        # Hide gridlines.
        plt.grid(False)

        # turn off scientific notation
        plt.ticklabel_format(useOffset=False)

        # mitigate overlap
        plt.tight_layout()

    plt.suptitle(f'{hr(len(df),prefix="")} \
Houses in $ Range > {hr(df.tax_value.min())} - {hr(df.tax_value.max())} <',
                 y=1.05,
                 size=20
                )
    plt.show()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~<  heatmaps  >~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def heatmaps(train, num_cols):
    '''
    Makes a correlation Heatmap for the numerical variables
    
    Parameters:
    ---------------
        train : DataFrame
        num_cols : a list of the numerical varibles
    
    '''
    # create the correlation matrix using pandas .corr()
    zill_corr = train[cum_cols].corr()

    # pass my correlation matrix to a heatmap
    kwargs = {'alpha':.9,
            'linewidth':3, 
            'linestyle':'-',
            'linecolor':'black'}

    sns.heatmap(zill_corr, cmap='Purples', annot=True,
            mask=np.triu(zill_corr), **kwargs)
    plt.show()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~<  slicer  >~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

def slicer(df, min=0, max=1_000_000, step=50_000):
    '''
    SLICER:
    ------------------------------
    Takes in a DataFrame and provides Histograms of each slice of tax_value
    the min max and step size of the bins can be set.
    also the standard deviation of each slice is output
    '''
    
    for i in range(min, max, step):
        price_range = 50_000
        houses = df[(i < df.tax_value) & (df.tax_value < i + price_range)]
        
    #     print(f'The standard deviation of houses between:\n\
    #     {i} and {i+price_range} is:\n ${round(houses.tax_value.std())}')
        
    #     print(houses.tax_value.describe())
        
        hists(houses, 'date')
        
        print(f'''
        σ = {round(houses.beds.std())} beds         |     \
    σ = {round(houses.baths.std())} baths      |     \
    σ = {round(houses.area.std())} sqft      |     \
    σ = {hr(houses.tax_value.std())} 
        ''')

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~<  haversine  >~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


from math import radians, cos, sin, asin, sqrt

def haversine(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """
    Calculate the great circle distance between two points on the 
    earth (specified in decimal degrees), returns the distance in
    meters.
    All arguments must be of equal length.
    :param lon1: longitude of first place
    :param lat1: latitude of first place
    :param lon2: longitude of second place
    :param lat2: latitude of second place
    :return: distance in meters between the two sets of coordinates
    """
    # Convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371 # Radius of earth in kilometers
    return c * r

