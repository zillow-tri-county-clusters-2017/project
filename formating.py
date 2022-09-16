
import os
import math
import pandas as pd
import numpy as np
from statistics import harmonic_mean


from IPython.display import display, Markdown, Latex

#   <  bold  >
#   <  underline  >
#   <  strike  >
#   <  hr (HUMAN READABLE)  >
#   <  percent  >

def get_formating():
    print(
        '''
from formating import bold, underline, strike, hr, percent
        '''
    )

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~<  BOLD  >~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

def bold(text):
    result = '\033[1m' + text + '\033[0m'
    return result


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~<  UNDERLINE  >~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

def underline(text):
    result = ''
    for c in text:
        result = result + c + '\u0332'
    return result

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~<  STRIKE  >~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

def strike(text):
    result = ''
    for c in text:
        result = result + c + '\u0336'
    return result

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~<  HR (HUMAN READABLE)  >~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

def hr(n, suffix='', places=2, prefix='$'):
    '''
    Return a human friendly approximation of n, using SI prefixes

    '''
    prefixes = ['','K','M','B','T']
    
    # if n <= 99_999:
    #     base, step, limit = 10, 4, 100
    # else:
    #     base, step, limit = 10, 3, 100

    base, step, limit = 10, 3, 100

    if n == 0:
        magnitude = 0 #cannot take log(0)
    else:
        magnitude = math.log(n, base)

    order = int(round(magnitude)) // step
    return '%s%.1f %s%s' % (prefix, float(n)/base**(order*step), \
    prefixes[order], suffix)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~<  Percent  >~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

class percent(float):
    def __str__(self):
        return '{:.2%}'.format(self)
    def __repr__(self):
        return '{:.2%}'.format(self)

