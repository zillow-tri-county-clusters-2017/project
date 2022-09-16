from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report


def get_tests():
    print(
        '''
from tests import output_chi2_contingency

        '''
    )

def output_chi2_contingency(observed):
    '''    
    This function will display the $χ\^2$ contingency output:
    
    Observed
    [[ aₒ xₒ]
     [ bₒ yₒ]]

    Expected
    [[ aₑ xₑ]
     [ bₑ yₑ]]

    ----
    chi^2 = χ^2
    p     = p-value
    
    Parameters
    ----------
    observed : array_like
    The contingency table. The table contains the observed frequencies
    (i.e. number of occurrences) in each category.  In the two-dimensional
    case, the table is often described as an "R x C table".
    '''
    from scipy import stats
    chi2, p, degf, expected = stats.chi2_contingency(observed)
    #output values
    print('Observed')
    print(observed.values)
    print('\nExpected')
    print(expected.astype(int))
    print('\n----')
    print(f'chi^2 = {chi2:.4f}')
    print(f'p     = {p:.4f}')
    print(f'degf  = {degf:.4f}')
    return p