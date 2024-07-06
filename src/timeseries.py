'''
Input data structure for analysis.
'''

import pandas as pd

def first_day_of_water_year(month: int, day: int, leap_year: bool = False) -> int:
    '''Returns the day of the year that is the first day of the water year.'''
    if leap_year:
        # 1904 is a leap year
        return pd.Timestamp(f'1904-{month}-{day}').dayofyear
    else:
        # 1900 is not a leap year
        return pd.Timestamp(f'1900-{month}-{day}').dayofyear
