'''Tests timeseries.py'''

import unittest

from src.timeseries import first_day_of_water_year

class TestTimeseries(unittest.TestCase):
    '''Tests timeseries.py.'''
    def test_first_day_of_water_year(self):
        '''Tests first_day_of_water_year.'''
        self.assertEqual(first_day_of_water_year(10, 1), 274)
        self.assertEqual(first_day_of_water_year(10, 1, leap_year=True), 275)
