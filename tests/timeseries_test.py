'''Tests timeseries.py'''
import unittest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1] / 'src'))
from timeseries import first_day_of_water_year, to_day_of_water_year, Timeseries # type: ignore pylint: disable=import-error wrong-import-position

import pandas as pd # pylint: disable=wrong-import-position wrong-import-order

class TestUtilities(unittest.TestCase):
    '''Tests timeseries.py functions.'''
    def test_first_day_of_water_year(self):
        '''Tests first_day_of_water_year.'''
        self.assertEqual(first_day_of_water_year(day=29, month=2), 59)
        self.assertEqual(first_day_of_water_year(day=1, month=10), 274)

    def test_first_day_of_water_year_raises_error(self):
        '''Tests first_day_of_water_year raises error.'''
        with self.assertRaises(ValueError):
            first_day_of_water_year(1, 0)
        with self.assertRaises(ValueError):
            first_day_of_water_year(1, 32)

    def test_to_day_of_water_year_at_start_of_wy(self):
        '''Tests to_day_of_water_year.'''
        self.assertEqual(to_day_of_water_year(pd.Timestamp('1900-10-01'), 274), 1)
        self.assertEqual(to_day_of_water_year(pd.Timestamp('1904-10-01'), 274), 1)
    def test_to_day_of_water_year_at_end_of_wy(self):
        '''Tests to_day_of_water_year before start.'''
        self.assertEqual(to_day_of_water_year(pd.Timestamp('1900-09-30'), 274), 365)
        self.assertEqual(to_day_of_water_year(pd.Timestamp('1904-09-30'), 274), 365)
    def test_to_day_of_water_year_at_end_of_yr(self):
        '''Tests to_day_of_water_year at start.'''
        self.assertEqual(to_day_of_water_year(pd.Timestamp('1900-12-31'), 274), 92)
        self.assertEqual(to_day_of_water_year(pd.Timestamp('1904-12-31'), 274), 92)
    def test_to_day_of_water_year_after_end_of_yr(self):
        '''Tests to_day_of_water_year at start.'''
        self.assertEqual(to_day_of_water_year(pd.Timestamp('1900-2-28'), 274), 151)
        self.assertEqual(to_day_of_water_year(pd.Timestamp('1904-2-29'), 274), 151)
    def test_to_day_of_water_year_raises_error(self):
        '''Tests to_day_of_water_year raises error.'''
        with self.assertRaises(ValueError):
            to_day_of_water_year(pd.Timestamp('1900-10-01'), 0)
        with self.assertRaises(ValueError):
            to_day_of_water_year(pd.Timestamp('1904-10-01'), 366)

class TestTimeseries(unittest.TestCase):
    '''Tests Timeseries class.'''
    def test_from_dataframe_good_data(self):
        '''Tests from_dataframe with good data.'''
        df = pd.DataFrame({'time': pd.date_range(start='1/1/1900', periods=10),
                           'value': range(10)}).set_index('time')
        self.assertEqual(Timeseries.from_dataframe(df).data.shape, (10, 1))

    def test_day_of_water_year_to_date_no_leap_yr(self):
        '''Tests day_of_water_year_to_datetime.'''
        df = pd.DataFrame({'time': pd.date_range(start='1/1/1900', periods=365),
                           'value': range(365)}).set_index('time')
        ts = Timeseries.from_dataframe(df, first_dowy=274)
        self.assertEqual(ts.day_of_water_year_to_date(dowy=1, year=1900),
                         pd.Timestamp('1900-10-01'))
        self.assertEqual(ts.day_of_water_year_to_date(dowy=92, year=1900),
                         pd.Timestamp('1900-12-31'))
        self.assertEqual(ts.day_of_water_year_to_date(dowy=93, year=1900),
                         pd.Timestamp('1900-01-01'))
        self.assertEqual(ts.day_of_water_year_to_date(dowy=365, year=1900),
                         pd.Timestamp('1900-09-30'))

    def test_day_of_water_year_to_date_last_dowy_equals_365(self):
        '''Tests day_of_water_year_to_datetime.'''
        df = pd.DataFrame({'time': pd.date_range(start='1/1/1900', periods=365),
                           'value': range(365)}).set_index('time')
        ts = Timeseries.from_dataframe(df, first_dowy=274)
        date = ts.day_of_water_year_to_date(dowy=365, year=1900)
        self.assertEqual(date, pd.Timestamp('1900-09-30'))
