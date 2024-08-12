'''Tests for src/chacteristic.py'''
import unittest

import numpy as np
import pandas as pd

from src.patterns import (comparison_fx,
                          moving_average, is_dowy_timeseries,
                          timing_fx, magnitude_fx, duration_fx)

class TestComparisionFx(unittest.TestCase):
    '''Tests for comparison_fx function.'''
    # Test for single bound
    def test_comparison_fx_lt(self):
        '''Test comparison_fx function.'''       
        fx = comparison_fx('<', 5, None, None)
        self.assertTrue(fx(4))
        self.assertFalse(fx(5))
        self.assertFalse(fx(6))

    def test_comparison_fx_le(self):
        '''Test comparison_fx function.''' 
        fx = comparison_fx('<=', 5, None, None)
        self.assertTrue(fx(4))
        self.assertTrue(fx(5))
        self.assertFalse(fx(6))

    def test_comparison_fx_gt(self):
        '''Test comparison_fx function.''' 
        fx = comparison_fx('>', 5, None, None)
        self.assertFalse(fx(4))
        self.assertFalse(fx(5))
        self.assertTrue(fx(6))

    def test_comparison_fx_ge(self):
        '''Test comparison_fx function.''' 
        fx = comparison_fx('>=', 5, None, None)
        self.assertFalse(fx(4))
        self.assertTrue(fx(5))
        self.assertTrue(fx(6))

    def test_comparison_fx_eq(self):
        '''Test comparison_fx function.''' 
        fx = comparison_fx('=', 5, None, None)
        self.assertFalse(fx(4))
        self.assertTrue(fx(5))

    def test_comparison_fx_ne(self):
        '''Test comparison_fx function.''' 
        fx = comparison_fx('!=', 5, None, None)
        self.assertTrue(fx(4))
        self.assertFalse(fx(5))


    # Test for two bounds
    def test_comparison_fx_btwn(self):
        '''Test comparison_fx function.'''
        fx = comparison_fx('<', 3, '<', 5)
        self.assertTrue(fx(4))
        self.assertFalse(fx(5))
        self.assertFalse(fx(6))

    def test_comparison_fx_btwneq(self):
        '''Test comparison_fx function.'''
        fx = comparison_fx('<=', 3, '<=', 5)
        self.assertTrue(fx(4))
        self.assertTrue(fx(5))
        self.assertFalse(fx(6))

    def test_comparison_fx_btwnop(self):
        '''Test comparison_fx function.'''
        fx = comparison_fx('>', 5, '>', 3)
        self.assertTrue(fx(4))
        self.assertFalse(fx(5))
        self.assertFalse(fx(6))

    def test_comparison_fx_btwnopeq(self):
        '''Test comparison_fx function.'''
        fx = comparison_fx('>=', 5, '>=', 3)
        self.assertTrue(fx(4))
        self.assertTrue(fx(5))
        self.assertFalse(fx(6))

class TestMovingAverage(unittest.TestCase):
    '''Tests for moving_average function.'''
    def test_moving_average(self):
        '''Test moving_average function.'''
        i = np.array([1, 2, 3, 4, 5, 6])
        self.assertTrue(np.array_equal(
            moving_average(i, 3),
            np.array([np.nan, np.nan, 2., 3., 4., 5.]), equal_nan=True))

    def test_moving_average_period1_returns_input(self):
        '''Test moving_average function.'''
        i = np.array([1, 2, 3, 4, 5, 6])
        self.assertTrue(np.array_equal(
            moving_average(i, 1),
            i, equal_nan=True))

    def test_moving_average_min_periods(self):
        '''Test moving_average function.'''
        i = np.array([1, 2, 3, 4, 5, 6])
        self.assertTrue(np.array_equal(
            moving_average(i, 3, min_periods=1),
            np.array([1, 1.5, 2., 3., 4., 5.]), equal_nan=True))

    def test_moving_average_min_periods2(self):
        '''Test moving_average function.'''
        i = np.array([1, 2, 3, 4, 5, 6])
        self.assertTrue(np.array_equal(
            moving_average(i, 3, min_periods=2),
            np.array([np.nan, 1.5, 2., 3., 4., 5.]), equal_nan=True))

class TestIsDowyTimeseries(unittest.TestCase):
    '''Tests for is_dowy_timeseries function.'''
    def test_is_dowy_timeseries(self):
        '''Test is_dowy_timeseries function.'''
        self.assertTrue(is_dowy_timeseries([1, 2, 3.0, 4]))

    def test_is_dowy_timeseries_false_for_nonint(self):
        '''Test is_dowy_timeseries function.'''
        self.assertFalse(is_dowy_timeseries([1, 2, 3.5, 4]))

df = pd.DataFrame({'col1': [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
                   'col2': [1, 2, 3, 4, 5, 6]})

class TestCharacteristicFx(unittest.TestCase):
    '''Tests for Characteristic fxs.'''
    def test_timing_fx(self):
        '''Test timing_fx function.'''
        fx = timing_fx(comparison_fx('<', 3, '<', 6))
        self.assertTrue(np.all(fx(df) == np.array([0, 0, 0, 1, 1, 0])))

    def test_magnitude_fx(self):
        '''Test magnitude_fx function.'''
        fx = magnitude_fx(comparison_fx('>', 50.0, None, None))
        self.assertTrue(np.all(fx(df) == np.array([0, 0, 0, 0, 0, 1])))

    def test_duration_fx_whole_order3(self):
        '''Test duration_fx function.'''
        order = 3
        o = np.ones(shape=(len(df), order-1))
        fx = duration_fx(comparison_fx('>', 5, None, None), order)
        self.assertTrue(np.all(fx(df, o) == np.ones(len(df))))
    def test_duration_fx_end_order3(self):
        '''Test duration_fx function.'''
        order = 3
        o = np.ones(shape=(len(df), order-1))
        o[2,:] = 0 # breaks up the streak of 1s
        fx = duration_fx(comparison_fx('>=', 3, None, None), order)
        self.assertTrue(np.all(fx(df, o) == np.array([0, 0, 0, 1, 1, 1])))
    def test_duration_fx_mid_order3(self):
        '''Test duration_fx function.'''
        order = 3
        o = np.zeros(shape=(len(df), order-1))
        o[2:5,:] = 1 # breaks up the streak of 1s
        fx = duration_fx(comparison_fx('>=', 3, None, None), order)
        self.assertTrue(np.all(fx(df, o) == np.array([0, 0, 1, 1, 1, 0])))
    def test_duration_fx_startstop_order3(self):
        '''Test duration_fx function.'''
        order = 3
        o = np.ones(shape=(len(df), order-1))
        o[0,:] = 0 # breaks up the streak of 1s
        o[3,:] = 0 # breaks up the streak of 1s
        fx = duration_fx(comparison_fx('>', 1, None, None), order)
        self.assertTrue(np.all(fx(df, o) == np.array([0, 1, 1, 0, 1, 1])))
    def test_duration_fx_start_ordermismatch(self):
        '''Test duration_fx function.'''
        order = 3
        o = np.zeros(shape=(len(df), 4))
        o[0:5,2:4] = 1 # add 1s in columns that matter
        fx = duration_fx(comparison_fx('>', 1, None, None), order)
        self.assertTrue(np.all(fx(df, o) == np.array([1, 1, 1, 1, 1, 0])))
