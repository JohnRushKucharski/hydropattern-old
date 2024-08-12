'''Test main module.'''
import unittest

import numpy as np
import pandas as pd

from main import (timing_parser, duration_parser, magnitude_parser)

single_ts_flow = np.ones(365)
single_ts_flow[305] = 10 # nov 2 (doy: 306)
single_ts_flow[296:304] = 0  # oct 24-31 (doy: 297-304)
df = pd.DataFrame({'col1': single_ts_flow, 'col2': np.arange(1, 366, 1)})

class TestTimingParser(unittest.TestCase):
    '''Tests for timing_parser function.'''
    def test_timing_parser_results(self):
        '''Test timing_parser function.'''
        data, order = [305, 335], 1
        timing = timing_parser(data, order)

        results = timing.fx(df)
        exp = np.zeros(365)
        exp[304:335] = 1 # nov 1 - dec 1
        np.testing.assert_array_equal(results, exp)

pre_duration_output = np.zeros((365, 4))
pre_duration_output[296:304, 0:2] = 1

class TestDurationParser(unittest.TestCase):
    '''Tests for duration_parser function.'''
    def test_duration_parser_results(self):
        '''Test duration_parser function.'''
        data, order = [">", 7], 3
        duration = duration_parser(data, order)

        results = duration.fx(df, pre_duration_output)
        exp = np.zeros(365)
        exp[296:304] = 1 # oct 24-31
        np.testing.assert_array_equal(results, exp)

pre_magnitude_output = np.zeros((365, 3))

class TestMagnitudeParser(unittest.TestCase):
    '''Tests for magnitude_parser function.'''
    def test_magnitude_parser_results(self):
        '''Test magnitude_parser function.'''
        data, order = [">", 2.0], 1
        magnitude = magnitude_parser(data, order)

        results = magnitude.fx(df, pre_magnitude_output)
        exp = np.zeros(365)
        exp[305] = 1 # oct 24-31
        np.testing.assert_array_equal(results, exp)
