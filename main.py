'''Entry point for CLI application.'''
import tomllib
from enum import Enum
from typing import Any

import typer
import numpy as np

from src.timeseries import Timeseries
import src.patterns as patterns

__version__ = '0.0.0'
__app_name__ = 'hydropattern'

app = typer.Typer()
state = {}

# @app.command(context_settings={'allow_extra_args': True})
# def load_timeseries(ctx: typer.Context,
#                     path: str, first_day_of_water_year: int=1):
#     '''Plot timeseries data in *.csv file at specified path.'''
#     state['timeseries'] = load_data(path, first_day_of_water_year, parse_kwargs(ctx))
#     print(state['timeseries'].data)
# region CLI Commands

# region Commands
@app.command()
def load_configuration(path: str = typer.Argument(..., help='''
                                               Path to *.toml file containing component configuration.
                                               See README.md @''')):
    '''Load configuration data from *.toml file.'''
    # load time series too (with all those options)
    # option to provide directory for all files (if directory, else single file)
    # option for strict order for evaluating component characteristics (maybe add to toml)
    # maybe add directory or file option for toml file
    # add timeseries option to toml file (date format, first day of water year)
    data = load_config_file(path)
    ts = parse_timeseries_data(data['timeseries'])
    components = parse_components(data['components'])
    out = patterns.evaluate_patterns(ts.data, components)
    print(out[0].iloc[290:310, np.r_[0:3,5:14]])

@app.command()
def plot_timeseries(path: str = typer.Argument(..., help='''
                                               Path to *.csv file containing data to plot.
                                               See README.md @ https://github.com/JohnRushKucharski/hydropattern
                                               for more information on the expected format of the data file.''' # pylint: disable=line-too-long
                                               ),
                    columns: list[int] = typer.Argument(None, help='''
                                                        Column(s) in data to plot, 
                                                        by default the first column is plotted.'''
                                                        ),
                    date_format: str = typer.Option('', '--format', '-f', help='''
                                                    Pandas will infer the date format from the data with a warning. 
                                                    If the date format code for the time column is known, it can be specified here. 
                                                    See: https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior''' # pylint: disable=line-too-long
                                                    ),
                    first_day_of_water_year: int = typer.Option(1, '--1st-day-of-WY', '-d', help='''
                                                                Integer day of year ths is the first day of water year. 
                                                                See: https://en.wikipedia.org/wiki/Water_year''' # pylint: disable=line-too-long
                                                                )):
    '''Plots timeseries of data using pandas and matplotlib.'''
    #todo: add additional optional argument for comparison file, split x-y axis, output path, etc.
    ts = load_timeseries(path, first_day_of_water_year, date_format)
    ts.plot_timeseries(data_columns=columns)
# endregion

def parse_kwargs(ctx: typer.Context) -> dict:
    '''Parse kwargs.'''
    kwargs = {}
    if len(ctx.args) % 2 != 0:
        print(f'Extra arguments: {ctx.args} were ignored, \
            because they could not be organized in key-value pairs.')
    else:
        i = 0
        while i < len(ctx.args):
            kwargs[ctx.args[i]] = ctx.args[i + 1]
            i += 2
    return kwargs

def load_config_file(path: str) -> dict[str, Any]:
    '''Load configuration file.'''
    with open(path, 'rb') as f:
        data = tomllib.load(f)
    return data

def parse_timeseries_data(data: dict[str, Any]) -> Timeseries:
    '''Parse timeseries data.'''
    path = data['path']
    date_format = data['date_format'] if 'date_format' in data else ''
    first_day_of_water_year = data['first_day_of_water_year'] if 'first_day_of_water_year' in data else 1 # pylint: disable=line-too-long
    return load_timeseries(path, first_day_of_water_year, date_format)

def load_timeseries(path: str, first_day_of_water_year: int, date_format: str) -> None:
    '''Load data.'''
    return Timeseries.from_csv(path, first_day_of_water_year, date_format)

def parse_components(data: dict[str, Any]) -> list[patterns.Component]:
    '''Build components.'''
    components = []
    for component_name, elements in data.items():
        characteristics: list[tuple[str, Any]] = []
        # since python 3.7 dictionaries are ordered
        # so the order of the elements is preserved
        verbose, success, order = True, True, 1
        # verbose, success set to defaults to start
        for name, metrics in elements.items():
            match name:
                case 'timing':
                    order = 1 if verbose else order
                    characteristics.append(timing_parser(metrics, order))
                case 'magnitude':
                    order = 1 if verbose else order
                    characteristics.append(magnitude_parser(metrics, order))
                case 'duration':
                    # order must be incremented value for duration.
                    characteristics.append(duration_parser(metrics, order))
                case 'rate_of_change':
                    order = 1 if verbose else order
                    characteristics.append(rate_of_change_parser(metrics, order))
                case 'frequency':
                    # order must be incremented value for frequency.
                    characteristics.append(frequency_parser(metrics, order))
                case 'verbose':
                    validate_verbose(order, metrics)
                    verbose = metrics
                case 'success_pattern':
                    validate_boolean(name, metrics)
                    success = metrics
                case _:
                    raise NotImplementedError(f'Characteristic {name} not found.')
            order += 1
        components.append(patterns.Component(component_name, characteristics, success))
    return components

# region Validation
def validate_timing_metrics(metrics: list[Any])-> None:
    '''Validate timing metrics.
    
    Parameters
    ----------
        metrics (list[int]): in the form...
            [start(int), end(int)]
            where start and end are first and last day of the water year 
            over which the characteristic is evaluated.
    Raises
    ------
        ValueError: if metrics are not in the correct form.    
    '''
    error_msg = f'''
                Provided timing metrics: {metrics} must be in the form:
                [start(int), end(int)].
                '''
    if len(metrics) != 2:
        raise ValueError(error_msg)
    if not all(isinstance(i, int) for i in metrics):
        raise ValueError(error_msg)

ComparisionType = Enum('ComparisionType', ['SIMPLE', 'BETWEEN'])

def validate_magnitude_metrics(metrics: list[Any]) -> ComparisionType:
    '''Validate magnitude metrics.
    
    Parameters
    ----------
        metrics (list[Any]): in the form...
            [symbol, threshold, (optional)moving_average_periods] or
            [minimum, maximum, (optional)moving_average_periods]
            where symbol is a comparision string (i.e., <, <=, etc.),
            minimum and maximum are exclusive (i.e., <, >,) boundaries for comparisons, and
            moving_average_periods is number of timesteps over which values are averaged.
    Returns
    -------
        ComparisionType: type of comparision (Simple or Inbetween).
    Raises
    ------
        ValueError: if metrics are not in the correct form.
    '''
    error_msg = f'''
                Provided metrics: {metrics} must be in the form:
                [symbol(str), threshold(Real), (optional)ma_periods(int)] or
                [minimum(Real), maximum(Real), (optional)ma_periods(int)].
                '''
    nentries = len(metrics)
    if nentries != 2 and nentries != 3:
        raise ValueError(error_msg)
    if nentries == 3:
        validate_ma_period(metrics)
    return validate_comparison_metrics(metrics)

def validate_duration_metrics(metrics: list[Any]) -> ComparisionType:
    '''Validate duration metrics.
    
    Parameters
    ----------
        metrics (list[Any]): in the form...
            [symbol, threshold] or
            [minimum, maximum]
            where symbol is a comparision string (i.e., <, <=, etc.),
            minimum and maximum are exclusive (i.e., <, >,) boundaries for comparisons.
    Returns
    -------
        ComparisionType: type of comparision (Simple or Inbetween).
    Raises
    ------
        ValueError: if metrics are not in the correct form.
    '''
    error_msg = f'''
                Provided metrics: {metrics} must be in the form:
                [symbol(str), threshold(Real)] or
                [minimum(Real), maximum(Real)].
                '''
    nentries = len(metrics)
    if nentries != 2:
        raise ValueError(error_msg)
    return validate_comparison_metrics(metrics)

def validate_rate_of_change_metrics(metrics: list[Any]) -> ComparisionType:
    '''Validate rate of change metrics.
    
    Parameters
    ----------
        metrics (list[Any]): in the form...
            [symbol, threshold, (optional)ma_periods, (optional)look_back] or
            [minimum, maximum, (optional)ma_periods, (optional)look_back]
            where symbol is a comparision string (i.e., <, <=, etc.),
            minimum and maximum are exclusive (i.e., <, >,) boundaries for comparisons,
            moving_average_periods is number of timesteps over which values are averaged, and
            look_back is number of timesteps back from current timestep to evaluate rate of change.
    Returns
    -------
        ComparisionType: type of comparision (Simple or Inbetween).
    Raises
    ------
        ValueError: if metrics are not in the correct form.
    '''
    error_msg = f'''
                Provided metrics: {metrics} must be in the form:
                [symbol(str), threshold(Real), (optional)ma_periods(int), (optional)look_back(int)] or
                [minimum(Real), maximum(Real), (optional)ma_periods(int), (optional)look_back(int)].
                '''
    nentries = len(metrics)
    if nentries < 2 or nentries > 4:
        raise ValueError(error_msg)
    if nentries > 2:
        validate_ma_period(metrics)
    if nentries > 3:
        validate_look_back(metrics)
    return validate_comparison_metrics(metrics)

def validate_frequency_metrics(metrics: list[Any]) -> ComparisionType:
    '''Validate frequency metrics.'''
    error_msg = f'''
                Provided metrics: {metrics} must be in the form:
                [symbol(str), threshold(Real), ma_period(int)] or
                [minimum(Real), maximum(Real), ma_period(int)].
                '''
    if len(metrics) != 3:
        raise ValueError(error_msg)
    validate_ma_period(metrics)
    return validate_comparison_metrics(metrics)

def validate_comparison_metrics(metrics: list[Any]) -> ComparisionType:
    '''Validate magnitude and duration comparison metrics.'''
    if isinstance(metrics[0], str):
        validate_simple_comparision_pair(metrics)
        return ComparisionType.SIMPLE
    if isinstance(metrics[0], (int, float)):
        validate_between_comparision_pair(metrics)
        return ComparisionType.BETWEEN
    raise ValueError(f'Invalid comparision metrics: {metrics}.')

def symbol_to_string(symbol: str) -> str:
    '''Convert symbol to string.'''
    return {
        '<': 'lt',
        '<=': 'le',
        '>': 'gt',
        '>=': 'ge',
        '=': 'eq',
        '!=': 'ne'
    }[symbol]

def validate_symbol(symbol: str) -> str:
    '''Validate symbol.'''
    try:
        symbol_to_string(symbol)
    except KeyError as e:
        raise NotImplementedError(f'Invalid comparision symbol: {symbol}.') from e

def validate_simple_comparision_pair(metrics: list[Any]) -> None:
    '''Validate comparision pair.'''
    validate_symbol(metrics[0])
    if not isinstance(metrics[1], (int, float)):
        raise ValueError(f'''Comparision requires a symbol, threshold value pair,
                          ({metrics[0]}, {metrics[1]}) found.''')

def validate_between_comparision_pair(metrics: list[Any]) -> None:
    '''Validate between comparision pair.'''
    if not isinstance(metrics[1], (int, float)):
        raise ValueError(f'''Between comparision requires two threshold values,
                          [{metrics[0]}, {metrics[1]}] found.''')
    if metrics[0] >= metrics[1]:
        raise ValueError(f'''Between comparsion requires two threshold values in accending order,
                         [{metrics[0]}, {metrics[1]}] found.''')

def validate_ma_period(metrics: list[Any]) -> None:
    '''Validate moving average period.'''
    error_msg = f'''
                Moving average parameter: {metrics[2]} in characteristic metrics: {metrics}
                must be an integer, representing the timesteps over which values are averaged.
                '''
    if not isinstance(metrics[2], int):
        raise ValueError(error_msg)

def validate_look_back(metrics: list[Any]) -> None:
    '''Validate look back period.'''
    error_msg = f'''
                Look back parameter: {metrics[3]} in characteristic metrics: {metrics}
                must be an integer, representing the number of timesteps back from the current timestep to evaluate rate of change.
                '''
    if not isinstance(metrics[3], int):
        raise ValueError(error_msg)

def validate_verbose(order: int, metrics: Any) -> None:
    '''Validate verbose.'''
    warning_msg = f'''
                "verbose = {metrics}" appeared after {order} component characteristics.
                First {order} characteristics evaluated as "verbose = True".
                '''
    validate_boolean('verbose', metrics)
    if metrics and order != 1:
        print(warning_msg)

def validate_boolean(name: str, metrics: Any) -> None:
    '''Validate boolean.'''
    if not isinstance(metrics, bool):
        raise ValueError(f'Boolean value expected for {name}, {metrics} found.')
# endregion

# region Parsers
def between_parser(metrics: list[Any], inclusive=True) -> patterns.comparison_fx:
    '''Generates comparision function for between metrics (i.e., [minimum, maximum]).'''
    if len(metrics) != 2 or not all(isinstance(i, (int, float)) for i in metrics):
        raise ValueError('Between metrics must have two numeric values.')
    if metrics[0] >= metrics[1]:
        raise ValueError('Between metrics must have values in ascending order.')
    if inclusive:
        return patterns.comparison_fx('<=', metrics[0], '<=', metrics[1])
    else:
        return patterns.comparison_fx('<', metrics[0], '<', metrics[1])

def timing_parser(metrics: list[Any], order: int) -> tuple[str, Any]:
    '''Parse timing metrics.
    
    Parameters
    ----------
        metrics (list[int]): in the form...
            [start(int), end(int)]
            where start and end are first and last day of the water year
            over which the characteristic is evaluated.
            Note: start and end are inclusive.
        order (int): Position in which characteristic is evaluated.
    Returns
    -------
        tuple[str, Any]: characteristic name and function.
    Raises
    ------
        ValueError: if metrics are not in the correct form.
    '''
    validate_timing_metrics(metrics)
    return patterns.Characteristic(
        name=f'{patterns.CharacteristicType.TIMING.name.lower()}_{metrics[0]}-{metrics[1]}',
        fx=patterns.timing_fx(between_parser(metrics[0:2]), order),
        type=patterns.CharacteristicType.TIMING
    )

def magnitude_parser(metrics: list[Any], order: int) -> tuple[str, Any]:
    '''Parse magnitude metrics.
    
    Parameters
    ----------
        metrics (list[Any]): in the form...
            [symbol, threshold, (optional)moving_average_periods] or
            [minimum, maximum, (optional)moving_average_periods]
            where symbol is a comparision string (i.e., <, <=, etc.),
            minimum and maximum are exclusive (i.e., <, >,) boundaries for comparisons, and
            moving_average_periods is number of timesteps over which values are averaged.
        order (int): Position in which characteristic is evaluated.
    Returns
    -------
        tuple[str, Any]: characteristic name and function.
    Raises
    ------
        ValueError: if metrics are not in the correct form.
    '''
    label = patterns.CharacteristicType.MAGNITUDE.name.lower()
    comparision_type = validate_magnitude_metrics(metrics)
    ma_periods = metrics[2] if len(metrics) == 3 else 1
    match comparision_type:
        case ComparisionType.SIMPLE:
            name=f'{label}_{symbol_to_string(metrics[0])}{metrics[1]}'
            comparison_fx=patterns.comparison_fx(metrics[0], metrics[1])
        case ComparisionType.BETWEEN:
            name=f'{label}_{metrics[0]}-{metrics[1]}'
            comparison_fx=between_parser(metrics[0:2], inclusive=False)
        case _:
            raise NotImplementedError('Invalid comparision type.')
    return patterns.Characteristic(
        name=name,
        fx=patterns.magnitude_fx(comparison_fx, order, ma_periods),
        type=patterns.CharacteristicType.MAGNITUDE
    )

def duration_parser(metrics: list[Any], order: int) -> tuple[str, Any]:
    '''Parse duration metrics.
    
    Parameters
    ----------
        metrics (list[Any]): in the form...
            [symbol, threshold] or
            [minimum, maximum]
            where symbol is a comparision string (i.e., <, <=, etc.),
            minimum and maximum are exclusive (i.e., <, >,) boundaries for comparisons, and
        order (int): Position in which characteristic is evaluated.
    Returns
    -------
        tuple[str, Any]: characteristic name and function.
    Raises
    ------
        ValueError: if metrics are not in the correct form.
    '''
    label = patterns.CharacteristicType.DURATION.name.lower()
    comparision_type = validate_duration_metrics(metrics)
    match comparision_type:
        case ComparisionType.SIMPLE:
            name=f'{label}_{symbol_to_string(metrics[0])}{metrics[1]}'
            comparison_fx=patterns.comparison_fx(metrics[0], metrics[1])
        case ComparisionType.BETWEEN:
            name=f'{label}_{metrics[0]}-{metrics[1]}'
            comparison_fx=patterns.comparison_fx('<', metrics[0], '>', metrics[1])
        case _:
            raise NotImplementedError('Invalid comparision type.')
    return patterns.Characteristic(
        name=name,
        fx=patterns.duration_fx(comparison_fx, order),
        type=patterns.CharacteristicType.DURATION
    )

def rate_of_change_parser(metrics: list[Any], order: int) -> tuple[str, Any]:
    '''Parse rate of change metrics.
    
    Parameters
    ----------
        metrics (list[Any]): in the form...
            [symbol, threshold, (optional)ma_periods, (optional)look_back, (optional)min] or
            [minimum, maximum, (optional)ma_periods, (optional)look_back, (optional)min]
            where symbol is a comparision string (i.e., <, <=, etc.),
            minimum and maximum are exclusive (i.e., <, >,) boundaries for comparisons, and
            ma_periods number of timesteps over which values are averaged.
                Defaults to 1. Must be 3rd parameter.
            look_back number of timesteps back from current timestep to evaluate rate of change.
                Defaults to 1. Must be 4th parameter.
            min is the minimum value hydrologic value is compared to.
                Defaults to 0. Must be 5th parameter.
        order (int): Position in which characteristic is evaluated.
    Returns
    -------
        tuple[str, Any]: characteristic name and function.
    Raises
    ------
        ValueError: if metrics are not in the correct form.
    Notes
    -----
        The order of the optional parameters is important.
            ma_period is always assumed to be the third parameter.
            look_back is always assumed to be the fourth parameter.
            min is always assumed to be the fifth parameter.
    '''
    label = patterns.CharacteristicType.RATE_OF_CHANGE.name.lower()
    comparision_type = validate_rate_of_change_metrics(metrics)
    ma_periods = metrics[2] if len(metrics) > 2 else 1
    look_back = metrics[3] if len(metrics) > 3 else 1
    min_val = metrics[4] if len(metrics) > 4 else 0
    match comparision_type:
        case ComparisionType.SIMPLE:
            name=f'{label}_{symbol_to_string(metrics[0])}{metrics[1]}'
            comparison_fx=patterns.comparison_fx(metrics[0], metrics[1])
        case ComparisionType.BETWEEN:
            name=f'{label}_{metrics[0]}-{metrics[1]}'
            comparison_fx=between_parser(metrics[0:2], inclusive=False)
        case _:
            raise NotImplementedError('Invalid comparision type.')
    return patterns.Characteristic(
        name=name,
        fx=patterns.rate_of_change_fx(comparison_fx, order, ma_periods, look_back, min_val),
        type=patterns.CharacteristicType.RATE_OF_CHANGE
    )

def frequency_parser(metrics: list[Any], order: int) -> tuple[str, Any]:
    '''Parse frequency metrics.
    
    Parameters
    ----------
        metrics (list[Any]): in the form...
            [symbol, threshold, ma_period] or
            [minimum, maximum, ma_period]
            where symbol is a comparision string (i.e., <, <=, etc.),
            minimum and maximum are exclusive (i.e., <, >,) boundaries for comparisons, and
            ma_period specifies that the comparison condition must be met over each ma_period
            number of years.
        order (int): Position in which characteristic is evaluated.
    Returns
    -------
        tuple[str, Any]: characteristic name and function.
    Raises
    ------
        ValueError: if metrics are not in the correct form.
    '''
    label = patterns.CharacteristicType.FREQUENCY.name.lower()
    comparision_type = validate_frequency_metrics(metrics)
    match comparision_type:
        case ComparisionType.SIMPLE:
            name=f'{label}_{symbol_to_string(metrics[0])}{metrics[1]}in{metrics[2]}yrs'
            comparison_fx=patterns.comparison_fx(metrics[0], metrics[1])
        case ComparisionType.BETWEEN:
            name=f'{label}_{metrics[0]}-{metrics[1]}in{metrics[2]}yrs'
            comparison_fx=between_parser(metrics[0:2], inclusive=False)
        case _:
            raise NotImplementedError('Invalid comparision type.')
    return patterns.Characteristic(
        name=name,
        fx=patterns.frequency_fx(comparison_fx, order, metrics[2]),
        type=patterns.CharacteristicType.FREQUENCY
    )
# endregion

if __name__ == '__main__':
    app(prog_name=__app_name__ + ' v' + __version__)
