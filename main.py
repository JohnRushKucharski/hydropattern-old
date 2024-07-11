'''Entry point for CLI application.'''
import typer

from src.timeseries import Timeseries

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
@app.command()
def load_timeseries(path: str = typer.Argument(..., help='''
                                               Path to *.csv file containing data to plot.
                                               See README.md @''')):
    '''Load timeseries data from *.csv file.'''
    print('dummy command.')


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
    ts = load_data(path, first_day_of_water_year, date_format)
    ts.plot_timeseries(data_columns=columns)

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

def load_data(path: str, first_day_of_water_year: int, date_format: str) -> None:
    '''Load data.'''
    kwargs = {'date_format': date_format} if date_format else {}
    return Timeseries.from_csv(path, first_day_of_water_year, **kwargs)

if __name__ == '__main__':
    app(prog_name=__app_name__ + ' v' + __version__)
