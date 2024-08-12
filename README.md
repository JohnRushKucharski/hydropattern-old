# hydropattern
Finds natural flow regimes type patterns in time series data.

## Background
Natural flow regimes are widely used in water resources management. Learn more about natural flow regimes:
> Poff, N. L., Allan, J. D., Bain, M. B., Karr, J. R., Prestegaard, K. L., Richter, B. D., Sparks, R. E., & Stromberg, J. C. (1997). The Natural Flow Regime. BioScience, 47(11), 769–784. https://doi.org/10.2307/1313099

The repository tends to use functional flows terminology. Functional flows are natural flow regimes linked to specific environmental processes. Learn more about functional flows:
> Yarnell, S. M., Stein, E. D., Webb, J. A., Grantham, T., Lusardi, R. A., Zimmerman, J., Peek, R. A., Lane, B. A., Howard, J., & Sandoval-Solis, S. (2020). A functional flows approach to selecting ecologically relevant flow metrics for environmental flow applications. River Research and Applications, 36(2), 318-324. https://doi.org/10.1002/rra.3575

> Note: Figure 2 and Table 2 are particularly helpful for understanding the natural flow regimes this program tracks.

Natural flow regimes can be adapted to classify hydrologic regimes in non-riverine environments, like static water levels in lakes. They can be used to evaluate the alteration of natural hydrologic patterns. This program imagines their usage in climate impact studies.

## Basic Terminology
To define a natural flow regime the following hierarchical labels must be defined:

**Component:** Natural flow regimes consist of one or more *components*.

**Characteristic:** Each component consists of one or more of the following *characteristics*.

- Timing: when the hydrologic pattern occurs (i.e., wet season).
- Magnitude: the size hydrologic pattern (i.e., flow, stage, etc.).
- Duration: how long the hydrologic pattern persists (i.e., 7 days).
- Frequency: how often the pattern occurs (i.e. in 1 out of every 5 years).
- Rate of Change: change in the size of the hydrologic pattern (i.e., doubling of the previous day's flow).

**Metric:** A metric defines the truth value for each characteristic. For example, the magnitude of flow > 100.

Examples are provided below.

## Inputs
The following inputs are required to parameterize the program:

1. One or more hydrologic time series provided in a .csv file. This file must have the following format:

time    | column_0      | column_1  | ... | column_n-1  | column_n      |
---     | ---           | ---       | --- | ---         | ---           | 
t_0     | value_0,0     | value_1,0 | ... | value_n-1,0 | value_n,0     |
t_1     | value_0,1     | ...       | ... | ...         | value_n,1     |
...     | ...           | ...       | ... | ...         | ...           |         
t_m-1   | value_0,m-1   | ...       | ... | ...         | value_n,m-1   |
t_m     | value_0,m     | value_1,m | ... | value_n-1,m | value_n,m     |

where the 'time' column contains a datetimestring that can be parsed as a pandas datetime index. By default pandas will, with a warning message and possible error, attempt to guess format of this string. The --format flag can be used to provide an explicit date format code, see: https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior. This column must contain the 'time' header. The n columns of hydrologic data must also contain headers, but no naming convention is enforced.

2. Configuration data provided in a .toml file (see: https://toml.io/en/). This file is used to define natural flow regime *components* (and their associated *characteristics* and *metrics*). A detailed example configuration file is available here: https://github.com/JohnRushKucharski/hydropattern/tree/main/examples/ex_config.toml. 

## Basic Usage


## Todo List
- [ ] Not in range comparision (i.e. opposite of between comparisions).
- [ ] Two or more of same characteristic types (i.e., timing =[1, 100], timing=[300, 365] component must occur either between days 1-100, or 300-365 of water year). Since these are held in as a dictionary in the Component.characteristics property, a name or naming convention would be required.
