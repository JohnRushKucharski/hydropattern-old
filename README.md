# hydropattern
Finds natural flow regimes type patterns in time series data.

## Background
Natural flow regimes are widely used in water resources management. Learn more about natural flow regimes:
> Poff, N. L., Allan, J. D., Bain, M. B., Karr, J. R., Prestegaard, K. L., Richter, B. D., Sparks, R. E., & Stromberg, J. C. (1997). The Natural Flow Regime. BioScience, 47(11), 769–784. https://doi.org/10.2307/1313099

The repository tends to use functional flows terminology. Functional flows are natural flow regimes linked to specific environmental processes. Learn more about functional flows:
> Yarnell, S. M., Stein, E. D., Webb, J. A., Grantham, T., Lusardi, R. A., Zimmerman, J., Peek, R. A., Lane, B. A., Howard, J., & Sandoval-Solis, S. (2020). A functional flows approach to selecting ecologically relevant flow metrics for environmental flow applications. River Research and Applications, 36(2), 318-324. https://doi.org/10.1002/rra.3575
> Note: Figure 2 and Table 2 are particularly helpful for understanding the natural flow regimes this program tracks.

Natural flow regimes can be adapted to classify hydrologic regimes in non-riverine environments, like lakes. They can be used to evaluate the alteration of natural hydrologic patterns. This program imagines their usage in climate impact studies.

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

## Basic Usage
The following inputs are required to parameterize the program:

1. Hydrologic time series as a .csv file in the following format:

dates | flows
--- | ---
datetime_t | flow_t
datetime_t+1 | flow_t+1
... | ...
datetime_T | flow_T
   
2. TOML configuration file used to define natural flow regime *components* (and associated *characteristics* and *metrics*).
3. Output file path.
