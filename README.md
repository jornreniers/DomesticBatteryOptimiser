# Domestic Battery Optimiser

The goal of this repository is to forecast domestic electricity consumption based on historic usage data and weather data, and then schedule the charging and discharging of a battery to reduce electricity costs. It is still a work in progress.

## Tooling & Getting started
The project is setup using UV, and the .toml file contains the packages and their version numbers. It needs to be installed from [their website](https://docs.astral.sh/uv/getting-started/) if you have not done so. 

To run the code, open a terminal and navigate to the project folder.

- `uv sync`: this will install all the required packages into your venv. It is needed only once.
- `uv run python main.py`: this will run the code.

The project was written in VS code on Windows, with two main VS code extensions from Astral: `Ruff` is used for formatting, and the dev version of `ty` for type checking.

Note that you need to manually add your electricity consumption data in the appropriate folder as explained below.

## Input data formats

Formats of input data are defined in the DataConfig-class. 

### Electricity consumption data
All the code really needs is a column with electricity consumption (in kWh) and a column with the corresponding time stamp. The data is assumed to be in the folder `data/electricity/raw_data` but this can be changed via the data configuration (`config/DataConfigcpy`).

The present configuration is for electricity consumption data as you can download from Octopus Energy. There is one file per year with half-hourly consumption data.
The columns are
- Consumption (kwh): consumed electricity in kWhy
- Estimated Cost Inc. Tax (p): expected cost in pence
- Start: start of the half-hour
- End: end of the half-hour

### Weather data
We pull the weather data comes from the weather station in [Oxford Physics](https://eodg.atm.ox.ac.uk/eodg/weather/weather.html). If you live far from Oxford, it is recommended to find another source of historic weather data. All the code needs is a column with outside temperature and a column with the time stamp. For optimal performance, the time grandularity should be equal to, or greater than, that of the electricity data.

The [Data Repository](https://weatherstationdata.physics.ox.ac.uk/) has data sorted per year. There is one file per day.

The files are downloaded with a script that is run once per day and pulls the latest data. The first time you run this code, it will download quite a bit of historic data so it might take a while. The details of each file are in the detailed_documentation_physics.txt which is copied from the readme of the data repository. The most useful columns are in the file MPYYYYMMDD.csv
- A: date (DD/MM/YY) in UTC
- B: hour (HH:MM:SS) in UTC
- E: dry bulb temperature (Celcius)
- I: relative humidity (percentage) (not currently used by the code)
Values of -999 indicate missing data. The day and time are in UTC times as confirmed by their [script](https://gitlab.physics.ox.ac.uk/povey/aopp_weather_station/-/blob/master/weather/instromet.py?ref_type=heads)

## Demand fitting and forecasting

A gaussian process is trained to forecast electricity consumption. Possible features are:

- local time of day (categorical)
- day of the week (categorical)
- weekday vs weekend (categorical)
- month (categorical)
- outside temperature
- daily mininimum, average or maximum temperature
- daily minimumum temperature ceiled to 0, 5, 10 or 15 degrees (useful to capture nonlinearities due to electric heating)
- daily maximum temperature floored to 15, 20, or 25 (useful capture nonlinearities due to aiconditioning)

Users can first manually select which features to consider (in InternalConfig) if desired. In either case, there is a three-stage feature-selection process to reduce the number of features. Firstly we select the features which best correlate with the consumption (using KBest), which is useful to remove many of the categorical-expanded-to-dummy variables. Then we remove features which are highly correlated with each other. In the final step, we iteratively remove the least useful feature with cross validation. Several figures are made where users can check the different features, their correlation etc.

The regression is a fairly straightforward Guassian process. There is some tuning of the noise-assumption, but this also depends on human judgement since a tigher fit (smaller MAPE) will result in larger noise assumption and hence larger uncertainty. Again, a large number of figures are made covering both training and validation periods for users to analyse the result.