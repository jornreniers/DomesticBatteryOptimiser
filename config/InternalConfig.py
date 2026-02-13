class InternalConfig:
    # Settings
    data_cleaning_max_timestep_minutes = 60.0  # consider data is missing if the time interval between two datapoints exceeds this value
    plot_level = 2  # 0 none, 1 data and results, 2 intermediate steps, 3 details
    plot_folder = "Results"

    # Features and regression settings
    # specify the number of features (as fraction of the total number of features)
    # you want to keep after the two stages of feature selection
    # the value for xx_rfecv must be <= the value of _kbest since it is applied after
    # Note that we ask for a fraction because the actual number depends strongly
    # on which categorical features were included since they are exploded into dummies.
    daily_fraction_of_features_to_keep_kbest = (
        0.8  # use kbest to select up to 20 features
    )
    daily_fraction_of_features_to_keep_rfecv = 0.5  # use RFECV to select up to 10 features, must be <= kbest since it is called afterward
    fullResolution_fraction_of_features_to_keep_kbest = 0.25  # set to None for tuning
    fullResolution_fraction_of_features_to_keep_rfecv = 0.25  # set to None for tuning
    max_autocorrelation_threshold = 0.8  # autocorrelation between features that we allow, above this value, features are removed
    # Noise assumptions for the guassian process to fit consumption
    # provided as a power of 10, eg -1 means we assume the error in the value is 10^-1, ie 0.1 kWh
    # this value is the variance of the random variation in daily consumption
    # ie the standard deviation is sqrt(10^lognoise)
    # If you provide a range, we do a grid search to find the best value
    # If min and max are the same value, we use that value
    lognoise_minimum = -1.66  # -0.1  # alpha in gaussian process is 10^(lognoise_minimum) <= alpha <= 10^(lognoise_maximim)
    lognoise_maximim = 0 - 1.66  # -0.1
    # number of days used for training vs validation
    training_days = 365

    # Column names
    colname_time = "Timestamp"
    colname_time_of_day = "TimeOfDay"
    colname_consumption_kwh = "Consumption (kwh)"
    colname_temperature_dry = "Temperature"
    colname_temperature_wet = "Temperature_wet"
    colname_humidity = "Humidity"
    colname_date = "Date"
    colname_day_of_week = "Day_of_week"
    colname_weekend = "Weekend"
    colname_period_index = "Half_hour_index"
    colname_month = "Month"
    colname_daily_consumption = "Total_daily_consumption"
    colname_daily_min_temperature = "Day_min_temperature"
    colname_daily_avg_temperature = "Day_mean_temperature"
    colname_daily_max_temperature = "Day_max_temperature"
    colname_daily_temperature_below_zero = "Day_min_temperature_below_zero"
    colname_daily_temperature_below_five = "Day_min_temperature_below_five"
    colname_daily_temperature_below_ten = "Day_min_temperature_below_ten"
    colname_daily_temperature_below_fifteen = "Day_min_temperature_below_fifteen"
    colname_daily_temperature_above_fifteen = "Day_max_temperature_above_fifteen"
    colname_daily_temperature_above_twenty = "Day_max_temperature_above_twenty"
    colname_daily_temperature_above_twentyfive = "Day_max_temperature_above_twentyfive"
    colname_training_data = "training_row_filter"
    colname_ydata = "y_data"
    colname_yfit = "y_fitted"
    colname_ystd_total = "y_std_total"  # total uncertainty, measurement noise & GP

    # Feature selection
    # This is a first-step manual selection to speed up computation
    # The results are based on the graphs made the feature-selection scripts.
    # If you set the values to None, we do a search. Otherwise we use the values provided
    features_daily_forecast = [
        colname_daily_min_temperature,
        colname_daily_avg_temperature,
        colname_daily_max_temperature,
        colname_daily_temperature_below_zero,
        colname_daily_temperature_below_five,
        colname_daily_temperature_below_ten,
        colname_daily_temperature_below_fifteen,
        colname_daily_temperature_above_fifteen,
        colname_daily_temperature_above_twenty,
        colname_daily_temperature_above_twentyfive,
        colname_month,  # integer 1-12
        colname_weekend,  # boolean true or false
    ]
    # see full_resolution_data_analyser.py, we keep
    #   min temperature (or T below/above to approximate bucketed values)
    #   weekday vs weekend
    #   time of day
    features_fullResolution_forecast = [  # set to None for tuning
        colname_temperature_dry,
        colname_daily_min_temperature,
        colname_daily_temperature_below_zero,
        colname_daily_temperature_below_five,
        colname_daily_temperature_below_ten,
        colname_daily_temperature_below_fifteen,
        colname_daily_temperature_above_fifteen,
        colname_daily_temperature_above_twenty,
        colname_daily_temperature_above_twentyfive,
        colname_weekend,  # boolean true or false
        colname_period_index,  # integer eg 0-47
    ]

    # Do not change value below this
    # continuous features (rescaled to mean 0 and std 1)
    features_continuous = [
        colname_temperature_dry,
        colname_daily_consumption,
        colname_daily_min_temperature,
        colname_daily_avg_temperature,
        colname_daily_max_temperature,
        colname_daily_temperature_below_zero,
        colname_daily_temperature_below_five,
        colname_daily_temperature_below_ten,
        colname_daily_temperature_below_fifteen,
        colname_daily_temperature_above_fifteen,
        colname_daily_temperature_above_twenty,
        colname_daily_temperature_above_twentyfive,
    ]
    # categorical features (expanded to dummies)
    features_categorical = [
        colname_period_index,  # integer eg 0-47
        colname_day_of_week,  # integer 1-7
        colname_month,  # integer 1-12
        colname_weekend,  # boolean true or false
    ]

    # values that will be completed by the code
    average_time_step = 0
