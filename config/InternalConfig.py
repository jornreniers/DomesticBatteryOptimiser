class InternalConfig:
    # Settings
    data_cleaning_max_timestep_minutes = 60.0  # consider data is missing if the time interval between two datapoints exceeds this value
    plot_level = 3  # 0 none, 1 data and results, 2 intermediate steps, 3 details
    plot_folder = "Results"

    # Features and regression settings
    min_number_of_features_dayFit = 10
    min_number_of_features_fullTimeFit = 25
    max_autocorrelation_threshold = 0.8
    try_all_fitting = False
    training_days = 365  # number of days used for training vs validation

    # Final/processed column names
    colname_time = "Timestamp"
    colname_time_of_day = "TimeOfDay"
    colname_consumption_kwh = "Consumption (kwh)"
    colname_temperature_dry = "Temperature"
    colname_temperature_wet = "Temperature_wet"
    colname_humidity = "Humidity"

    # Features
    colname_date = "Date"
    colname_day_of_week = "Day_of_week"
    colname_period_index = "Half_hour_index"
    colname_month = "Month"
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

    # List with all features
    # continuous features
    features_continuous = [
        colname_temperature_dry,  # for full time-resolution only
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
    # categorical features
    features_categorical = [
        colname_period_index,  # integer eg 0-47, for full time-resolution only
        colname_day_of_week,  # integer 1-7
        colname_month,  # integer 1-12
    ]

    all_features = features_continuous + features_categorical

    # values that will be completed by the code
    average_time_step = 0
