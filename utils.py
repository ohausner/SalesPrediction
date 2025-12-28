# Utils

def rmsle_manual(y_true, y_pred):
    """
    Calculates the Root Mean Squared Logarithmic Error (RMSLE).

    Args:
        y_true (pd.Series or np.array): Actual target values.
        y_pred (pd.Series or np.array): Predicted values.

    Returns:
        float: The RMSLE score.
    """
    # Ensure all values are non-negative, as log is used
    if (y_true < 0).any() or (y_pred < 0).any():
        raise ValueError("RMSLE cannot be used when targets contain negative values.")
        
    # Calculate squared logarithmic errors
    squared_log_errors = np.power(np.log(y_pred + 1) - np.log(y_true + 1), 2)
    
    # Calculate the mean of the squared logarithmic errors and take the square root
    rmsle_score = np.sqrt(np.mean(squared_log_errors))
    
    return rmsle_score
def specific_df(df, store, family):
    return df[(df['store_nbr'] == store) & (df['family'] == family)]
# 1. Barplot Function
def plot_categorical_bar(df, category_col, target_col='sales', agg_type='mean', top_k=None, tilt_angle=0):
    """
    Generates a bar plot with value labels.
    - If 'mean': shows the mean with Standard Deviation error bars.
    - If 'sum': shows the total sum (no error bars).
    """
    plt.figure(figsize=(10, 6))
    
    # Define estimator and error bar settings based on user input
    if agg_type == 'mean':
        estimator_func = 'mean'
        error_bar_setting = 'sd' # Standard Deviation
        title_text = f"Mean of {target_col} by {category_col} (with Std Dev)"
    elif agg_type == 'sum':
        estimator_func = 'sum'
        error_bar_setting = None # No error bars for simple sums
        title_text = f"Sum of {target_col} by {category_col}"
    else:
        raise ValueError("agg_type must be either 'mean' or 'sum'")

    order_df = df.groupby(category_col)[target_col].agg(estimator_func).reset_index()
    order_df = order_df.sort_values(target_col, ascending=False)

    if top_k:
            order_df = order_df.head(top_k)

    sort_order = order_df[category_col].tolist()

    # Create the plot
    ax = sns.barplot(
        data=df, 
        x=category_col, 
        y=target_col, 
        estimator=estimator_func, 
        # errorbar=error_bar_setting,
        capsize=0.1, # Adds little caps to the error bars
        palette='viridis',
        order=sort_order
    )

    # Add value labels on top of bars
    for container in ax.containers:
        # We assume the first container holds the bars. 
        # Note: If error bars exist, they are distinct artists, handled automatically by bar_label in newer mpl versions
        ax.bar_label(container, fmt='%.2f', padding=3)

    plt.xticks(rotation=tilt_angle)
    plt.title(title_text)
    plt.ylabel(f"{target_col} ({agg_type})")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

# 2. Statistical Test Function
def run_stat_test(df, category_col, target_col='sales'):
    """
    Analyzes the relationship between a categorical and numeric variable.
    - 2 Categories: Independent T-Test
    - >2 Categories: One-way ANOVA
    """
    # Drop NaNs for the calculation
    clean_df = df[[category_col, target_col]].dropna()
    
    # Get unique categories
    groups = clean_df[category_col].unique()
    
    # Create a list of arrays (one for each category)
    group_data = [clean_df[clean_df[category_col] == g][target_col] for g in groups]
    
    print(f"--- Statistical Test: {target_col} by {category_col} ---")
    
    if len(groups) == 2:
        print(f"Detected 2 groups: {groups}. Running T-Test.")
        stat, p_val = stats.ttest_ind(group_data[0], group_data[1])
        test_name = "T-Test"
    elif len(groups) > 2:
        print(f"Detected {len(groups)} groups. Running One-way ANOVA.")
        stat, p_val = stats.f_oneway(*group_data)
        test_name = "ANOVA"
    else:
        print("Error: Need at least 2 categories to perform a statistical test.")
        return None

    # Output results
    print(f"{test_name} Statistic: {stat:.4f}")
    print(f"P-Value: {p_val:.4e}")
    
    if p_val < 0.05:
        print("Result: Significant difference (p < 0.05)")
    else:
        print("Result: No significant difference (p >= 0.05)")
        
    return stat, p_val

# 3. Scatter Plot Function
def plot_scatter(df, x_col, y_col, heat_col=None):
    plt.figure(figsize=(10, 6))

    plt.scatter(df[x_col], df[y_col], c=df[heat_col] if heat_col else None, cmap='viridis')
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    # Move legend outside if crowded
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()

def density_plot(df, y_col):
    mean = df[y_col].mean()
    plt.figure(figsize=(10, 6))
    sns.kdeplot(df[y_col], shade=True)
    plt.xlabel(y_col)
    plt.ylabel("Density")
    plt.axvline(mean, color='red', linestyle='--', label=f"Mean: {mean:.2f}")
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.title(f"Density Plot: {y_col}")
    plt.tight_layout()
    plt.show()

def boxplot(df, y_col):
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=df[y_col])
    plt.xlabel(y_col)
    plt.ylabel("Count")
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.title(f"Boxplot: {y_col}")
    plt.tight_layout()
    plt.show()
# 4. Timeline Plot Function
def plot_timeline(df, time_col, target_col, category_col=None, rolling_window=0):
    """
    Plots a target variable over time.
    - If category_col is provided: plots multiple lines (one per category).
    - Automatically sorts by time_col to prevent "zigzag" lines.
    """
    plt.figure(figsize=(12, 6))
    
    # Sort data by time to ensure the line flows correctly from left to right
    df_sorted = df.sort_values(by=time_col)
    
    if rolling_window > 0:
        df_sorted[target_col] = df_sorted[target_col].rolling(rolling_window).mean()

    sns.lineplot(
        data=df_sorted,
        x=time_col,
        y=target_col,
        hue=category_col,       # Optional category split
        # marker='o',             # Adds dots to data points for visibility
        palette='tab10' if category_col else None
    )
    
    title = f"Timeline: {target_col} over {time_col}"
    if category_col:
        title += f" (grouped by {category_col})"
        
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

# 5. Time-Series Decomposition Function
def plot_ts_decomposition(df, time_col, target_col, period=7, model='additive'):
    """
    Decomposes a time series into Trend, Seasonality, and Residuals.
    
    Parameters:
    - period: The frequency of the data (e.g., 12 for monthly, 7 for daily, 4 for quarterly).
    - model: 'additive' or 'multiplicative'. 
      Use 'additive' if the magnitude of seasonality is constant.
      Use 'multiplicative' if seasonality increases as the trend increases.
    """
    # Prepare data: Set time as index and sort
    # We create a copy to avoid modifying the original dataframe outside the function
    ts_data = df.copy()
    
    # Ensure time_col is datetime objects (optional but recommended for real dates)
    # If your input is just numbers (1, 2, 3...), this step is skipped automatically.
    if pd.api.types.is_string_dtype(ts_data[time_col]):
         ts_data[time_col] = pd.to_datetime(ts_data[time_col])
         
    ts_data = ts_data.set_index(time_col).sort_index()
    
    # Check for missing values which break decomposition
    if ts_data[target_col].isnull().any():
        print("Warning: Missing values detected. Interpolating linearly.")
        ts_data[target_col] = ts_data[target_col].interpolate(method='linear')

    # Perform Decomposition
    print(f"Running decomposition (Period={period}, Model={model})...")
    decomposition = seasonal_decompose(ts_data[target_col], model=model, period=period)
    
    # Plotting
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    
    decomposition.observed.plot(ax=ax1, title='Observed (Original Data)', color='black')
    decomposition.trend.plot(ax=ax2, title='Trend', color='blue')
    decomposition.seasonal.plot(ax=ax3, title='Seasonality', color='green')
    decomposition.resid.plot(ax=ax4, title='Residuals (Noise)', color='red', marker='o', linestyle='None')
    
    plt.tight_layout()
    plt.show()

def aggregate_for_time_series(df, time_col, target_col, method='mean'):
    """
    Groups data by time_col and aggregates the target_col.
    method: 'mean' (for averages) or 'sum' (for totals like total revenue).
    """
    if method == 'mean':
        # Useful for things like 'Average Price' or 'Temperature'
        df_agg = df.groupby(time_col)[target_col].mean().reset_index()
    elif method == 'sum':
        # Useful for 'Total Sales' or 'Total Clicks'
        df_agg = df.groupby(time_col)[target_col].sum().reset_index()
    
    return df_agg


def fill_missing_dates(df, date_col, category_cols, target_col='sales', fill_value=0):
    """
    Ensures the dataframe has a continuous timeline for every group.
    - category_cols: List of columns defining the groups (e.g. ['Store', 'Category'])
    """
    # 1. Convert to datetime if needed
    df[date_col] = pd.to_datetime(df[date_col])
    
    # 2. Get the full range of dates (Min date to Max date)
    # This automatically includes Dec 12th and any other gaps
    full_dates = pd.date_range(start=df[date_col].min(), 
                               end=df[date_col].max(), 
                               freq='D')
    
    print(f"Filling missing dates from {full_dates.min().date()} to {full_dates.max().date()}")
    
    # 3. Get unique values for your groups (Store, Category, etc.)
    unique_groups = [df[c].unique() for c in category_cols]
    
    # 4. Create the "Master Index" (Cartesian Product)
    # This creates every possible combination of Date + Store + Category
    # structure: [Dates] x [Stores] x [Categories]
    iterables = [full_dates] + unique_groups
    multi_index = pd.MultiIndex.from_product(iterables, names=[date_col] + category_cols)
    
    dates_missing = full_dates.difference(df[date_col].unique())
    print(f"Missing dates to be filled: {dates_missing.date.tolist()}")
    # 5. Reindex the dataframe
    # We set the index to match our Master Index, then reindex.
    # 'fill_value=0' puts a 0 where data was missing (e.g., Dec 12)
    df_filled = df.set_index([date_col] + category_cols)\
                  .reindex(multi_index)\
                  .reset_index()
    df_filled[target_col] = df_filled[target_col].fillna(fill_value)
    
    return df_filled

# 1. Feature Engineering Function (Your Logic Encapsulated)
def engineer_features(df_in):
    """
    Applies the specific rolling/lag logic provided by the user.
    """
    df = df_in.copy()
    
    # --- A. Family Level Features ---
    # Note: We must maintain the index or merge carefully. 
    # The safest way in a loop is to calculate on the full temp DF and return columns.
    
    # 1. Rolling mean of sales per family
    # (We group by date+family first to get daily family totals, then calc lags)
    family_daily = df.groupby(['date', 'family'])['sales'].mean().reset_index()
    
    # Calculate the LAGS (Shift 1)
    family_daily['sales_last_day_family'] = family_daily.groupby('family')['sales'].transform(lambda x: x.shift(1))
    
    # Now calculate rolling on the LAGGED value (to avoid leakage)
    # Note: Your original logic used transform on 'sales_last_day_family', which is correct for recursion
    grouper = family_daily.groupby('family')['sales_last_day_family']
    
    family_daily['rolling_mean_family_7'] = grouper.transform(lambda x: x.rolling(7).mean())
    family_daily['rolling_mean_family_30'] = grouper.transform(lambda x: x.rolling(30).mean())
    family_daily['sales_week_ago_family'] = grouper.transform(lambda x: x.shift(7))
    family_daily['sales_month_ago_family'] = grouper.transform(lambda x: x.shift(30))
    
    # Ratios / Slopes (Handle division by zero if necessary, usually np.inf or NaN)
    family_daily['weekly_change_family'] = family_daily['sales_last_day_family'] / family_daily['sales_week_ago_family']
    family_daily['weekly_slope_family'] = (family_daily['sales_last_day_family'] - family_daily['sales_week_ago_family']) / 7
    family_daily['monthly_slope_family'] = (family_daily['sales_last_day_family'] - family_daily['sales_month_ago_family']) / 30
    
    # Merge back to main df
    df = df.merge(family_daily.drop(columns=['sales']), on=['date', 'family'], how='left')

    # --- B. Store-Family Level Features ---
    store_family_daily = df.groupby(['date', 'store_nbr', 'family'])['sales'].mean().reset_index()
    store_family_daily['sales_last_day_store_family'] = store_family_daily.groupby(['store_nbr', 'family'])['sales'].transform(lambda x: x.shift(1))
    
    grouper_sf = store_family_daily.groupby(['store_nbr', 'family'])['sales_last_day_store_family']
    
    store_family_daily['rolling_mean_store_family_7'] = grouper_sf.transform(lambda x: x.rolling(7).mean())
    store_family_daily['rolling_mean_store_family_30'] = grouper_sf.transform(lambda x: x.rolling(30).mean())
    store_family_daily['sales_week_ago_store_family'] = grouper_sf.transform(lambda x: x.shift(7))
    store_family_daily['sales_month_ago_store_family'] = grouper_sf.transform(lambda x: x.shift(30))
    
    store_family_daily['weekly_change_store_family'] = store_family_daily['sales_last_day_store_family'] / store_family_daily['sales_week_ago_store_family']
    store_family_daily['weekly_slope_store_family'] = (store_family_daily['sales_last_day_store_family'] - store_family_daily['sales_week_ago_store_family']) / 7
    store_family_daily['monthly_slope_store_family'] = (store_family_daily['sales_last_day_store_family'] - store_family_daily['sales_month_ago_store_family']) / 30

    df = df.merge(store_family_daily.drop(columns=['sales']), on=['date', 'store_nbr', 'family'], how='left')

    # --- C. Store Level Features ---
    store_daily = df.groupby(['date', 'store_nbr'])['sales'].mean().reset_index()
    store_daily['sales_last_day_store'] = store_daily.groupby('store_nbr')['sales'].transform(lambda x: x.shift(1))
    
    grouper_s = store_daily.groupby('store_nbr')['sales_last_day_store']
    
    store_daily['rolling_mean_store_7'] = grouper_s.transform(lambda x: x.rolling(7).mean())
    store_daily['rolling_mean_store_30'] = grouper_s.transform(lambda x: x.rolling(30).mean())
    store_daily['sales_week_ago_store'] = grouper_s.transform(lambda x: x.shift(7))
    store_daily['sales_month_ago_store'] = grouper_s.transform(lambda x: x.shift(30))
    
    store_daily['weekly_change_store'] = store_daily['sales_last_day_store'] / store_daily['sales_week_ago_store']
    store_daily['weekly_slope_store'] = (store_daily['sales_last_day_store'] - store_daily['sales_week_ago_store']) / 7
    store_daily['monthly_slope_store'] = (store_daily['sales_last_day_store'] - store_daily['sales_month_ago_store']) / 30

    df = df.merge(store_daily.drop(columns=['sales']), on=['date', 'store_nbr'], how='left')
    
    return df
def prepare_and_fit_train(model, train_df, categorical_features=[], 
                          numerical_features=[], 
                          target_col='sales', 
                          feature_selection=False, 
                          rmsle_metric=True):

    """ train_df should include all engineered features"""

    X_train = train_df[categorical_features + numerical_features]
    y_train = train_df[target_col]

    X_train.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_train.fillna(0, inplace=True)

    cols_to_min_max = ['year', 'month', 'day', 'dayofweek', 'dayofyear', 'quarter']
    cols_not_scale = ['month_sin', 'month_cos', 'dayofweek_sin', 'dayofweek_cos', 'quarter_sin', 'quarter_cos']
    cols_to_z_normalize = [col for col in numerical_features if col not in (cols_not_scale+cols_to_min_max)]
    
    minmax_scaler = MinMaxScaler()
    X_train[cols_to_min_max] = minmax_scaler.fit_transform(X_train[cols_to_min_max])

    z_scaler = StandardScaler()
    X_train[cols_to_z_normalize] = z_scaler.fit_transform(X_train[cols_to_z_normalize])


    X_train = pd.get_dummies(X_train, columns=categorical_features, drop_first=True)

    if rmsle_metric:
        y_train = np.log1p(y_train)
        
    if feature_selection:
        # categorical_features = [x for x in categorical_features if x in feature_selection]
        # numerical_features = [x for x in numerical_features if x in feature_selection]
        # cols_to_min_max = [x for x in cols_to_min_max if x in feature_selection]
        # cols_to_z_normalize = [x for x in cols_to_z_normalize if x in feature_selection]
        X_train = X_train[feature_selection]
    
    model.fit(X_train, y_train)

    predictions = np.clip(model.predict(X_train), 0, None) # Clip negative predictions to 0.0
    results = pd.DataFrame({'actual': y_train, 'predicted': predictions, 'date': train_df['date']})

    ## We know that sales on January 1st and December 25th are 0 for all stores and categories.
    januaries_1st = ["2015-01-01", "2016-01-01", "2017-01-01", "2018-01-01", "2019-01-01"]
    christmans_dates = ["2015-12-25", "2016-12-25", "2017-12-25", "2018-12-25", "2019-12-25"]
    results.loc[(results['date'].isin(januaries_1st+christmans_dates)), 'predicted'] = 0
    corrected_predictions = results['predicted']

    if rmsle_metric:
        y_train = np.expm1(y_train)
        corrected_predictions = np.expm1(corrected_predictions)

    print(f"Train RMSLE: {rmsle_manual(y_train, corrected_predictions)}")
    print(f"Train RMSE: {root_mean_squared_error(y_train, corrected_predictions)}")
    print(f"Train MAE: {mean_absolute_error(y_train, corrected_predictions)}")

    return model, minmax_scaler, z_scaler, X_train.columns

    
def predict_test(trained_model, minmax_scaler, z_scaler, test_df, train_df, X_cols, categorical_features=[], numerical_features=[], 
                 target_col='sales', feature_selection=False,
                 rmsle_metric=True):
    
    dates_to_predict = sorted(test_df['date'].unique())
    dates_trained_on = sorted(train_df['date'].unique())
    history_dates_to_use = dates_trained_on[-35:]

    cols_to_min_max = ['year', 'month', 'day', 'dayofweek', 'dayofyear', 'quarter']
    cols_not_scale = ['month_sin', 'month_cos', 'dayofweek_sin', 'dayofweek_cos', 'quarter_sin', 'quarter_cos']
    cols_to_z_normalize = [col for col in numerical_features if col not in (cols_not_scale+cols_to_min_max)]

    columns = test_df.columns
    history_data = train_df[train_df['date'].isin(history_dates_to_use)][columns]

    predicted_results_list = []

    for current_date in tqdm(dates_to_predict, desc='Predicting test set'):
        # print(f'Preparing data for date: {current_date.date()}')
        current_date_data = test_df[test_df['date']==current_date]
        current_date_data.drop(columns=['id'], inplace=True, errors='ignore')
        
        engineered_df = pd.concat([history_data, current_date_data], axis=0)
        engineered_df = engineer_features(engineered_df)
        engineered_test = engineered_df[engineered_df['date']==current_date]
    

        X_test = engineered_test[categorical_features + numerical_features]
        y_test = engineered_test[target_col]

        X_test.replace([np.inf, -np.inf], np.nan, inplace=True)
        X_test.fillna(0, inplace=True)
        X_test[cols_to_min_max] = minmax_scaler.transform(X_test[cols_to_min_max])

        X_test[cols_to_z_normalize] = z_scaler.transform(X_test[cols_to_z_normalize])

        X_test = pd.get_dummies(X_test, columns=categorical_features, drop_first=True)
        X_test = X_test.reindex(columns=X_cols, fill_value=0)

        if feature_selection:
            # categorical_features = [x for x in categorical_features if x in feature_selection]
            # numerical_features = [x for x in numerical_features if x in feature_selection]
            # cols_to_min_max = [x for x in cols_to_min_max if x in feature_selection]
            # cols_to_z_normalize = [x for x in cols_to_z_normalize if x in feature_selection]
            X_test = X_test[feature_selection]        

        # print(f'Predicting sales for date: {current_date.date()}')
        test_predictions = np.clip(trained_model.predict(X_test), 0, None) # Clip negative predictions to 0.0
        test_results = pd.DataFrame({'actual': y_test, 'predicted': test_predictions, 'date': [current_date] * len(test_predictions)})
        januaries_1st = ["2015-01-01", "2016-01-01", "2017-01-01", "2018-01-01", "2019-01-01"]
        christmans_dates = ["2015-12-25", "2016-12-25", "2017-12-25", "2018-12-25", "2019-12-25"]
        test_results.loc[(test_results['date'].isin(januaries_1st+christmans_dates)), 'predicted'] = 0
        corrected_test_predictions = test_results['predicted']
        
        history_data = pd.concat([history_data, current_date_data], axis=0)
        new_dates_of_history = history_data['date'].unique()[-35:]
        history_data = history_data[history_data['date'].isin(new_dates_of_history)]
        predicted_results_list.extend(corrected_test_predictions.tolist())
        # print(f"num predictions is {len(predicted_results_list)}")

    if rmsle_metric:
        predicted_results_list = np.expm1(np.array(predicted_results_list))
    else:
        predicted_results_list = np.array(predicted_results_list)

    print(f"Test RMSLE: {rmsle_manual(test_df[target_col], predicted_results_list)}")
    print(f"Test RMSE: {root_mean_squared_error(test_df[target_col], predicted_results_list)}")
    print(f"Test MAE: {mean_absolute_error(test_df[target_col], predicted_results_list)}")
    
    return predicted_results_list
def add_time_features(data):
    # Engineer features
    data['date'] = pd.to_datetime(data['date'])

    data['year'] = data['date'].dt.year
    data['month'] = data['date'].dt.month
    data['day'] = data['date'].dt.day
    data['dayofweek'] = data['date'].dt.dayofweek
    data['dayofyear'] = data['date'].dt.dayofyear
    data['quarter'] = data['date'].dt.quarter

    data['is_weekend'] = data['dayofweek'].isin([5,6])
    data['is_holiday'] = data['type_holiday'].apply(lambda x: 1 if x != 'None' else 0)
    cyclic_cols = {'month': 12, 'dayofweek': 6, 'quarter': 4}

    data['is_first_january'] = (data['month'] == 1) & (data['day'] == 1)

    for feat in cyclic_cols.keys():
        data[f'{feat}_sin'] = np.sin(2 * np.pi * data[feat] / cyclic_cols[feat])
        data[f'{feat}_cos'] = np.cos(2 * np.pi * data[feat] / cyclic_cols[feat])

    return data
        