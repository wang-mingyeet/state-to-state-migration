import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from fredapi import Fred

API_KEY = 'e9a0bfa2ee04031efa59f3baf0a3b2f4'

# all 50 state abbreviations
STATES = ['AL','AK','AZ','AR','CA','CO','CT','DE','FL','GA',
          'HI','ID','IL','IN','IA','KS','KY','LA','ME','MD',
          'MA','MI','MN','MS','MO','MT','NE','NV','NH','NJ',
          'NM','NY','NC','ND','OH','OK','OR','PA','RI','SC',
          'SD','TN','TX','UT','VT','VA','WA','WV','WI','WY']

def get_data(api_key=API_KEY):
    '''
    Gets unemployment rate data from FRED for all 50 states.
    '''
    # use FRED API, initialize with API key
    fred = Fred(api_key=api_key)

    # dictionary to hold data
    data = {}
    
    # get the time series for each state
    for state in STATES:
        # each state is labelled as state id and UR for unemployment rate
        series_id = f"{state}UR"
        # use try-except in case it errors
        try:
            data[state] = fred.get_series(series_id)
        except Exception as e:
            print(f"Failed to fetch data for {state}: {e}")
    
    return pd.DataFrame(data)

def clean_data(df):
    '''
    Clean, filter, and process the data.
    '''
    # there are no NAs in the data - no need to fill or drop NAs
    df_clean = df.copy()
    # filter for years 2005–2023
    df_clean = df_clean[(df_clean.index >= '2005-01-01') & (df_clean.index <= '2023-12-31')]
    # convert to date time.
    df_clean.index = pd.to_datetime(df_clean.index)
    # sort by year
    df_clean = df_clean.sort_index()
    return df_clean

def detect_outliers(df, threshold=3):
    '''
    Detect outliers using z-score threshold.

    Args:
        df: pandas dataframe
        threshold: 
    Returns:
        All extreme values above the z-score threshold.
    '''
    z_scores = (df - df.mean()) / df.std()
    return (z_scores.abs() > threshold).sum().sort_values(ascending=False)

def reshape(df):
    '''
    Reshape DataFrame with years as rows and states as columns to long format with 'state', 'year'.
    '''
    df.index.name = 'year'
    df = df.copy()
    df.index = df.index.year  # convert datetime to year
    # this will make it consistent with the migration data and easier to merge
    df_long = df.reset_index().melt(id_vars='year', var_name='state', value_name='unemployment_rate')
    return df_long

def add_yoy_change(df):
    '''
    Add year-over-year change in unemployment rate per state.
    '''
    # feature engineering - it would be helpful to add a column representing the change in unemployment rate 
    # this helps models detect directional shifts and not just static values
    df = df.copy()
    df.sort_values(['state', 'year'], inplace=True)
    # the first year (ie. 2005) will have NaN since there is no previous year value - fill with zeros
    df['unemployment_yoy_change'] = df.groupby('state')['unemployment_rate'].diff().fillna(0)
    return df

def plot_unemployment_trends(df, states_subset=None):
    '''
    Line plot of unemployment over time.

    Args:
        df: pandas dataframe
        states_subset: (optional) subset of state codes to show unemployment data
    '''
    plt.figure(figsize=(14, 8))
    if states_subset:
        df[states_subset].plot()
    else:
        df.plot(legend=False, alpha=0.5)
    plt.title("Unemployment Rates by State")
    plt.xlabel("Date")
    plt.ylabel("Unemployment Rate (%)")
    plt.show()

def plot_correlation_heatmap(df):
    '''
    Heatmap of correlations across states.
    '''
    correlation = df.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation, cmap="coolwarm", center=0, square=True)
    plt.title("Correlation Heatmap of State Unemployment Rates")
    plt.tight_layout()
    plt.show()

def plot_choropleth(df):
    '''
    Animated choropleth map of unemployment by state and year.
    '''
    fig = px.choropleth(
        df,
        locations='state',
        locationmode='USA-states',
        color='unemployment_rate',
        animation_frame='year',
        scope='usa',
        color_continuous_scale='Viridis',
        title='U.S. Unemployment Rate by State (2005–2023)'
    )
    fig.show()

def plot_scatter_yoy_vs_rate(df):
    '''
    Scatter plot of mean unemployment rate vs mean YoY change per state.
    '''
    # group the data by state and get the mean unemployment and YoY change
    grouped = df.groupby('state').agg({
        'unemployment_rate': 'mean',
        'unemployment_yoy_change': 'mean'
        }).reset_index()
    # plot
    sns.scatterplot(data=grouped, x='unemployment_rate', y='unemployment_yoy_change', hue='state', legend=False)
    plt.title("Mean Unemployment Rate vs Mean YoY Change by State (2005–2023)")
    plt.xlabel("Mean Unemployment Rate (%)")
    plt.ylabel("Mean YoY Change (%)")
    plt.tight_layout()
    plt.show()