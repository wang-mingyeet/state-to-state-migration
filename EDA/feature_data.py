import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

# data set was downloaded from IPUMS, containing some features I think that would influence the migration rate
df = pd.read_csv("/Users/linh/Downloads/usa_00006.csv")

# Inspect the whole data set
print(len(df))
df.head()

# inspect each column

# for col in df.columns:
#     print(df[col].unique())
# df[df['INCTOT']== 1].shape[0]
# df['INCTOT'].value_counts().sort_index()

# drop unneeded columns
df = df.drop(columns=['OWNERSHPD', 'PERNUM', 'EDUCD', 'EMPSTATD'])

# check data type

# print(df.info())
df['PERWT'] = df['PERWT'].astype(int)

print(df.info())
# df


# Below are the descriptions for each import data features:
# 1. YEAR: from 2010 to 2023 excluding 2020
# 2. STATEFIP: 51 codes that identify different states (link to the info: https://usa.ipums.org/usa-action/variables/STATEFIP#codes_section)
# 3. OWNERSHP: 
# 
#         0: N/A
#         1: Owned or being bought (loan)
#         2: Rented
# 
# 4. PERWT: how many persons in the U.S. population are represented by a given person
# 5. SEX: 
#         
#         1: Male
#         2: Female
# 
# 6. Age: Age of person
# 7. EDUC:
# 
#          00	N/A or no schooling
#          01	Nursery school to grade 4
#          02	Grade 5, 6, 7, or 8
#          03	Grade 9
#          04	Grade 10
#          05	Grade 11
#          06	Grade 12
#          07	1 year of college
#          08	2 years of college
#          09	3 years of college
#          10	4 years of college
#          11	5+ years of college
# 
# 8. EMPSTAT: 
# 
#         0: N/A
#         1: Employed
#         2: Unemployed
#         3: Not in labor force
# 
# 9. INCTOT: Total personal income
# 
#         0000000 = None
#         9999999 = N/A

# since each feature has unique representation for nan values, need to convert them to nan
df.loc[df['OWNERSHP'] == 0, 'OWNERSHP'] = np.nan
df.loc[df['EMPSTAT'] == 0, 'EMPSTAT'] = np.nan
df.loc[df['INCTOT'] == 9999999, 'INCTOT'] = np.nan


# Some I think shoul include in the model:
# - % Renters: Higher renter proportions may indicate greater short-term migration potential
# - Median Income: Reflects economic opportunity and affordability, both of which influence migration decisions
# - % Unemployed: Measures job scarcity
# - % College Graduates: Proxy for human capital and economic potential of the state
# - % Young Adults (20–34): Peak migration cohort
# - % Elderly (65+): Reflects retirement-driven migration
# - % Female: May be useful for understanding gender-based labor and caregiving migration dynamics

# I will use IPUMS microdata and aggregated by state and year. To ensure population-level accuracy, I will apply PERWT (person weight) when computing:
# 	•	Proportions (e.g., % renters, % unemployed): weighted share of individuals meeting the condition
# 	•	Medians (e.g., income, age): weighted median based on cumulative PERWT
# 
# Formulas:
# - **Weighted proportion**:  
#   $$
#   \frac{\sum (\text{condition}_i \cdot \text{PERWT}_i)}{\sum \text{PERWT}_i}
#   $$
# - **Weighted Median**: Sort values by x_i, compute cumulative sum of PERWT, and find the value at 50% of total weight.

# Plan for calculation:
# | Feature Name           | How It Was Calculated                                                             |
# |------------------------|------------------------------------------------------------------------------------|
# | % Renters              | Share where `OWNERSHP == 2`, weighted by `PERWT`                                  |
# | Median Age             | Weighted median of `AGE`                                                          |
# | % Female               | Share where `SEX == 2`, weighted                                                  |
# | % Age 20–34            | Share where `20 <= AGE <= 34`, weighted                                           |
# | % Age 65+              | Share where `AGE >= 65`, weighted                                                 |
# | % College Grads        | Share where `EDUC >= 10`, weighted (4+ years of college)                          |
# | % Unemployed           | Share where `EMPSTAT == 2`, weighted                                              |
# | % Not in Labor Force   | Share where `EMPSTAT == 3`, weighted                                              |
# | Median Income          | Weighted median of `INCTOT`, after removing `INCTOT == 0` and `9999999`           |
# 

# helper functions
def weighted_proportion(condition, weights):
    '''
    Returns weighted proportion of True values in `condition`.
    Ignores missing values.
    '''
    mask = condition.notna() & weights.notna()
    return (condition[mask] * weights[mask]).sum() / weights[mask].sum()

def weighted_median(values, weights):
    """
    Returns weighted median of `values`.
    Ignores missing values.
    """
    mask = values.notna() & weights.notna()
    df_w = pd.DataFrame({
        'val': values[mask],
        'wt' : weights[mask]
    }).sort_values('val')
    cum_w = df_w['wt'].cumsum()
    cutoff = df_w['wt'].sum() / 2
    return df_w.loc[cum_w >= cutoff, 'val'].iloc[0]

# calculate each feature by state & year
state_features = df.groupby(['STATEFIP', 'YEAR']).apply(lambda g: pd.Series({
        'pct_renters'      : weighted_proportion(g['OWNERSHP'] == 2,    g['PERWT']),
        'median_age'       : weighted_median(    g['AGE'],               g['PERWT']),
        'pct_female'       : weighted_proportion(g['SEX']   == 2,    g['PERWT']),
        'pct_age_20_34'    : weighted_proportion((g['AGE'] >= 20) & (g['AGE'] <= 34), g['PERWT']),
        'pct_age_65_plus'  : weighted_proportion(g['AGE']   >= 65,   g['PERWT']),
        'pct_college_grads': weighted_proportion(g['EDUC']  >= 10,   g['PERWT']),
        'pct_unemployed'   : weighted_proportion(g['EMPSTAT'] == 2,   g['PERWT']),
        'pct_nilf'         : weighted_proportion(g['EMPSTAT'] == 3,   g['PERWT']),
        'median_income'    : weighted_median(g['INCTOT'], g['PERWT'])
    })
).reset_index()

state_features.head()


# # EDA:

# summary of numeric features
state_features.describe()


# ### Correlation

# year-over-Year First Differences
# compute year-to-year changes for each state
diff_df = (
    state_features
    .sort_values(['STATEFIP', 'YEAR'])
    .groupby('STATEFIP')
    .diff()
    .dropna()
)

feature_cols = [
    'pct_renters', 'median_age', 'pct_female',
    'pct_age_20_34', 'pct_age_65_plus', 'pct_college_grads',
    'pct_unemployed', 'pct_nilf', 'median_income'
]

# Correlation of the first differences
corr_diff = diff_df[feature_cols].corr()
corr_diff

# heatmap
plt.figure(figsize=(8,6))
plt.imshow(corr_diff, vmin=-1, vmax=1, cmap='RdYlBu')
plt.colorbar(label='r')
plt.xticks(range(len(feature_cols)), feature_cols, rotation=45, ha='right')
plt.yticks(range(len(feature_cols)), feature_cols)
plt.title("Correlation of Year-over-Year Changes")
plt.tight_layout()
plt.show()


# Finding:
# - Young adults boost rentals (ρ ≈ +0.35): States gaining 20–34 year-olds see bigger jumps in renter share.
# - Income up → inactivity down (ρ ≈ –0.36): Rising median income coincides with fewer people out of the labor force.
# - Jobs ↔ rentals (ρ ≈ –0.30): As unemployment falls, renter shares tend to climb.
# - Income–education link (ρ ≈ +0.17): Increases in median income modestly align with more college grads.
# - Young inflows lower age (ρ ≈ –0.33): A surge in 20–34 year-olds drives median age downward.

# ### Time series by state: trace trends in a few key states

# plot feature evolves over time for a few states
featurelist = ['median_income', 'pct_college_grads','pct_unemployed', 'pct_nilf','median_age']
states = [1, 6, 12]  # Alabama, California, Florida
# create one subplot per feature (stacked vertically)
fig, axes = plt.subplots(
    nrows=len(featurelist),
    ncols=1,
    figsize=(8, len(featurelist) * 2.5),
    sharex=True
)
# plot each feature over time for the selected states
for ax, feature in zip(axes, featurelist):
    for st in states:
        subset = state_features[state_features['STATEFIP'] == st]
        ax.plot(subset['YEAR'], subset[feature], label=f"FIPS {st}")
    ax.set_title(feature)
    ax.set_ylabel(feature)
    ax.legend()

axes[-1].set_xlabel("Year")
plt.tight_layout()
plt.show()


# Findings:
# - Median income climbs steadily in all three states
# - College‐grad share also rises
# - Unemployment falls everywhere
# - % Not in labor force rises into the mid-2010s then dips slightly
# - Median age steadily increased

# ### Scatter Plots: test pairwise relationships

# Income vs. % renters
plt.scatter(state_features['pct_renters'], state_features['median_income'], alpha=0.6)
plt.xlabel("% Renters")
plt.ylabel("Median Income")
plt.title("Median Income vs. % Renters")
plt.show()

# % Unemployment vs. % NILF
plt.scatter(state_features['pct_unemployed'], state_features['pct_nilf'], alpha=0.6)
plt.xlabel("% Unemployed")
plt.ylabel("% Not in Labor Force")
plt.title("Unemployment vs. Not in Labor Force")
plt.show()


# Findings:
# - Income vs. % Renters: The scatter is pretty diffuse, but there’s a slight tendency for states with more renters to have lower median incomes. Most states cluster around 0.25–0.35 renters and $20k–$35k income. A few high-renter outliers push income up to $40k–$60k, so it’s not a super-clean relationship.
# - % Unemployed vs. % Not in LF: Here you see a modest positive trend (r ≈ +0.3): states with higher unemployment also tend to have more people out of the labor force, poiinting that joblessness and inactivity often move together.

# ### Choropleth: watch how a factor evolves geographically

# fips code for all the states
fips_map = {
    1:  "AL",  2:  "AK",  4:  "AZ",  5:  "AR",  6:  "CA",
    8:  "CO",  9:  "CT", 10:  "DE", 11:  "DC", 12:  "FL",
    13: "GA", 15: "HI", 16:  "ID", 17:  "IL", 18:  "IN",
    19: "IA", 20:  "KS", 21:  "KY", 22:  "LA", 23:  "ME",
    24: "MD", 25: "MA", 26:  "MI", 27:  "MN", 28:  "MS",
    29: "MO", 30:  "MT", 31:  "NE", 32:  "NV", 33:  "NH",
    34: "NJ", 35: "NM", 36:  "NY", 37:  "NC", 38:  "ND",
    39: "OH", 40:  "OK", 41:  "OR", 42:  "PA", 44:  "RI",
    45: "SC", 46:  "SD", 47:  "TN", 48:  "TX", 49:  "UT",
    50: "VT", 51:  "VA", 53:  "WA", 54:  "WV", 55:  "WI",
    56: "WY"
}

# plot
df_plot = state_features.copy()
df_plot['state_code'] = df_plot['STATEFIP'].map(fips_map) # need to map to the actual state name
featurelist = ['median_income', 'pct_college_grads','pct_unemployed', 'pct_nilf','median_age']
for feature in featurelist:
    fig = px.choropleth(
        df_plot,
        locations="state_code", 
        locationmode="USA-states",
        color=feature,
        animation_frame="YEAR",
        scope="usa",
        color_continuous_scale="Viridis",
        title=f"{feature} by State Over Time"
    )
    fig.show()
