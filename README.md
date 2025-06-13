# State to State Migration

### Group Contribution Statement
Mingye headed the acquisition and cleaning of the Census Bureau migration data, as well as writing the script that generated the migration matrix and net migration rates. He conducted the EDA on the migration data and was in charge of working on the clustering models. He also plotted the clustering results using PCA, geography plots, and time series plots.

Rainie conducted most of the literature review and background research on similar projects. Then, she performed the EDA for the home value index data. She was also responsible for the classification models for predicting the direction of net migration (inflow/outflow) using Random Forest, XGBoost, Logistic Regession, and ensemble models.

Lingyin led the acquisition and cleaning of raw microdata from IPUMS and computed weighted state-year aggregates for all the feature variables to produce a processed dataset that supported the entire team’s analysis. Lingyin also conducted EDA on the feature data and performed feature selection by running Lasso regression. Finally, he implemented, tuned, and tested the three regression models (Random Forest, Gradient Boosting, and a dropout-regularized MLP) through grid searches and architectural refinements.
