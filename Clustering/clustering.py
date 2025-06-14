# standard pkgs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# plotting
import plotly.express as px
import plotly.graph_objects as go

# modeling
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

state_abbrev = {
    'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR',
    'California': 'CA', 'Colorado': 'CO', 'Connecticut': 'CT', 'Delaware': 'DE',
    'Florida': 'FL', 'Georgia': 'GA', 'Hawaii': 'HI', 'Idaho': 'ID',
    'Illinois': 'IL', 'Indiana': 'IN', 'Iowa': 'IA', 'Kansas': 'KS',
    'Kentucky': 'KY', 'Louisiana': 'LA', 'Maine': 'ME', 'Maryland': 'MD',
    'Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota': 'MN', 'Mississippi': 'MS',
    'Missouri': 'MO', 'Montana': 'MT', 'Nebraska': 'NE', 'Nevada': 'NV',
    'New Hampshire': 'NH', 'New Jersey': 'NJ', 'New Mexico': 'NM',
    'New York': 'NY', 'North Carolina': 'NC', 'North Dakota': 'ND',
    'Ohio': 'OH', 'Oklahoma': 'OK', 'Oregon': 'OR', 'Pennsylvania': 'PA',
    'Rhode Island': 'RI', 'South Carolina': 'SC', 'South Dakota': 'SD',
    'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT', 'Vermont': 'VT',
    'Virginia': 'VA', 'Washington': 'WA', 'West Virginia': 'WV',
    'Wisconsin': 'WI', 'Wyoming': 'WY'
}
# dictionary for state name abbreviations, useful for plotly

class clustermodel:
    '''
    this is a class that we will use to perform cluster modeling on state migration inflows
        we will assume that the data has been cleaned per the functions in the get_migration_rates file
    class will have a clustering model as well as functions to plot visuals
    '''

    def __init__(self, data, year = None, prop = True):
        '''
        initialization function
        args:
            data: dataframe to be passed in, for now we assume it has been cleaned
            year: optional argument to set the year
        '''

        self.df = data
        self.year = year
        # stores the year
        self.standardized = False
        # a status to check if the data has been standardized
        self.prop = prop
        # status to check if we are looking at proportions or raw numbers

        if self.df.shape != (51,51):
            raise ValueError("Dataframe must be 51 x 51!")
        # we require that the data does not contain any information except for the state to state inflow data

        self.X = data.copy()
        for state in self.X.index:
            if state in self.X.columns:
                self.X.at[state, state] = 0
        # since we only care about state flow to other states, all diagonal entries are set to 0
            # since the overwhelming majority of people do not move states

        self.X = self.X.to_numpy()
        # X will be where we perform all data manips on, it will be stored as a numpy array

        # note: we expect the dataframe to be a migration matrix created from get_migration_rates
            # however, we do not care if the matrix is proportion or raw numbers, both are useful in their own ways

        self.clusters = None
        # eventually we will store the cluster values for each state
        
    def standardize(self):
        '''
        standardizes the data
        '''

        scaler = StandardScaler()
        self.X = scaler.fit_transform(self.X)

        self.standardized = True
        print("Data has been standardized!")
    
    def inertia_plot(self, k):
        '''
        provides a plot of inertias ranging over 1 - k clusters
        the user can use the elbow method to determine the optimal cluster
        args:
            k: up to how many clusters we want to test, must be at least 1
        '''

        inertias = []
        ks = list(range(1, k+1))
        for k in ks:
            model = KMeans(n_clusters=k, random_state=0).fit(self.X)
            inertias.append(model.inertia_)

        fig = go.Figure(data=go.Scatter(
            x=ks,
            y=inertias,
            mode='lines+markers',
            text=[f"k = {k}" for k in ks],
            hoverinfo='text+y'
        ))

        fig.update_layout(
            title="Elbow Method for Optimal k",
            xaxis_title="Number of Clusters (k)",
            yaxis_title="Inertia",
            hovermode="closest"
        )

    def model_fit(self, k):
        '''
        fits a KMeans clustering model to the data
        args:
            k: how many clusters we want
        '''

        model = KMeans(n_clusters=k, random_state=0)
        labels = model.fit_predict(self.X)
        print("Model fit to data successfully.")

        centers = model.cluster_centers_
        axis_values = centers[:, 0]
        ordered_cluster_ids = np.argsort(axis_values)
        label_map = {old: new for new, old in enumerate(ordered_cluster_ids)}
        self.clusters = np.vectorize(label_map.get)(labels)

    def plot_clusters(self, graph = True, return_data = False):
        '''
        performs PCA with 2 components on the data, creates a dataframe, and plots the clusters in 2D space
        args:
            graph: by default, we set this to true because we want to see the clustering
                setting to false means we don't graph
            return_data: returns the PCA dataframe as well as the clustering data, only returns if true
        '''

        if not self.standardized:
            warnings.warn("Warning... Data is not standardized, results may be skewed")

        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(self.X)
        # performs PCA on fitted data

        pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
        pca_df['cluster'] = self.clusters
        pca_df['State'] = self.df.index
        # create the PCA dataframe and assign the cluster values

        if graph:
            fig = px.scatter(
                pca_df,
                x='PC1',
                y='PC2',
                color='cluster',
                hover_name='State',
                title='PCA Visualization of KMeans Clustering of States',
                color_discrete_sequence='Viridis'
            )

            fig.update_traces(marker=dict(size=10, line=dict(width=1, color='DarkSlateGrey')))
            fig.update_layout(title_x=0.5)
            fig.show()
        # use plotly to graph our clustering after PCA

        if return_data:
            return pca_df
        # return the PCA dataset if called for

    def plot_map(self, graph = True, return_data = False):
        '''
        this plots the states on a map where we can see the clustering between states
            each state has its 3 highest contributors to inflow plotted as hover data
        args:
            graph: by default, we set this to true because we want to see the clustering
                setting to false means we don't graph
            return_data: returns the top three dataframe as well as the clustering data, only returns if true
        '''
        
        def top_3_with_values(row):
            top = row.sort_values(ascending=False).head(4)
            return pd.Series([top.index[1], top.iloc[1],
                            top.index[2], top.iloc[2],
                            top.index[3], top.iloc[3]])
        # function that goes through each row, pulls the top three highest values and their corresponding index
            # since most people stay in state, we pull the 2nd - 4th highest values instead

        output = self.df.apply(top_3_with_values, axis=1)
        output.columns = ['Top_1', 'Value_1', 'Top_2', 'Value_2', 'Top_3', 'Value_3']
        # apply the function and label the columns

        output["cluster"] = self.clusters

        if graph:
            if self.prop:
                output['hover'] = (
                    'Top 1: ' + output['Top_1'] + ' (' + (output['Value_1']*100).round(2).astype(str) + '%)<br>' +
                    'Top 2: ' + output['Top_2'] + ' (' + (output['Value_2']*100).round(2).astype(str) + '%)<br>' +
                    'Top 3: ' + output['Top_3'] + ' (' + (output['Value_3']*100).round(2).astype(str) + '%)'
                )
                # we multiply the proportions by 100 to get percentages
            else:
                output['hover'] = (
                    'Top 1: ' + output['Top_1'] + ' (' + (output['Value_1']).astype(str) + '%)<br>' +
                    'Top 2: ' + output['Top_2'] + ' (' + (output['Value_2']).astype(str) + '%)<br>' +
                    'Top 3: ' + output['Top_3'] + ' (' + (output['Value_3']).astype(str) + '%)'
                )
                # we don't do this if it is raw numbers though

            output['State'] = output.index.map(state_abbrev)
            # add state abbreviations

            fig = px.choropleth(
                output,
                locations='State',
                locationmode='USA-states',
                color='cluster',
                hover_name='State',
                hover_data={'hover': True, 'State': False, 'cluster': False},
                scope='usa',
                color_continuous_scale='Viridis',
                title='Clustering States by Migration Inflow'
            )

            fig.update_traces(marker_line_width=0.5)
            fig.update_layout(title_x=0.5)
            fig.show()
        
        if return_data:
            return output

                    
