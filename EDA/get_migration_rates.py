import os
import numpy as np
import pandas as pd

states = [
    "Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado", "Connecticut", 
    "Delaware", "District of Columbia ", "Florida", "Georgia", "Hawaii", "Idaho", "Illinois", "Indiana", "Iowa", 
    "Kansas", "Kentucky", "Louisiana", "Maine", "Maryland", "Massachusetts", "Michigan", 
    "Minnesota", "Mississippi", "Missouri", "Montana", "Nebraska", "Nevada", 
    "New Hampshire", "New Jersey", "New Mexico", "New York", "North Carolina", 
    "North Dakota", "Ohio", "Oklahoma", "Oregon", "Pennsylvania", "Rhode Island", 
    "South Carolina", "South Dakota", "Tennessee", "Texas", "Utah", "Vermont", 
    "Virginia", "Washington", "West Virginia", "Wisconsin", "Wyoming"   
]

# list of states

def clean_data(year):
    '''
    function designed to read a migration dataset by US Census and clean it
    args:
        - year: year we want to read
            function has different actions depending on year value
    returns:
        - pandas dataframe containing migration data between states as well as totals
    '''


    text_file = f"/Users/wangmingye/Documents/Homework/PIC 16B/Final Project/Raw Data/Migration flows/State_to_State_Migration_Table_{year}.xls"
    # url to read the data (locally stored)
        # we will change this to the github links later
        # all datasets will be uploaded to the github repository

    data = pd.read_excel(text_file, index_col=0)

    data = data.loc[states]     # get only rows with states


    data.index.name = "State"   # rename index

    for col in data.columns:
        converted = pd.to_numeric(data[col], errors='coerce')
        if converted.isna().all():
            data.drop(columns=[col], inplace=True)

    # convert all columns to numerical, if they are all NA (characters only will cause this), drop the column

    data.drop(data.columns[1::2], axis=1, inplace=True)
    # drop every even-numbered column (this is the margin of error and we don't care about it yet)
        

    if year <= 2009:
    # this is because datasets before and after 2009 are formatted differently

        data.drop(data.columns[39],  axis=1, inplace=True)
        # drop Puerto Rico because it's a column in the dataset we don't care about
        data.columns = states
        # rename the columns
        data["diff_residence_total"] = data.apply(lambda x: x.sum() - x.max(), axis=1)
        # make a column that is the total people in a different residence last year
        data["Pop_total"] = data.iloc[:, :-1].sum(axis = 1)
        # aggregate total population

    else:
    # data after 2009 has extra columns, most of which we don't need

        data.drop(data.columns[-3:], axis = 1, inplace = True)
        # drop the last three columns (contain specifics about moving from abroad and we don't care)
        data.columns = ["Pop_total", "Same house", "Same state", "diff_residence_total"] + states + ["abroad_total"]
        # rename the columns so we are consistent with datasets 2009 and prior

        for s in states:
            data[s] = data.apply(lambda x: x['Same house'] + x['Same state'] if pd.isna(x[s]) else x[s], axis=1)
            # for each row, we add the values from same house and same state and put it into the NA values
                # for a state, if the state of residence from last year is the same state, the value is entered 
                # as NA and we would like to have a full matrix, so we use the columns that record same residency from
                # the previous year

        data.drop(data.columns[[1,2]], axis = 1, inplace = True)
        # drop same house and same state columns because we don't need them

    data["year"] = year
    # make a column with the year

    return data
    # return processed data

def migration_matrix(data, prop = False):
    '''
    function that returns a matrix detailing migration values between states
    args:
        - data: dataframe containing information
        - prop: whether or not to show proportions
            proportions are based off of total population value
    returns:
        - 51 x 52 dataframe (51 x 51 states, one column for year)
    '''

    output = data.loc[:,states + ["Pop_total", "year"]].copy()
    # make a copy with only states and population totals

    if prop:
        output = output.iloc[:,:-2].div(output.iloc[:, -2], axis=0)
    # if prop is true, calculate the percentage of people who were in each state last year for each state
    
    output.drop(output.columns[-2], axis = 1, inplace = True)
    # drop the total population column, we don't need it
    
    return output

def migration_rates(data):
    '''
    function returning a dataframe for a year and the migration rates in, out, and net by state
    args:
        - data: dataframe
    returns: 
        - dataframe with migration data
    '''

    output = data.copy()
    # make a copy

    output["net in"] = output.apply(lambda x: x["diff_residence_total"] / x["Pop_total"], axis = 1)
    # calculate net in by dividing total ppl who weren't in state last year by total population
    
    for s in states:
        total = output[s].sum()
        out = (total - output[s].max()) / total
        output.loc[s, "net out"] = out
    # calculate net out by getting sum of columns (total migration for this year), divide by sum people who 
        # aren't in same state this year
    
    output["net total"] = output["net in"] - output["net out"]
    # net total is net in - net out
    output = output[["year", "net in", "net out", "net total"]]
    # only keep migration rates and year

    return output


    
