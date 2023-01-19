import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.optimize as opt
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score

from typing import List


def read_file_to_frame(path: str, col_to_drop: List[str], skiprows: int) -> pd.DataFrame:
    '''
    Creates a dataframe from csv file.

    Parameters
    ----------
    path : str
        The file path to csv file

    col_to_drop: str
        Column to drop

    skiprows: int
        Rows to skip
        
    Returns
    -------
    Pandas DataFrame
        A DataFrame representations of the csv file content
    '''

    # Read content in the file
    df = pd.read_csv(path, skiprows=skiprows)

    # Remove unnecessary columns
    df = df.drop(col_to_drop, axis=1)

    # Reshape dataframe to long format
    df = df.melt(
        id_vars=['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code'],
        var_name="year",
        value_name="value"
    )

    return df


# Read in the Life Expectancy data
life_exp = read_file_to_frame(
    'API_SP.DYN.LE00.IN_DS2_en_csv_v2_4770434.csv',
    col_to_drop=['Unnamed: 66'],
    skiprows=4
)


# Read in the GDP per capita data
gdp_data = read_file_to_frame(
    'API_NY.GDP.PCAP.CD_DS2_en_csv_v2_4770417.csv',
    col_to_drop=['Unnamed: 66'],
    skiprows=4
)


# Concatenate the two datasets
df = pd.concat([life_exp, gdp_data])

# select relevant columns
df = df[['Country Name', 'Indicator Name', 'year', 'value']].copy()
# pivot the dataframe
df2 = df.pivot(index=['Country Name', 'year'],
                                columns='Indicator Name', 
                                values='value').reset_index()
# convert year to int
df2['year'] = df2['year'].astype(int)

# Define unwanted list of rows to remove
unwanted_list = ['Africa Eastern and Southern','Arab World','Caribbean small states','Central African Republic', 'Central Europe and the Baltics',
'Early-demographic dividend', 'East Asia & Pacific',
       'East Asia & Pacific (IDA & IBRD countries)',
       'East Asia & Pacific (excluding high income)','Europe & Central Asia',
       'Europe & Central Asia (IDA & IBRD countries)',
       'Europe & Central Asia (excluding high income)', 'European Union',
 'Fragile and conflict affected situations','French Polynesia','Heavily indebted poor countries (HIPC)',
 'High income', 'IBRD only',
       'IDA & IBRD total', 'IDA blend', 'IDA only', 'IDA total','Late-demographic dividend',
 'Latin America & Caribbean',
       'Latin America & Caribbean (excluding high income)',
       'Latin America & the Caribbean (IDA & IBRD countries)',
       'Least developed countries: UN classification', 'Low & middle income', 'Low income', 'Lower middle income',
 'Middle East & North Africa',
 'Middle East & North Africa (IDA & IBRD countries)',
       'Middle East & North Africa (excluding high income)',
       'Middle income', 'Not classified',
       'OECD members', 'Other small states',
       'Pacific island small states','Post-demographic dividend',
       'Pre-demographic dividend','Small states','South Asia (IDA & IBRD)','Sub-Saharan Africa', 
 'Sub-Saharan Africa (IDA & IBRD countries)',
       'Sub-Saharan Africa (excluding high income)','Upper middle income', 'West Bank and Gaza',
                 'World','Africa Western and Central'
]

# Subset the data to only include countries that are not in the unwanted_list
data = df2[~df2['Country Name'].isin(unwanted_list)]

# Subset the data for the years between 1970 and 1980
sixties_data = data[data['year'].between(1970, 1980)]

def group_and_mean(df, group_by, mean_columns):
    """
    Group a DataFrame and compute the mean value of multiple columns.
    
    Parameters
    ----------
    df : Pandas DataFrame
        The DataFrame to be grouped.
    group_by : str or list of str
        The column(s) to group by.
    mean_columns : str or list of str
        The column(s) to compute the mean value of.
        
    Returns
    -------
    Pandas DataFrame
        A DataFrame with the mean values of the specified columns, grouped by the specified column(s).
    """
    # Group the DataFrame
    grouped = df.groupby(group_by)
    
    # Compute the mean value of the specified columns
    mean_df = grouped[mean_columns].mean()
    
    return mean_df.reset_index()

# Compute the mean values for the data from the 1970s
sixties = group_and_mean(sixties_data, 'Country Name', ['GDP per capita (current US$)','Life expectancy at birth, total (years)'])

# Subset the data for the years between 2010 and 2021
recent_data = data[data['year'].between(2010, 2021)]

# Compute the mean values for the recent data
recent = group_and_mean(recent_data, 'Country Name', ['GDP per capita (current US$)','Life expectancy at birth, total (years)'])

# Remove missing values
recent = recent.dropna()

# Normalize the data

scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(recent.drop('Country Name', axis =1))

# Use the elbow method to determine the optimal number of clusters
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(data_scaled)
    wcss.append(kmeans.inertia_)

# Plot the WCSS values

plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Initialize the model with the optimal number of clusters
kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=0)

# Fit the model to the data
kmeans.fit(data_scaled)

# Use the silhouette score to evaluate the quality of the clusters
print(f'Silhouette Score: {silhouette_score(data_scaled, kmeans.labels_)}')

# Extract centroid values for the three clusters
centroids = kmeans.cluster_centers_
centroids = pd.DataFrame(centroids, columns=['GDP per capita (current US$)','Life expectancy at birth, total (years)'])
centroids.index = np.arange(1, len(centroids)+1) # Start the index from 1

# Plot the scatter plot of the clusters
plt.scatter(data_scaled[:, 0], data_scaled[:, 1], c=kmeans.labels_)
plt.title('K-Means Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

# Assign each country to a cluster
y_pred = kmeans.fit_predict(recent.drop('Country Name', axis =1))
recent['cluster'] = y_pred +1

# Plot the scatterplot of clusters against GDP and life expectancy
plt.figure(figsize=(12,6))
sns.set_palette("pastel")  
sns.scatterplot(y= recent['GDP per capita (current US$)'],
                x= recent['Life expectancy at birth, total (years)'], 
                hue= recent['cluster'], 
                palette='bright')
plt.title('Country Clusters Based on GDP per Capita and Life Expectancy', fontsize = 18)
plt.show()

# Logistic function forecasting
def logistic(t, n0, g, t0):
    """Calculates the logistic function with scale factor n0 and growth rate g"""
    f = n0 / (1 + np.exp(-g*(t - t0)))
    return f


import scipy.optimize as opt
import matplotlib.pyplot as plt

# apply the logistic function to GDP data for Switzerland
swz_data = data[data['Country Name']== 'Switzerland'].dropna()

# fit the logistic function to the data
param, covar = opt.curve_fit(logistic, swz_data["year"], swz_data["GDP per capita (current US$)"], 
                             p0=(500061.36, 0.03, 1999))

sigma = np.sqrt(np.diag(covar))

# calculate the forecast
swz_data["fit"] = logistic(swz_data["year"], *param)

# Plot the forecast
plt.plot(swz_data["year"], swz_data["GDP per capita (current US$)"], 
         label="GDP per capita (current US$)")
plt.plot(swz_data["year"], swz_data["fit"], label="forecast")
plt.xlabel("year")
plt.ylabel("GDP per capita (current US$)")
plt.legend()
plt.title('Switzerland GDP Forecast')
plt.show()

# Forecast for the next 10 years
year = np.arange(1960, 2030)
forecast = logistic(year, *param)

plt.figure()
plt.plot(swz_data["year"], swz_data["GDP per capita (current US$)"], 
         label="GDP per capita (current US$)")
plt.plot(year, forecast, label="forecast")
plt.xlabel("year")
plt.ylabel("GDP per capita (current US$)")
plt.legend()
plt.title('Switzerland GDP Forecast for next 10 years')
plt.show()


# Error ranges calculation
def err_ranges(x, func, param, sigma):
    """
    This function calculates the upper and lower limits of function, parameters and
    sigmas for a single value or array x. The function values are calculated for 
    all combinations of +/- sigma and the minimum and maximum are determined.
    This can be used for all number of parameters and sigmas >=1.
    """

    import itertools as iter
    
    # initiate arrays for lower and upper limits
    lower = func(x, *param)
    upper = lower
    
    # Create a list of tuples of upper and lower limits for parameters
    uplow = []   
    for p,s in zip(param, sigma):
        pmin = p - s
        pmax = p + s
        uplow.append((pmin, pmax))
        
    pmix = list(iter.product(*uplow))
    
    # calculate the upper and lower limits
    for p in pmix:
        y = func(x, *p)
        lower = np.minimum(lower, y)
        upper = np.maximum(upper, y)
        
    return lower, upper

# calculate error ranges 
low, up = err_ranges(year, logistic, param, sigma)

plt.figure()
plt.plot(swz_data["year"], swz_data["GDP per capita (current US$)"], 
         label="GDP per capita (current US$)")
plt.plot(year, forecast, label="forecast")

plt.fill_between(year, low, up, color="yellow", alpha=0.7)
plt.xlabel("year")
plt.ylabel("GDP")
plt.legend()
plt.title('Switzerland GDP Forecast and Confidence Interval')
plt.show()

print(err_ranges(2030, logistic, param, sigma))

# Get Nigeria GDP data and drop missing values
nig_data = data[data['Country Name']== 'Nigeria'].dropna()

# Plot trend of GDP per capita over the years
nig_data.plot("year", "GDP per capita (current US$)")
plt.title('Nigeria GDP Trend')
plt.show()

# Use logistic function to fit the data and predict forecast
param, covar = opt.curve_fit(logistic, nig_data["year"], nig_data["GDP per capita (current US$)"], 
                             p0=(5000, 0.05, 1960))
sigma = np.sqrt(np.diag(covar))
print("parameters:", param)
print("std. dev.", sigma)
nig_data["fit"] = logistic(nig_data["year"], *param)
nig_data.plot("year", ["GDP per capita (current US$)", "fit"])
plt.title('Nigeria GDP Trend')
plt.show()

# Generate year range to predict forecast
year = np.arange(1960, 2030)
forecast = logistic(year, *param)

# Plot the forecasted GDP per capita
plt.figure()
plt.plot(nig_data["year"], nig_data["GDP per capita (current US$)"], 
         label="Life expectancy at birth, total (years)")
plt.plot(year, forecast, label="forecast")
plt.xlabel("year")
plt.ylabel("GDP per capita (current US$)")
plt.legend()
plt.title('Nigeria GDP Forecast')
plt.show()

# Calculate error ranges and plot the forecast with confidence intervals
low, up = err_ranges(year, logistic, param, sigma)
plt.figure()
plt.plot(nig_data["year"], nig_data["GDP per capita (current US$)"], 
         label="GDP per capita (current US$)")
plt.plot(year, forecast, label="forecast")
plt.fill_between(year, low, up, color="yellow", alpha=0.7)
plt.xlabel("year")
plt.ylabel("GDP")
plt.legend()
plt.title('Nigeria GDP Forecast and Confidence Interval')
plt.show()

# Print the error range for GDP per capita in 2030
print(err_ranges(2030, logistic, param, sigma))

# Get residuals, standard deviation and Z-scores
nig_data["diff"] = nig_data["GDP per capita (current US$)"] - nig_data["fit"]
sigma = nig_data["diff"].std()
nig_data["z"] = abs(nig_data["diff"] / sigma)

# Filter out data points outside 3 standard deviation
nig_data = nig_data[nig_data["z"]<3.0].copy()

# Re-plot the forecast with filtered data

plt.figure()
plt.plot(nig_data["year"], nig_data["GDP per capita (current US$)"], 
         label="GDP per capita (current US$)")
plt.plot(year, forecast, label="forecast")


plt.figure()
plt.plot(nig_data["year"], nig_data["GDP per capita (current US$)"], 
         label="GDP per capita (current US$)")
plt.plot(year, forecast, label="forecast")

plt.fill_between(year, low, up, color="yellow", alpha=0.7)
plt.xlabel("year")
plt.ylabel("GDP")
plt.legend()
plt.show()

