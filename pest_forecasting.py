# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime,date
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook
import numpy as np
import pandas as pd
from itertools import product
import warnings
warnings.filterwarnings('ignore')
import math
import random
from pySTARMA import starma_model as sm

query_df = pd.read_pickle("Normalised_KCC_Data.pkl")
query_df['CreatedOn'] = pd.to_datetime(query_df['CreatedOn'])
query_df['MonthYear'] = query_df['CreatedOn'].map(lambda x: str("1")+"-"+str(x.month) +"-"+ str(x.year))
query_df['MonthYear']= pd.to_datetime(query_df['MonthYear'])
#query_df["Year"] = query_df['CreatedOn'].dt.year

query_df = query_df[query_df["Year"] != 2018]
query_df = query_df[query_df["Year"] != 2019]
query_df = query_df[query_df["Year"] != 2020]
#query_df['Month'] = query_df['CreatedOn'].dt.month
#query_df['Date'] = query_df['CreatedOn'].dt.date
#query_df['Month_Year'] = pd.to_datetime(['{}-{}-01'.format(y, m) for y, m in zip(query_df.Year, query_df.Month)])
query_df['Harvest Area']=pd.to_numeric(query_df['Harvest Area'])
query_df['Count']=pd.to_numeric(query_df['Count'])
query_df['NormFrequency']=query_df['Count']/query_df['Harvest Area']
df = query_df.groupby(['Dist Name','Pest','CreatedOn'])['Count'].agg('sum').reset_index(name='pest_count')

#df = query_df.groupby(['Dist Name','Pest','CreatedOn']).size().reset_index(name='pest_count')
def optimize_SARIMA(parameters_list, d, D, s, exog):
    """
        Return dataframe with parameters, corresponding AIC and SSE
        
        parameters_list - list with (p, q, P, Q) tuples
        d - integration order
        D - seasonal integration order
        s - length of season
        exog - the exogenous variable
    """
    
    results = []
    
    for param in tqdm_notebook(parameters_list):
        try: 
            model = SARIMAX(exog, order=(param[0], d, param[1]), seasonal_order=(param[2], D, param[3], s)).fit(disp=-1)
        except:
            continue
            
        aic = model.aic
        results.append([param, aic])
        
    result_df = pd.DataFrame(results)
    result_df.columns = ['(p,q)x(P,Q)', 'AIC']
    #Sort in ascending order, lower AIC is better
    result_df = result_df.sort_values(by='AIC', ascending=True).reset_index(drop=True)
    
    return result_df
from statsmodels.tsa.stattools import adfuller
def test_stationarity(timeseries):

    #Determing rolling statistics
    rolmean = timeseries.rolling( window=12).mean()
    rolstd = timeseries.rolling( window=12).std()

    #Plot rolling statistics:
    
    
    #Perform Dickey-Fuller test:
    #print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    #check if test stat is within limits
    if dfoutput['Critical Value (5%)'] > dfoutput['Test Statistic']:
        fig = plt.figure(figsize=(15, 5))
        orig = plt.plot(timeseries, color='blue',label='Original')
        mean = plt.plot(rolmean, color='red', label='Rolling Mean')
        std = plt.plot(rolstd, color='black', label = 'Rolling Std')
        plt.legend(loc='best')
        plt.title('Rolling Mean & Standard Deviation')
        plt.show()
        return True,dfoutput
    else:
        return False
    
for pest in df.Pest.unique():# df.pest.value_counts().index:
    try:
        
        data = df[df['Pest'] == pest]
        data.drop(['Pest','Dist Name'],axis=1,inplace=True)
        data = data.groupby(['CreatedOn'])['pest_count'].sum().reset_index()
        data.set_index(['CreatedOn'],inplace=True)
        data.columns = ['data']
        plt.figure(figsize=[15, 7.5]); # Set dimensions for figure
        plt.plot(data['data'])
        plt.title("Log Difference of Quarterly frequency for "+ pest)
        plt.show()
        p_range = range(0, 12, 1)
        d = 1
        q_range = range(0, 12, 1)
        P_range = range(0, 12, 1)
        D = 1
        Q_range = range(0, 12, 1)
        s_range = range(120, 360, 30)
        parameters = product(p_range, q_range, P_range, Q_range,s_range)
        parameters_list = list(parameters)
        print(len(parameters_list))        
        test_stationarity(data)
        data.dropna(inplace=True)
        last_mean = data['data'].mean()
        for parameter in parameters_list:
            p= int(parameter[0])
            q= int(parameter[1])
            P= int(parameter[2])
            Q =int(parameter[3])
            s = int(parameter[4])
            data['data'] = data.values
            data['data'] = np.log(data['data'])     
            data['data'] = data['data'].diff(s) 
            data = data[~data.isin([np.nan, np.inf, -np.inf]).any(1)]
            condition,dftest = test_stationarity(data)
            if condition:
                results = SARIMAX(data['data'], order=(p, d, q), seasonal_order=(P, D, Q, dftest['#Lags Used'])).fit()                
                forecast_values = results.get_forecast(steps=9)
                forecast_ci = forecast_values.conf_int()
                forecast_ci.index = pd.Series(pd.date_range(data.index[-1], freq="M", periods=9))            
                forecast_values.predicted_mean = forecast_ci.mean(axis=1)
                ax = data.plot()
                forecast_values.predicted_mean.plot(ax=ax,label = 'Forecasts')
                ax.fill_between(forecast_ci.index,forecast_ci.iloc[:,0],forecast_ci.iloc[:,1],color='g',alpha=0.5)
                ax.set_xlabel('Time')
                ax.set_ylabel('queries')
                plt.legend()
                plt.title(pest)
                plt.savefig("result/pre_"+pest+'.png')
                plt.show()
                break
                
    except Exception as ex:
            print("found some exceptations",ex)
            continue
            
    
    





'''plt.figure(); # Set dimensions for figure
plt.plot(df.index,df.pest_count)

plt.title('Quarterly EPS for Johnson & Johnson')
plt.ylabel('EPS per share ($)')
plt.xlabel('Date')
plt.xticks(rotation=90)
plt.grid(True)
plt.show()

plot_pacf(df);
plot_acf(df);

ad_fuller_result = adfuller(df)
print(f'ADF Statistic: {ad_fuller_result[0]}')
print(f'p-value: {ad_fuller_result[1]}')'''




'''ad_fuller_result = adfuller(data['data'])
print(f'ADF Statistic: {ad_fuller_result[0]}')
print(f'p-value: {ad_fuller_result[1]}')

plot_pacf(data['data']);
plot_acf(data['data']);'''




