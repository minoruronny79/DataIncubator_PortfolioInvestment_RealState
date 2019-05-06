import pandas as pd
import numpy as np
import datetime as dt
import itertools
from scipy import optimize as spo

# For graphs
import matplotlib.pyplot as plt
import seaborn as sns

#For statmodels
import statsmodels.formula.api as smf
from statsmodels.tsa import api as tsa
from statsmodels.tsa import stattools
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
from statsmodels.tsa.stattools import adfuller

#Files
state=pd.read_csv("F:/PythonTraining/Python_CaseStudy/PCS_DataIncubator_PriceHouses_GDP/DB_State.csv", sep=",")
gdp=pd.read_csv("F:/PythonTraining/Python_CaseStudy/PCS_DataIncubator_PriceHouses_GDP/DB_Bea_Quartelygdp.csv", sep=",")


##############################################################################
####1. CROSS SECTIONAL ANALYSIS
##############################################################################

#Converting to datetime
state["Date"]=pd.to_datetime(state.Date)

#Reducing dataframe
houses_nbedrooms=state[["Date", "RegionName", "MedianListingPrice_1Bedroom", 
                        "MedianListingPrice_2Bedroom", "MedianListingPrice_3Bedroom",
                       "MedianListingPrice_4Bedroom", "MedianListingPrice_5BedroomOrMore",
                       "MedianRentalPrice_1Bedroom", "MedianRentalPrice_2Bedroom",
                       "MedianRentalPrice_3Bedroom", "MedianRentalPrice_4Bedroom",
                       "MedianRentalPrice_5BedroomOrMore", "ZHVI_1bedroom", "ZHVI_2bedroom",
                       "ZHVI_3bedroom", "ZHVI_4bedroom", "ZHVI_5BedroomOrMore"]]
houses_nbedrooms.dtypes

#Measure price houses and rent
houses_nbedrooms["value_rent_1bedroom"]=houses_nbedrooms.apply(lambda row: 
                                                               (row.MedianRentalPrice_1Bedroom*12)/
                                                               row.MedianListingPrice_1Bedroom, axis=1)

houses_nbedrooms["value_rent_2bedroom"]=houses_nbedrooms.apply(lambda row: 
                                                               (row.MedianRentalPrice_2Bedroom*12)/
                                                               row.MedianListingPrice_2Bedroom, axis=1)

houses_nbedrooms["value_rent_3bedroom"]=houses_nbedrooms.apply(lambda row: 
                                                               (row.MedianRentalPrice_3Bedroom*12)/
                                                               row.MedianListingPrice_3Bedroom, axis=1)

houses_nbedrooms["value_rent_4bedroom"]=houses_nbedrooms.apply(lambda row: 
                                                               (row.MedianRentalPrice_4Bedroom*12)/
                                                               row.MedianListingPrice_4Bedroom, axis=1)

houses_nbedrooms["value_rent_5bedroomOrMore"]=houses_nbedrooms.apply(lambda row: 
                                                               (row.MedianRentalPrice_5BedroomOrMore*12)/
                                                               row.MedianListingPrice_5BedroomOrMore, axis=1)
#Year from date
houses_nbedrooms2=houses_nbedrooms
houses_nbedrooms2["Year"]=houses_nbedrooms2.Date.dt.year

#Pivot table of number of houses
avghouses_nbeds=pd.pivot_table(houses_nbedrooms2, values=["MedianListingPrice_1Bedroom", 
                                          "MedianListingPrice_2Bedroom",
                                         "MedianListingPrice_3Bedroom",
                                         "MedianListingPrice_4Bedroom",
                                         "MedianListingPrice_5BedroomOrMore"], 
               index=["Year", "RegionName"], aggfunc="median" )

avghouses_nbeds2017=avghouses_nbeds.iloc[1058:1110,:]
#Drop NA's
#avg_houses_nbed2017_b=avghouses_nbeds2017.dropna()
#avg_houses_nbed2017_b.head()



############################################################################
##Heatmaps
avghouses_nbeds2017.drop("UnitedStates", level=1, axis=0, inplace=True)
plt.figure(figsize=(10,10))
sns.heatmap(avghouses_nbeds2017)
plt.xticks(rotation=15)


##############################################################################
####2. TIME SERIES ANALYSIS
##############################################################################
aux_stocks1=houses_nbedrooms
aux_stocks1=aux_stocks1.set_index(["Date","RegionName"])
aux_stocks1=aux_stocks1[['ZHVI_1bedroom', 'ZHVI_2bedroom', 'ZHVI_3bedroom', 'ZHVI_4bedroom', 
                 'ZHVI_5BedroomOrMore']]
aux_stocks1=aux_stocks1.unstack(level="RegionName")
aux_stocks1.head()


aux_stock1_reduced=aux_stocks1.iloc[165:260,:]  #Reduce timeframe
aux_stock1_reduced_b=aux_stock1_reduced.loc[:, aux_stock1_reduced.notnull().sum()>len(aux_stock1_reduced)*.95]


##################################################################
##GENERATING RETURNS
#Generating the return by house
stocks1_final=aux_stock1_reduced_b.pct_change()

#Reindexing from two levels to one level (columns)
stocks1_final.columns = ['_'.join(col) for col in stocks1_final.columns]

stocks1_final
stocks1_final.shape

#Generating top10 assets (By mean)
report_stocks1=stocks1_final.describe()
report_stocks1=report_stocks1.iloc[1:3,:]
report_stocks1=report_stocks1.T.sort_values(by="mean", ascending=False)
report_stocks1=report_stocks1.iloc[0:10,:]
report_stocks1

#Generating a list with names of top10 assets
activos=report_stocks1.T.columns
activos=activos.tolist()
activos

#Graph with Stocks
figure1=plt.figure(figsize=(10,5), dpi=80)
top10=stocks1_final.loc[:,activos] #Selecting columns of top10 assets
plt.plot(top10)
plt.axhline(y=0, color='red', linestyle='-')
# plt.title("Top 10 historic returns")
plt.legend(top10, loc=4, bbox_to_anchor=(0., 1.02, 1., .102), ncol=2)


##############################################################################
####3. OPTIMIZATION AND PREDICTION
##############################################################################

################################################
####3.1 ARIMA Analysis
top10=top10.drop(top10.index[0])
top10.to_csv("top10.csv")

#Sequence of combinations
p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
pdq

#Loop for all assets
bic_arima = [] 
for x in range(len(top10.columns)): 
    for y in pdq: 
        try:
            arima=tsa.ARIMA(top10.iloc[:,x], order=y).fit()
            bic_arima.append((x,y,arima.bic))
        except:
            pass
        
df_bic_arima=pd.DataFrame(bic_arima)
df_bic_arima=df_bic_arima.rename(columns={0:"stock", 1:"arima",2:"bic"})
df_bic_arima.head(15)
selected_rows=df_bic_arima.groupby("stock")[["arima", "bic"]].idxmin()
tabla_arima2=pd.merge(df_bic_arima, selected_rows, left_index=True, right_index=True, how="inner")
top10lista=top10.columns.tolist()
tabla_arima=df_bic_arima.set_index(["stock", "arima"])
tabla_arima=tabla_arima.unstack("stock")
# tabla_arima.columns=tabla_arima.columns.droplevel() #To erase one level of columns
tabla_arima.columns=top10lista 
tabla_arima

#Best ARIMA for top ten stocks
lista_order=tabla_arima2.iloc[:,1]
lista_order=lista_order.tolist()  #Getting arimas as a list
lista_activo=tabla_arima2.index.values.tolist() #Getting the table as a list
activo_order=list(zip(lista_activo, lista_order))
activo_order

arima_reg=[]
for x,y in activo_order:
    try:
        arima=tsa.ARIMA(top10.iloc[:,x], order=y).fit().summary()
        arima_reg.append((x,y,arima))
    except:
        pass
    
#ARIMA ZHVI_3bedroom_DistrictColumbia
DistrictColumbia_3bed=top10.iloc[:,2:3]
DistrictColumbia_3bed.head()

decomp2=seasonal_decompose(DistrictColumbia_3bed)
fig1=decomp2.plot()
XY=DistrictColumbia_3bed.iloc[:,0].values  #Trick to read Dickey Fuller

#Over level
result2=adfuller(XY)
print("HO: There is unit root on level serie")
print("If pvalue is >0.10 there is unit root")
print('2Bedroom Nevada:\nADF Statistic: %f' % result2[0])
print('p-value: %f' % result2[1])
for key, value in result2[4].items():
    print('\t%s: %.3f' % (key, value))
    
arima_Columbia3bed=tsa.ARIMA(DistrictColumbia_3bed, order=(2,1,3)).fit(trend="nc")
arima_Columbia3bed.summary()

fig, ax = plt.subplots(figsize=(9,7))
fig = arima_Columbia3bed.plot_predict(start='2010-03-31', end='2018-12-31', ax=ax)
plt.axhline(y=0, color='green', linestyle='-')
# legend = ax.legend(loc='upper left')

########################################################
#3.2 Vector Autoregresive
Asset_select=top10[['ZHVI_3bedroom_DistrictofColumbia', 'ZHVI_2bedroom_Nevada']]

#Transforming assets in quartely basis
Asset_select_Q=Asset_select.groupby(pd.PeriodIndex(Asset_select.index, freq="Q"), axis=0).mean()
Asset_select_Q2=Asset_select_Q.iloc[0:31,:]

#Selecting GDP
gdp2a=gdp[(gdp["GeoName"]=="United States") | (gdp["GeoName"]=="Nevada") | (gdp["GeoName"]=="District of Columbia")]
gdp2b=gdp2a.iloc[5:6,:]
gdp2c=gdp2a.iloc[29:30,:]
gdp2d=gdp2a.iloc[53:54,:]
gdp2=pd.concat([gdp2b, gdp2c, gdp2d], axis=0)

gdp2
gdp3=gdp2.iloc[:,7:59]
gdp_Q=gdp3.transpose()
gdp_Q.columns=["USA", "District of Columbia", "Nevada"]
gdp_Q=gdp_Q.iloc[21:59,:]

#Joining tables
Asset_select_Q3=Asset_select_Q2.reset_index()
gdp_Q2=gdp_Q.reset_index()
asset_gdp=pd.concat([Asset_select_Q3, gdp_Q2], axis=1)
asset_gdp=asset_gdp.set_index("Date")

##GDP and Asset District of Columbia
asset_gdp_Columbia=asset_gdp.loc[:,["ZHVI_3bedroom_DistrictofColumbia", "District of Columbia"]]
asset_gdp_Columbia["District of Columbia"]=pd.to_numeric(asset_gdp_Columbia["District of Columbia"])
asset_gdp_Columbia_pct=asset_gdp_Columbia.pct_change()
asset_gdp_Columbia_pct.columns=["Asset", "GDP"]
asset_gdp_Columbia_pct

#Model
model1 = tsa.VAR(asset_gdp_Columbia_pct.values[1:,:]).fit()
model1.summary()
model1.plot()
irf=model1.irf(10)
irf.plot()

