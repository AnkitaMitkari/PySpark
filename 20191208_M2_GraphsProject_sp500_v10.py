# Databricks notebook source
# MAGIC %md ## Project Objective 
# MAGIC  - Stock market data can be interesting to analyze and as a further incentive, strong predictive models can have large financial payoff. The amount of financial data on the web is seemingly endless. A    large and well structured dataset on a wide array of companies can be hard to come by. In this project I have used a dataset with historical stock prices (5 years) for all companies           currently found on the S&P 500 index.
# MAGIC   Action Items -
# MAGIC   - In this project I have run multiple regression models to identify the best model to fit the dataset.  
# MAGIC    - I have downloaded the S&P 500 stocks from 2013 to 2018
# MAGIC    - Performed Exploratory Data Analysis (EDA) on the Facebook, Amazon, Apple, Netflix and Google (FAANG) stocks
# MAGIC    - I have plotted the yearly performance for the FAANG stocks
# MAGIC    - The key parameters have been explained using graphical representation of data.
# MAGIC   - I have run multiple regression models and compared them to find the best model for this dataset
# MAGIC    - For the best model I evaluated on a test dataset and report the r2 value and represent it visually.

# COMMAND ----------

try:
  displayHTML('<img src="https://media.giphy.com/media/10MEBgSHIhglMY/giphy.gif"/>') 
  #displayHTML('<img src="https://media.giphy.com/media/ycBC2muuv1yBlCL6Mn/giphy.gif"/>') 
except Exception as e:
  print("Gif load failed")

# COMMAND ----------

# MAGIC %md ###Installing Libraries that we are going to use in this notebook

# COMMAND ----------

# MAGIC %sh
# MAGIC pip install plotly

# COMMAND ----------

import plotly.graph_objects as go
import plotly as py
from pyspark.sql.functions import unix_timestamp
import numpy as np

# COMMAND ----------

# MAGIC %md ### Download the data for S&P 500 stocks from 2013 to 2018 
# MAGIC  - Data source : Downloaded dataset from Kaggle.
# MAGIC  link - https://www.kaggle.com/camnugent/sandp500/download
# MAGIC  - Alternatively, same data can be scraped from google Finance.
# MAGIC  link - https://www.google.com/finance

# COMMAND ----------

# MAGIC %sh
# MAGIC wget 'https://www.dropbox.com/s/f2oetiwm4iswhus/all_stocks_5yr.csv?dl=0' -O sp500faang.csv
# MAGIC   

# COMMAND ----------

# MAGIC %sh
# MAGIC ls  #check the files in the directory

# COMMAND ----------

# MAGIC %md ## Data conversion and cleaning 
# MAGIC - Converting the downloaded csv file into pyspark dataframe
# MAGIC - Converting Date column to unix_timestamp
# MAGIC - Dropping the Null values
# MAGIC - Preparing the data by filtering the data for each FAANG stocks store in seperate dataframes

# COMMAND ----------

df_data = spark.read\
  .format("csv")\
  .option('header', 'true')\
  .option('inferSchema', 'true')\
  .load("file:/databricks/driver/sp500faang.csv")
df_data.show(3)

#test1.show(1)

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.functions import date_format
from pyspark.sql.functions import col
from pyspark.sql.functions import udf,desc


#Creating a pyspark dataframe for future use
AAPLdf2 = df_data.where("Name = 'AAPL'")
AAPLdf3 = AAPLdf2.withColumn("date_converted", unix_timestamp("date", "yyyy-MM-dd").cast("double"))

split_col = F.split(AAPLdf3['date'], '-')
split_col2 = F.split(split_col.getItem(2),' ') 

startyear = 2013
AAPLdf3 = AAPLdf3.withColumn('DateofMon', split_col2.getItem(0))
AAPLdf3 = AAPLdf3.withColumn('Month', split_col.getItem(1))
AAPLdf3 = AAPLdf3.withColumn('Year', split_col.getItem(0)-startyear)

AAPLdf3 = AAPLdf3.withColumn('WeekSeq', ((AAPLdf3.Year)*12*4 + (AAPLdf3.Month-2)*4 + (AAPLdf3.DateofMon)/7 ).cast("int") )

#AAPLdf3 = AAPLdf3.withColumn('DayofWeek', ((AAPLdf3.DateofMon)%7).cast("int") )
#weekDay =  udf(lambda x: datetime.strptime(x, '%Y-%m-%d').strftime('%w'))
#AAPLdf3 = AAPLdf3.withColumn('DayofWeek', weekDay(AAPLdf3['date']))

AAPLdf3.show(3)    


# COMMAND ----------

df_data.printSchema()


# COMMAND ----------

# MAGIC %md Converting the date field to unix_timestamp format for feeding the data to the fitting model

# COMMAND ----------



df2 = df_data.withColumn("date_converted", unix_timestamp("date", "yyyy-MM-dd").cast("double"))
df2 = df2.withColumn("date_unix", unix_timestamp("date", "yyyy-MM-dd"))


#df.withColumn('total_col', df.a + df.b + df.c)

#split_date= split(df2['date'], '-')     
#df2= df2.withColumn('Year', split_date.getItem(0))
#df2= df2.withColumn('Month', split_date.getItem(1))
#df2= df2.withColumn('Day', split_date.getItem(2))

#df345 = df2.select("some_id", year(df2["date"]).alias('year'), month(df2["date"]).alias('month'), dayofmonth(df2["date"]).alias('day'),hour(df2["date"]).alias('hour')).show()
#df345.show(5)


# COMMAND ----------

# MAGIC   %md 
# MAGIC   ### Data cleaning 
# MAGIC   * Identifying the null values and dropping them
# MAGIC   * There were a few rows with null values that were dropped

# COMMAND ----------

df2.count()

# COMMAND ----------

df2 = df2.toPandas()
df2.isnull().sum()

# COMMAND ----------

df2 = df2.dropna()
df2.count()

# COMMAND ----------

# MAGIC %md # Data Explained
# MAGIC 
# MAGIC  We have filtered the data for each FAANG stocks , stored it in seperate dataframe and compared the performance of these stocks on yearly basis.
# MAGIC  
# MAGIC All the dataframes have the following columns: 
# MAGIC 
# MAGIC ## Key fields - 
# MAGIC - Key fields - These are the features from the original dataset
# MAGIC * date - in format: yyyy-mm-dd
# MAGIC * open - price of the stock at market open (this is NYSE data so all in USD)
# MAGIC * high - Highest price reached in the day
# MAGIC * low  - Lowest price reached in the day
# MAGIC * close - price of the stock at market close (in USD) 
# MAGIC * volume - Number of shares traded
# MAGIC * Name - the stock's ticker name
# MAGIC 
# MAGIC Engineered Features
# MAGIC * We engineered the following features to improve our model. 
# MAGIC * avgof7D_lag3D   - running average of 7 Day with a lag of 3 days ( calculated from open price)
# MAGIC * avgof14D_lag7D  - running average of 14 Day with a lag of 7 days ( calculated from open price)
# MAGIC * avgof28D_lag14D - running average of 28 Day with a lag of 14 days ( calculated from open price)
# MAGIC * DateofMon - the date of a month as a number
# MAGIC * Month - the month as a number
# MAGIC * Year - the year as a number with Year 2013 as year 0. (at the start of the time line.)
# MAGIC * WeekSeq  - the week number starting with week 0 as the start of the time line. 
# MAGIC * date_converted - Since we are using this as a time series analysis, we are converting the date into a double format.  
# MAGIC 
# MAGIC ## Predictor field - 
# MAGIC 
# MAGIC * We added the feature "avgof14D_lag0D" from the open value and we are trying to predict this smoothened value
# MAGIC - avgof14D_lag0D   - running average of 7 Day with a lag of 0 days ( calculated from open price)
# MAGIC 
# MAGIC We ended up using only the feature "14 Days with a lag of 7 days" among all the lag features, since we found that to be the most useful feature to get the best R2 value
# MAGIC 
# MAGIC ## Detailed explanation of the reason to engineer new features
# MAGIC 
# MAGIC - In m1, we used randomsplit to split data into training and testing. This gave us a good r2 value.
# MAGIC - At that point we were not predicting. When we split data with time; the r2 value was close to 0.2 .( Poor fitting ) 
# MAGIC * * Training data: from 2013 to mid 2017 and testing data: from 2017 mid to 2018 mid; 
# MAGIC - This meant our model over trained on train data and was not doing well in predicting future (test data)
# MAGIC - We thought, the reason is the fluctuation of the stock price from day to day, the model was getting over trained.
# MAGIC - To mitigate this, we decided to take a moving average filter.
# MAGIC - To take moving average filter we used the "window" functionality in the pyspark.sql 
# MAGIC - We tried moving average filter for 7 days, 14 days and 28 days
# MAGIC - For our final results 14 days average was working better for us, so we used that here.
# MAGIC - We also added the following features. Dateofmonth, Month, Year, weekseq, date_converted
# MAGIC - The idea of using the date, month and year was to let the model develop and detect any patterns from monthly, weekly or yearly patterns. 
# MAGIC - These features helped improve the R2 model but only very slightly. ( 0.83 for APPL went to 0.85). 
# MAGIC - For predicting the stock values what really helped us was adding the stock values from 7 days back as a feature. This was from our domain knowledge of stock market, which is the essence of feature engineering.

# COMMAND ----------

## filter the data for each FAANG stocks store in seperate dataframes
AAPL = df2[(df2['Name'] == 'AAPL')]
GOOGL = df2[(df2['Name'] == 'GOOGL')]
FB = df2[(df2['Name'] == 'FB')]
AMZN = df2[(df2['Name'] == 'AMZN')]
NFLX = df2[(df2['Name'] == 'NFLX')]

# COMMAND ----------

# MAGIC %md # Visualization of data

# COMMAND ----------

#Plotting yearly performance of the FAANG stocks
import matplotlib.pyplot as plt  
from pylab import rcParams
import seaborn as sns  
rcParams['figure.figsize'] = 10,10
fig = plt.figure()
#ax1 = fig.add_subplot(221)
ax1 = sns.lineplot(data=AMZN, x="date", y="open",label ='APPLE Stock Movement', markers=True, dashes=False)
ax1 = sns.lineplot(data=GOOGL, x="date", y="open",label ='GOOGLE Stock Movement', markers=True, dashes=False)
ax1 = sns.lineplot(data=AAPL, x="date", y="open",label ='APPLE Stock Movement', markers=True, dashes=False)
ax1 = sns.lineplot(data=FB, x="date", y="open",label ='FACEBOOK Stock Movement', markers=True, dashes=False)
ax1 = sns.lineplot(data=NFLX, x="date", y="open",label ='NETFLIX Stock Movement', markers=True, dashes=False)
fig.suptitle('Comparing Yearly Performance of the FAANG Stocks', fontsize=18)
display(fig)

# COMMAND ----------

import matplotlib.pyplot as plt  
from pylab import rcParams
import seaborn as sns  
rcParams['figure.figsize'] = 10,7
fig = plt.figure()
#ax1 = fig.add_subplot(221)
ax1 = sns.lineplot(data=AMZN, x="date", y="open",label ='APPLE Stock Movement', markers=True, dashes=False)
ax1 = sns.lineplot(data=GOOGL, x="date", y="open",label ='GOOGLE Stock Movement', markers=True, dashes=False)
fig.suptitle('Comparing Yearly Performance high performing Stocks - Google and Apple', fontsize=18)
display(fig)

# COMMAND ----------

import matplotlib.pyplot as plt  
from pylab import rcParams
import seaborn as sns  
rcParams['figure.figsize'] = 10,7
fig = plt.figure()
#ax1 = fig.add_subplot(221)
ax1 = sns.lineplot(data=AAPL, x="date", y="open",label ='APPLE Stock Movement', markers=True, dashes=False)
fig.suptitle('Yearly Performance of Apple stock of last 5 years', fontsize=18)
display(fig)

# COMMAND ----------

# MAGIC %md ###Plotting data from different fields against 'date' parameter for Apple Stock to get a  better understanding of the fields.

# COMMAND ----------

from pylab import rcParams
rcParams['figure.figsize'] = 20, 10
fig = plt.figure()
ax1 = fig.add_subplot(221)
ax1 = sns.lineplot(x="date", y="open",
                  markers=True, dashes=False,label ='APPL', data=AAPL)
ax2 = fig.add_subplot(222, sharex=ax1, sharey=ax1)
ax2 = sns.lineplot(x="date", y="high",
                  markers=True, dashes=False,label ='APPL',data=AAPL)
ax3 = fig.add_subplot(223, sharex=ax1, sharey=ax1)
ax3 = sns.lineplot(x="date", y="low",
                  markers=True, dashes=False,label ='APPL', data=AAPL)
ax4 = fig.add_subplot(224, sharex=ax1, sharey=ax1)
ax4 = sns.lineplot(x="date", y="close",
                  markers=True, dashes=False,label ='APPL', data=AAPL)

display(fig)

# COMMAND ----------

# Taking a closer look at the Apple Stock trend from first week of January to first week of February in 2018
apple2018 = AAPL[(AAPL['date'] > '2018-01-05' )]
apple_m = apple2018.groupby(['date'])
agg_m = apple2018.aggregate({'open':np.mean, 'close':np.mean, 'high':np.mean, 'low':np.mean})
agg_m = apple2018.reset_index()

trace = go.Candlestick(x = agg_m['date'],
                       open = agg_m['open'],
                       high = agg_m['high'],
                       low = agg_m['low'],
                       close = agg_m['close']
                      )

data = [trace]

layout = {
    'title':'Closer look at Apple Stock trend',
    'xaxis': {'title':'Date',
             'rangeslider':{'visible':False}},
    'yaxis':{'title':'Price in US Dollars'}
}

fig_candle1 = go.Figure(data=data, layout=layout)
py.offline.iplot(fig_candle1)

# COMMAND ----------

# MAGIC %md  Insights : The performance of the stock is seen to be improving during the second week of January 2018. Where as it gradually plummets towrards the end of the month.

# COMMAND ----------

# MAGIC %md ##Feature Engineering
# MAGIC * Added more features from existing features
# MAGIC * We added the following features 
# MAGIC - avgof7D_lag3D   - running average of 7 Day with a lag of 3 days
# MAGIC - avgof14D_lag7D  - running average of 14 Day with a lag of 7 days
# MAGIC - avgof28D_lag14D - running average of 28 Day with a lag of 14 days
# MAGIC 
# MAGIC We ended up using only the 14 Days with a lag of 7 days, since we found that to be the most useful feature to get the best R2 value

# COMMAND ----------

# SMOOTHEN DATA
from pyspark.sql import functions as F
from pyspark.sql.window import Window

days = lambda i: i * 86400 # convert days to seconds. This is used in rangeBetween
df4 = AAPLdf3.withColumn('date', AAPLdf3.date.cast('timestamp'))

#create window by casting timestamp to long (number of seconds)
w7 = (Window.orderBy(F.col("date").cast('long')).rangeBetween(-days(7), 0))
w14 = (Window.orderBy(F.col("date").cast('long')).rangeBetween(-days(14), 0))
w28 = (Window.orderBy(F.col("date").cast('long')).rangeBetween(-days(28), 0))

df5 = df4.withColumn('avgof7D_lag3D', F.avg("open").over(w7))
df6 = df5.withColumn('avgof14D_lag7D', F.avg("open").over(w14))
df7 = df6.withColumn('avgof28D_lag14D', F.avg("open").over(w28))

df7.show(5)

# IN M1, WE USED RANDOMSPLIT TO SPLIT DATA INTO TRAINING AND TESTING
# THIS GAVE US A GOOD R2 VALUE.
# AT THAT POINT WE WERE NOT PREDICTING. 
# WHEN WE SPLIT DATA WITH TIME. TRAINING DATA: FROM 2013 TO MID 2017 AND TESTING DATA: FROM 2017 MID TO 2018 MID; 
# THE R2 VALUE WAS CLOSE TO 0.1 

# THIS MEANT OUR MODEL OVER TRAINED ON TRAIN DATA AND WAS NOT DOING WELL IN PREDICTING FUTURE (TEST DATA)
# WE THOUGHT, THE REASON IS THE FLUCTUATION OF THE STOCK PRICE FROM DAY TO DAY, THE MODEL WAS GETTING OVER TRAINED.

# TO MITIGATE THIS, WE DECIDED TO TAKE A MOVING AVERAGE FILTER
# TO TAKE MOVING AVERAGE FILTER WE USED THE "WINDOW" FUNCTIONALITY IN THE PYSPARK.SQL 
# WE TRIED MOVING AVERAGE FILTER FOR 7 DAYS, 14 DAYS AND 28 DAYS
# FOR OUR FINAL RESULTS 14 DAYS AVERAGE WAS WORKING BETTER FOR US, SO WE USED THAT HERE. 

# THE MOVING AVERAGE FILTER SMOOTHENED OUT THE DATA, BUT WE WERE STILL GETTING LOW R2 VALUE IN THE TEST DATA
# TO IMPROVE THE RESULT FURTHER WE THOUGHT OF SOME NEW IDEAS. 
# WE TRIED TO GET A DATASET WITH FINANCIAL NEWS, CONVERT TO SENTIMENT ANALYSIS ETC. 
# (DATA WAS AVAILBALE BUT IT WAS EXPENSIVE $125/MONTH SUBSCRIPTION WAS THE BEST WE COULD FIND. - EVEN FOR HISTORICAL DATA)
# NEXT WE TRIED TO BE MORE CREATIVE WITH FEATURE ENGINEERING, WITH DOMAIN KNOWLEDGE

# COMMAND ----------

# INTRODUCE LAG TO DATA SO THAT OLD Y value CAN BE USED AS A FEATURE FOR FUTURE
# SMOOTHENING MOVES DATA TO FUTURE BY (N+1)/2 - standard fitler delay 

from pyspark.sql.functions import monotonically_increasing_id, lag
from pyspark.sql.window import Window

# Add ID to be used by the window function
df7 = df7.withColumn('id', monotonically_increasing_id())
# Set the window
w = Window.orderBy("id")
# Create the lagged value
value_lag = lag('avgof14D_lag7D',-7).over(w)
# Add the lagged values to a new column
df7 = df7.withColumn('avgof14D_lag0D', value_lag).dropna()
df7.show(15)

df_stg2 = df7

# TO IMPROVE OUR RESULTS WE DECIDED TO USE OUR DOMAIN KNOWLEDGE IN STOCK MARKET PREDICTION 
# THE BEST STOCK MARKET MODELS USE A TECHNIQUE CALLED LSTM (Long short-term memory), A "RECURRENT NEURAL NETWORK"
# WE COULD NOT FIGURE OUT A WAY TO USE LSTM WITH PYSPARK PIPELINE

# TO GET AROUND THIS 
# FOR PREDICITNG FUTURE STOCK VALUES, WE USE THE STOCK VALUES FROM 7 DAYS BACK AS A FEATURE
# WE USED A LAG FUNCTION TO CREATE A NEW FEATURE THAT DELAYS THE CURRENT DATA BY 7 DAYS. 
# FOR MORE EXPLANATION, LET US SHOW YOU TWO FIGURES
# FIRST ONE IS THE WHOLE DATA AND THEN THE ZOOMED IN VERSION 


# COMMAND ----------

import matplotlib.pyplot as plt  
from pylab import rcParams
import seaborn as sns  

pdata = df_stg2.toPandas()

rcParams['figure.figsize'] = 25, 10
fig = plt.figure()

ax1 = sns.lineplot(data=pdata, x="date", y="open", markers=True, dashes=False,label ='raw open')
#ax1 = sns.lineplot(data=pdata, x="date", y="avg7", markers=True, dashes=False)
ax1 = sns.lineplot(data=pdata, x="date", y="avgof14D_lag0D", markers=True, dashes=False,label ='avgof14D_lag0D')
#ax1 = sns.lineplot(data=pdata, x="date", y="avg28", markers=True, dashes=False)
ax1 = sns.lineplot(data=pdata, x="date", y="avgof14D_lag7D", markers=True, dashes=False,label ='avgof14D_lag7D')
display(fig)


# COMMAND ----------

import matplotlib.pyplot as plt  
from pylab import rcParams
import seaborn as sns  

data_zoom = df_stg2[(df_stg2['date'] > '2017-01-01' )]

pdata = data_zoom.toPandas()

rcParams['figure.figsize'] = 25, 10
fig = plt.figure()

ax1 = sns.lineplot(data=pdata, x="date", y="open", markers=True, dashes=False,label ='raw open')
#ax1 = sns.lineplot(data=pdata, x="date", y="avg7", markers=True, dashes=False)
ax1 = sns.lineplot(data=pdata, x="date", y="avgof14D_lag0D", markers=True, dashes=False,label ='avgof14D_lag0D')
#ax1 = sns.lineplot(data=pdata, x="date", y="avg28", markers=True, dashes=False)
ax1 = sns.lineplot(data=pdata, x="date", y="avgof14D_lag7D", markers=True, dashes=False,label ='avgof14D_lag7D')
display(fig)

# THIS IS THE ZOOMED IN VERSION, THE BLUE IS THE RAW VALUES 
# THE ORANGE IS THE SMOOTHENED VERSION OF RAW ( SMOOTHENED OVER 14 DAYS) - "avgof14D_lag0D"
# WE ARE PREDICTING THIS NO LAG "avgof14D_lag0D"

# WE USE DATA FROM 7 DAYS BACK AS FEATURE "avgof14D_lag7D" ( delayed data)
# THIS REALLY HELPED US IMPROVE OUR R2 VALUE TO A RESPECTABLE 0.83 FOR APPL, 0.87 FOR FB, 0.93 FOR AMAZON, 0.85 FOR FB, AND 0.86 FOR GOOGLE. 
# WE RAN THE NOTE BOOK MULTIPLE TIMES TO GET THESE R2 VALUES. ( WITH DIFFERENT COMAPNIES DATAFRAME)

# COMMAND ----------

# MODEL CREATION IS DONE USING TRAINING DATA ONLY. 
# TEST RESULTS ARE CALCULATED BASED ON TEST DATA ONLY
from pyspark.sql.functions import unix_timestamp, lit

train_data = df_stg2.filter(df_stg2["date"] < unix_timestamp(lit('2017-07-01 00:00:00')).cast('timestamp'))
test_data  = df_stg2.filter(df_stg2["date"] > unix_timestamp(lit('2017-07-01 00:00:00')).cast('timestamp'))

print("Number of training records: " + str(train_data.count()))
print("Number of testing records : " + str(test_data.count()))


# COMMAND ----------

# MAGIC %md ##Data modeling - Testing with one model
# MAGIC * Removed label and undesired columns
# MAGIC * Formatted data to fit RFormula format
# MAGIC * Tested pipeline and paramgrid for a single linear regression model
# MAGIC * CVfit with 3 fold cross validation
# MAGIC * Formula : avgof14D_lag0D ~ volume + date_converted + DateofMon + Month + Year + WeekSeq + avgof14D_lag7D

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler,RFormula
from pyspark.ml.regression import LinearRegression, GeneralizedLinearRegression, DecisionTreeRegressor
from pyspark.ml import Pipeline, Model
from pyspark.ml.evaluation import RegressionEvaluator 
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder 

columns = train_data.columns
# Not using Price (label) or address in features
columns.remove('date')
columns.remove('open')
columns.remove('high')
columns.remove('low')
columns.remove('close')
columns.remove('Name')
#columns.remove('market_value')
columns.remove('avgof7D_lag3D')
columns.remove('avgof14D_lag0D')
columns.remove('avgof28D_lag14D')
columns.remove('id')

formula = "{} ~ {}".format("avgof14D_lag0D", " + ".join(columns))
print("Formula : {}".format(formula))
rformula = RFormula(formula = formula)
lr = LinearRegression()
pipeline = Pipeline(stages=[rformula, lr])
# Parameter grid
paramGrid = ParamGridBuilder()\
          .addGrid(lr.regParam,[0.01, .04])\
          .build()
cv = CrossValidator()\
      .setEstimator(pipeline)\
      .setEvaluator(RegressionEvaluator()\
                       .setMetricName("r2"))\
      .setEstimatorParamMaps(paramGrid)\
      .setNumFolds(3)

cvModel = cv.fit(train_data)


# COMMAND ----------

# MAGIC %md ## Setting up Pipeline for Multiple regression models
# MAGIC * Removed all columns other than converted date and volume
# MAGIC * Created framework for the models 
# MAGIC * * Linear regression, 
# MAGIC * * Generalized Linear regression and 
# MAGIC * * Decision tree regression
# MAGIC * Paramgrid setup for all three models
# MAGIC * Runs crossvalidation (cvfit) with 3-fold for all three models

# COMMAND ----------

#references
# https://stackoverflow.com/questions/45697720/with-pyspark-sql-functions-unix-timestamp-get-null
from pyspark.ml.feature import VectorAssembler,RFormula
from pyspark.ml.regression import LinearRegression, GeneralizedLinearRegression, DecisionTreeRegressor
from pyspark.ml import Pipeline, Model
from pyspark.ml.evaluation import RegressionEvaluator 
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder 

columns = train_data.columns
# Not using Price (label) or address in features
columns.remove('date')
columns.remove('open')
columns.remove('high')
columns.remove('low')
columns.remove('close')
columns.remove('Name')
#columns.remove('market_value')
columns.remove('avgof7D_lag3D')
columns.remove('avgof14D_lag0D')
columns.remove('avgof28D_lag14D')
columns.remove('id')

# WE ARE PREDICTING "avgof14D_lag0D"
# OUR FORMULA FOR PIPELINE IS Formula : avgof14D_lag0D ~ volume + date_converted + avgof14D_lag7D
# WE ARE USING VOLUME, DATE_CONVERETED ( WHICH IS DATE IN SECONDS), AND THE LAG DATA. TO PREDICT THE FUTURE VALUES

formula = "{} ~ {}".format("avgof14D_lag0D", " + ".join(columns))
print("Formula : {}".format(formula))
rformula = RFormula(formula = formula)

# Pipeline basic to be shared across model fitting and testing
pipeline = Pipeline(stages=[])  # Must initialize with empty list!

# base pipeline (the processing here should be reused across pipelines)
basePipeline =[rformula]

#############################################################
# Specify Linear Regression model
lr = LinearRegression()
pl_lr = basePipeline + [lr]
pg_lr = ParamGridBuilder()\
          .baseOn({pipeline.stages: pl_lr})\
          .addGrid(lr.regParam,[0.01, .04])\
          .build()
#############################################################
# Specify Random Forrest model
rf = GeneralizedLinearRegression()
pl_rf = basePipeline + [rf]
pg_rf = ParamGridBuilder()\
      .baseOn({pipeline.stages: pl_rf})\
      .build()

#############################################################
# Specify Decision Tree model
dt = DecisionTreeRegressor()
pl_dt = basePipeline + [dt]
pg_dt = ParamGridBuilder()\
      .baseOn({pipeline.stages: pl_dt})\
      .build()

# One grid from the individual grids
paramGrid = pg_lr + pg_rf + pg_dt

# COMMAND ----------

# The regression metric can be mse, rmse, mae, r2
# See the metrics here https://spark.apache.org/docs/latest/mllib-evaluation-metrics.html#regression-model-evaluation
# Should run more than 3 folds, but here we simplify so that it will complete
cv = CrossValidator()\
      .setEstimator(pipeline)\
      .setEvaluator(RegressionEvaluator()\
                       .setMetricName("r2"))\
      .setEstimatorParamMaps(paramGrid)\
      .setNumFolds(3)

cvModel = cv.fit(train_data)

# COMMAND ----------

import numpy as np
print("Best Model")
print(cvModel.getEstimatorParamMaps()[ np.argmax(cvModel.avgMetrics) ])
print("Worst Model")
print (cvModel.getEstimatorParamMaps()[ np.argmin(cvModel.avgMetrics) ])


# COMMAND ----------

#references
# https://stackoverflow.com/questions/45697720/with-pyspark-sql-functions-unix-timestamp-get-null


# COMMAND ----------

# MAGIC %md ## Model evaluation
# MAGIC * Code necessary to plot r2 value of all three models
# MAGIC * Plotting of r2 value to compare models

# COMMAND ----------

import re
def paramGrid_model_name(m):
  params = [v for v in m.values() if type(v) is not list]
  name = [v[-1] for v in m.values() if type(v) is list][0]
  name = re.match(r'([a-zA-Z]*)', str(name)).groups()[0]
  return "{}{}".format(name,params)

# Resulting metric and model description
# get the measure from the CrossValidator, cvModel.avgMetrics
# get the model name & params from the paramGrid
# put them together here:
measures = zip(cvModel.avgMetrics, [paramGrid_model_name(m) for m in paramGrid])
metrics,model_names = zip(*measures)

# COMMAND ----------

import seaborn as sns
import matplotlib.pyplot as plt

plt.clf() # clear figure
fig = plt.figure( figsize=(5, 5))
plt.style.use('fivethirtyeight')
axis = fig.add_axes([0.1, 0.3, 0.8, 0.6])
# plot the metrics as Y
#plt.plot(range(len(model_names)),metrics)
plt.bar(range(len(model_names)),metrics)
# plot the model name & param as X labels
plt.xticks(range(len(model_names)), model_names, rotation=70, fontsize=6)
plt.yticks(fontsize=6)
#plt.xlabel('model',fontsize=8)
plt.ylabel('R2 (higher is better)',fontsize=8)
plt.title('Model evaluations')
display(plt.show())

# COMMAND ----------

# MAGIC %md ##Insight 
# MAGIC * Visual representation of comparison regression model based on r2 values. Out of all the models Decision Tree is identified as best model that fits this dataset.
# MAGIC * Based on the r2 value, all the three models were able to do well on the training data. 

# COMMAND ----------

# MAGIC %md ##Testing the best model
# MAGIC * Create test, train dataset
# MAGIC * Evaluate test data with best model

# COMMAND ----------

## Make predictions on test documents. 
#train_data, test_data  = AAPLdf3.randomSplit([0.6, 0.4], 24)   # proportions [], seed for random
# CrossValidator.fit() is in cvModel, which is the best model found.
predictions = cvModel.transform(test_data)
display(predictions.select('label', 'prediction').limit(100))

# COMMAND ----------

# MAGIC %md ## Best model metrics 
# MAGIC * Display R2 value
# MAGIC * Display RMSE value

# COMMAND ----------

import sklearn.metrics
from pyspark.ml.evaluation import RegressionEvaluator
# Select (prediction, true label) and compute test error
evaluator = RegressionEvaluator(labelCol="label",
                                predictionCol="prediction",
                                metricName="rmse")

# Summarize the model over the training set and print out some metrics
print("Best pipeline", cvModel.bestModel.stages)
print("Best model", cvModel.bestModel.stages[1])

y_true = predictions.select('label').toPandas()
y_pred  = predictions.select('prediction').toPandas()

r2_score = sklearn.metrics.r2_score(y_true, y_pred)
print('r2_score: {0}'.format(r2_score))

rmse = evaluator.evaluate(predictions)
print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)

# COMMAND ----------

# MAGIC %md ## Best model visualizaton ( y_true vs y_pred )
# MAGIC * Plot true values and predicated values
# MAGIC * Since plot is linear, we infer that model is good.

# COMMAND ----------

fig = plt.figure( figsize=(5, 5))
plt.scatter(y_true,y_pred)
plt.xlabel('y_true',fontsize=8)
plt.ylabel('y_pred ',fontsize=8)
plt.title('Best model visualization')
display(fig)

# COMMAND ----------

# MAGIC %md ##Prediction (or discovery)
# MAGIC 
# MAGIC * For the best model I have evaluated on a test dataset and reported the r2 value and represented it visually.
# MAGIC * I USEd DATA FROM 7 DAYS BACK AS FEATURE "avgof14D_lag7D" ( delayed data)
# MAGIC * THIS REALLY HELPED US IMPROVE OUR R2 VALUE TO A RESPECTABLE LEVEL. ( WITHOUT THIS FEATURE OUR R2 value was in the 0.2 to 0.3 range)
# MAGIC * I reported the following R2 VALUES TO A RESPECTABLE 
# MAGIC  * 0.854 FOR APPL,
# MAGIC  * 0.87 FOR FB, 
# MAGIC  * 0.93 FOR AMAZON,
# MAGIC  * 0.85 FOR FB, AND 
# MAGIC  * 0.86 FOR GOOGLE.
# MAGIC * I RAN THE NOTE BOOK MULTIPLE TIMES TO GET THESE R2 VALUES. ( WITH DIFFERENT COMAPNIES DATAFRAME in command 11 )

# COMMAND ----------

# MAGIC %md ##Conclusion
# MAGIC   
# MAGIC    - I have identified the best regression model that fits the dataset for Apple stock over a period of 5 years (2013 -2018)   
# MAGIC    - After comparing the performance of all the models, I identified Decision Tree regression model (depth 5 with 63 nodes) as the best fit model.
# MAGIC    - During the EDA I compared yearly performance of the Facebook, Amazon, Apple, Netflix and Google (FAANG) stocks.
# MAGIC    - Additionally, I compared the two high performing stocks - Apple and google and represented it graphically.
# MAGIC    - The key parameters have been explained using graphical representation of data. 
# MAGIC    - I have compared 'date' against the fields 'open', 'close', 'high' and 'low' for Apple Stock, to get a  better understanding of the fields.
# MAGIC    - I have fit the model with closing stock market price as the dependent variable and, volume and converted_date as the independent variables
# MAGIC    - During model fitting, I encountered data leakage due to high correlation between features 'close' and 'high', 'low', 'open'. Hence, we removed those features from the fit.
# MAGIC    - For the best model I have evaluated on a test dataset and reported the r2 value and represented it visually.
# MAGIC    

# COMMAND ----------

# MAGIC %md ##To pass on "Lessons Learned"
# MAGIC   
# MAGIC    - Predicting future stock market prices is hard. -- some stocks are easier to predict than others
# MAGIC    - Splitting data based on "randomSplit" shows uncharacteristically high R2 values - use time based seperation of train and test data early on
# MAGIC    - Databricks community edition becomes hard to work with when number of rows is around 600,000!
# MAGIC    - Time series analysis helps to a certain extent. ( Adding Lag. PySpark supports "Lag" innately)
# MAGIC    - Adding more features is useful. #( from multiple sources)
# MAGIC    - Moving average of data is essential to create a model. 
# MAGIC    -- Model learned unneccesary details ( daily fluctuations and performed worse in test data.)
# MAGIC    - I could only predict as far as 7-10 days. After that the R2 values of the models started moving to less than 50 percent
# MAGIC    - I ran the model for all FAANG stocks
