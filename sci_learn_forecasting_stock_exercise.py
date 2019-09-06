

import pandas as pd
import matplotlib.pyplot as plot
import numpy as np
from numpy import array
from sklearn import linear_model as LM
from sklearn.model_selection import train_test_split

x = []
y = []

#filling my x and y lists with dataframe Adj close data , and appending the number of days to the y list, and then transforming them to a matrix with numpy


stock = pd.read_csv('https://raw.githubusercontent.com/beccarebecca/machine-learning-tools-sci-learn-assignment/master/AAPL(3).csv')
for ind in stock.index: 
     y.append(float(stock['Adj Close'][ind])) 
     x.append(ind + 1)

x = array(x)
#x = np.reshape(x,(len(x), 1))
#y = np.reshape(y, (len(y),1))
# reshape
x = x.reshape((x.shape[0], 1))

y = array(y)

# reshape
y = y.reshape((y.shape[0], 1))



#splitting my x and y data into training, and testing (for perventing overfitting)

xTrain, xTest, yTrain, yTest = train_test_split(x,y, test_size = .25, random_state = 0)

# ordinary least squares regression training, and prediction

reg = LM.LinearRegression()

reg.fit(xTrain,yTrain)

LR_forecast = reg.predict(xTest)

#ridgeCV regression training,and prediction
rgCV = LM.RidgeCV()

rgCV.fit(xTrain,yTrain)

ridge_forecast = rgCV.predict(xTest)

# lassoLars regression training, and prediction
lassoL = LM.LassoLars()

lassoL.fit(xTrain,yTrain)

lasso_forecast = lassoL.predict(xTest)

#visual view to the Apple stock prediction against the actual outcome

plot.scatter(xTest, yTest, color = 'blue')
plot.plot(xTest, LR_forecast, color ='red')
plot.plot(xTest,lasso_forecast, color ='green')
plot.plot(xTest,ridge_forecast, color ='hotPink')
plot.title('Apple Stock Forecast (assignment)') 
plot.xlabel('Day of the Year')
plot.ylabel('Adjusted Stock At Close')
plot.linewidth=4
plot.show()

#score of the most accurate linear model, and percentage

linear_regressionForecast= {reg.score(xTest, yTest):'lr_leastsquares',rgCV.score(xTest, yTest):'ridgeCV',lassoL.score(xTest,yTest):'lasso lars'}
lrModel, accur = max(zip(linear_regressionForecast.values(), linear_regressionForecast.keys()))
print("The linear model that preformed the best in my forecast program was the",lrModel, "at a percent accuacy of" , accur, ".")
print(linear_regressionForecast)

