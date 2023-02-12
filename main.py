import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import linear_model



datos=pd.read_csv("advertising.csv")
print(datos.isnull().sum())


# Creating dataset
np.random.seed(10)

data_1 = np.array(datos["TV"])
data_2 = np.array(datos["Radio"])
data_3 = np.array(datos["Newspaper"])
data = [data_1, data_2, data_3]

# Creating plot
fig=plt.boxplot(data)
plt.title("grafica de ventas por medio ")
plt.xlabel("medios")
plt.ylabel("ventas")
# show plot
plt.show()

x1=np.array(datos["TV"]).reshape(-1, 1)
x2=np.array(datos["Radio"]).reshape(-1, 1)
x3=np.array(datos["Newspaper"]).reshape(-1, 1)
y=np.array(datos["Sales"])

x_t,x_test,y_t,y_test=train_test_split(x1,y,test_size=0.3,random_state=0)
x_t2,x_test2,y_t2,y_test2=train_test_split(x2,y,test_size=0.3,random_state=0)
x_t3,x_test3,y_t3,y_test3=train_test_split(x2,y,test_size=0.3,random_state=0)

TV = linear_model.LinearRegression()
TV.fit(x_t, y_t)
#print( TV.predict(x_test))
print( TV.coef_)
print( TV.intercept_)

Radio = linear_model.LinearRegression()
Radio.fit(x_t2, y_t2)
#print( Radio.predict(x_test2))
print( Radio.coef_)
print( Radio.intercept_)

Newspaper = linear_model.LinearRegression()
Newspaper.fit(x_t3, y_t3)
#print( Newspaper.predict(x_test3))
print( Newspaper.coef_)
print( Newspaper.intercept_)