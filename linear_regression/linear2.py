from sklearn import linear_model, datasets
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
lin = linear_model.LinearRegression()
data = pd.read_csv("test.csv")
lin.fit(data['Age'][:,None],data['Pay Rate'])
print(lin.predict(10))



# import numpy as np
# from sklearn import linear_model
# x = np.random.rand(100,1)
# y = 3 + 4*x + np.random.rand(100,1)
# reg = linear_model.LinearRegression()
# a = reg.fit(x, y)
# print(a.coef_)
# print(a.intercept_)
