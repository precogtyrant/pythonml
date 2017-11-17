import re
import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
data = pd.read_csv("test.csv")
data['DOB'] = data['DOB'].apply(lambda x : re.sub('.*/','',x))
gender = {"Male":0,"Female":1}
data['Sex'] = data['Sex'].apply(lambda x : gender[x])
#print(data.filter(["Age","Pay Rate","DOB","Sex"]))


#self implementation
def linear_reg(x_list,y_list):
	x_sum = sum(x_list)
	y_sum = sum(y_list)
	x_multiplied_y_sum = sum([x_list*y_list for x_list,y_list in zip(x_list,y_list)])
	sum_x_multiplied_sum_y = x_sum*y_sum
	n = len(x_list)
	x_square_sum = sum(map(lambda x: x**2, x_list))
	y_square_sum = sum(map(lambda y: y**2, y_list))
	zip(x_list,y_list)

	coeff_a = ((y_sum*x_square_sum) - (x_sum*x_multiplied_y_sum))/((n*x_square_sum) - x_sum**2)
	coeff_b = ((n*x_multiplied_y_sum) - (x_sum*y_sum))/((n*x_square_sum) - x_sum**2)
	return [coeff_a,coeff_b]

age = list(data["Age"])
pay_rate = list(data["Pay Rate"])
print(linear_reg(age,pay_rate))



#implementation using sklearn
model = linear_model.LinearRegression()
age = data['Age']
pay_rate =data['Pay Rate']
d = model.fit(age[:,None],pay_rate)
print(d.coef_)
print(d.intercept_)



