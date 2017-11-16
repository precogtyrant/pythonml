import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data = pd.read_csv("test.csv")
data['DOB'] = data['DOB'].apply(lambda x : re.sub('.*/','',x))
gender = {"Male":0,"Female":1}
data['Sex'] = data['Sex'].apply(lambda x : gender[x])
print(data.filter(["Age","Pay Rate","DOB","Sex"]))

