import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rule_miner import RuleMiner

temp_df = pd.read_csv("groceries.csv", header=None)
values = temp_df.values.ravel()
values = [value for value in pd.unique(values) if not pd.isnull(value)]

value_dict = {}
for i, value in enumerate(values):
    value_dict[value] = i
    
print(value_dict)