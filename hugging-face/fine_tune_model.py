#!/usr/bin/env python

"""
Fine Tuning emotion text classifier with Friends data 

Based on official tutorial
"""

from transformers import AutoTokenizer
from datasets import load_dataset, load_metric
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
import numpy as np
import pandas as pd
import json

# Load the dataset
# convert friends json to dataframe
#df = pd.read_json("friends.json")

with open("friends.json") as f:
    data = json.load(f)

# convert to dataframe
df = pd.DataFrame(data)
#print(df.head())
df = df.stack().reset_index()
df = df.drop(['level_0', 'level_1'], axis=1)
print(df) 

# get utterance and label


# transpose dataframe
#transposed = df[1:].T
#print(transposed)

#print(df.iloc[:, :1])
# add first column with transposed
#final_df = df.iloc[:, :1].append(transposed, ignore_index=True)
#print(final_df.head())
