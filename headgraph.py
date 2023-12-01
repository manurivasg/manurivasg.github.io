from flask import Flask, render_template
import pickle as pkl
import pandas as pd
import numpy as np
import chardet
import lxml
import time
import pytz
from datetime import datetime
import databasecreationfunctions as d


app = Flask(__name__)

df=d.random_df(1000) #Puedes usar cualuiqera, uno random o uno que ellos creen, desde que tenga todas las categorias (yo cree otro por simplicidad).
bins = [0, 20, 40, 60, 80, 100, df['Age'].max() + 10]
df['Age_Category'] = pd.cut(df['Age'], bins=bins)
df = df.dropna(subset=['Age_Category'])
df['Age_Category'] = pd.IntervalIndex(df['Age_Category']).sort_values()
category_counts = df.groupby('Age_Category')['uci'].sum().tolist()
age_categories = df['Age_Category'].unique().astype(str).tolist() 
@app.route('/')
def index():
    return render_template('bar_plot.html', age_categories=age_categories, category_counts=category_counts)

if __name__ == '__main__':
    app.run(debug=True)



