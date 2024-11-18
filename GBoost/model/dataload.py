import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_data():
    #Define column names
    column_names = ['user_id', 'item_id']

    path = '../data'
    interaction = pd.read_csv(path + '/u.data', sep='\t', names=column_names, usecols=['user_id', 'item_id'])
    

load_data()