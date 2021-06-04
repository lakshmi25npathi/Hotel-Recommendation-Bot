import pandas as pd

# load the data
def load_data(data):   
    return pd.read_csv(data)

# Define a function to extract unique items from dataset features.       
def unique_items(df):
    unique_list = []
    for i in df.unique():
        string_split = i.split(',')
        for j in string_split:
            if not j in unique_list:
                unique_list.append(j)
    unique_list = [x.strip(' ') for x in unique_list]
    unique_list =set(unique_list)
    unique_list = list(set(unique_list))
    return unique_list
    
def find_clusterdata(cluster):
    if cluster == 0:
        df = pd.read_csv('df0_kmeans.csv')
        return df
    elif cluster == 1:
        df = pd.read_csv('df1_kmeans.csv')
        return df
    elif cluster == 2:
        df = pd.read_csv('df2_kmeans.csv')
        return df
    
