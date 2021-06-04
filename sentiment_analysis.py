
import pandas as pd

def recommendation(cluster_df, sentiment_df):
    df = pd.merge(sentiment_df, cluster_df,how = 'inner', left_on = 'Name', right_on = 'Name')
    df = df.sort_values(by=['Positive_score'],ascending=False) 
    return df 
