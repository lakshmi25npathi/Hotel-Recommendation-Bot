
from typing import Any, Text, Dict, List

from rasa_sdk import Action, Tracker
from rasa_sdk.events import SlotSet,AllSlotsReset,Restarted
from rasa_sdk.executor import CollectingDispatcher
from clustering import load_data, unique_items,find_clusterdata
from sentiment_analysis import recommendation
import pandas as pd 
import numpy as np
import pickle
import joblib
import warnings

warnings.filterwarnings("ignore")

# load the data 
df = load_data('Restaurants_preprocessed_final.csv')

# Extract unique_cuisines
unique_cuisines = unique_items(df['CUISINES'])
# Remove following elements from list as not needed for clustering
r_list = ['Vegetarian Friendly','Vegan Options','Gluten Free Options']
for i in r_list:
    unique_cuisines.remove(i)
unique_cuisines = [i.lower() for i in unique_cuisines]
print(unique_cuisines)

# Extract unique_features
unique_features = unique_items(df['FEATURES'])
# Remove 'akeout', & '有泊車位' from unique_features
for i in ['akeout','有泊車位']:
    unique_features.remove(i)
unique_features = [i.lower() for i in unique_features]
print(unique_features)

# Extract unique_specialdiets
unique_specialdiets = unique_items(df['SPECIAL DIETS'])
unique_specialdiets = [i.lower() for i in unique_specialdiets]
print(unique_specialdiets)

# Extract unique_meals
unique_meals = unique_items(df['Meals'])
# Remove Table Service, Seating, Takeout
for i in ['Table Service','Seating','Takeout']:
    unique_meals.remove(i)
unique_meals = [i.lower() for i in unique_meals]
print(unique_meals)

# Extract unique price range
unique_budget = df['Budget'].unique()
unique_budget = [i.lower() for i in unique_budget]
print(unique_budget)

def convert_budget(input_item):
    
    if input_item == 'less than LKR 5000':
       return 'Low'
        
    elif input_item == 'Between LKR 5000 & LKR 50000':
        return 'Medium'
        
    else:
        return 'High'
        
def convert_onehot(input_item, unique_items):
    item_list = []
    for item in unique_items:
        if item == input_item:
            value = 1
            item_list.append(value)
        else:
            value = 0
            item_list.append(value)
    return item_list

class ActionRecommendRestaurants(Action):

    def name(self) -> Text:
         return "action_RecommendRestaurants"

    def run(self, dispatcher, tracker, domain):
        
        budget = tracker.get_slot('budget')
        
        cuisine = tracker.get_slot('cuisine')
        cuisine = cuisine.lower()
    
        feature = tracker.get_slot('feature')
        feature = feature.lower()
        
        special_diet = tracker.get_slot('special_diet')
        special_diet = special_diet.lower()
        
        meals = tracker.get_slot('meals')
        meals = meals.lower()
        
        # Encoding categorical variables
        budget_value = convert_budget(budget)
        
        # Create one hot coding 0 or 1, 1 if item is present else 0
        cuisine_list = convert_onehot(cuisine,unique_cuisines)
        feature_list = convert_onehot(feature,unique_features)
        special_diet_list = convert_onehot(special_diet, unique_specialdiets)
        meals_list = convert_onehot(meals, unique_meals)
        budget_list = convert_onehot(budget_value, unique_budget)
        
        # Combine all lists
        final_list = cuisine_list + feature_list + special_diet_list + meals_list + budget_list
        
        # Create a dataframe from list
        final_list = np.array(final_list)
        # Reshape 
        final_list = final_list.reshape(1,-1)
        
        # K-Means clustering
        k_means = joblib.load('K_means.joblib')
        pred_cluster = k_means.predict(final_list)
        cluster_df = find_clusterdata(pred_cluster)
        
        # Sentiment analysis
        sentiment_df = pd.read_csv('Restaurants_with_sentiment_scores_BERT.csv')

        recommend_restaurants = recommendation(sentiment_df, cluster_df)

        #Top 10 recommended restaurants 
        top_10_df = list(recommend_restaurants.Name[:10])
        
        dispatcher.utter_message('Here are the top 10 Recommend Restaurants based on your preferences:')
        
        count=1
        for i in top_10_df:
            
            result = str(count)+'.'+ str(i)
            count+=1
            dispatcher.utter_message(result)
            
        return []

class ActionContinue(Action):

    def name(self):
        return "action_continue"
        
    def run(self, dispatcher, tracker, domain):

        dispatcher.utter_message(template="utter_greet")

        return [AllSlotsReset()]

     
