# Libraries needed
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def preprocess_lyrics(df):
    """Function that preprocesses a dataframe before feeding it to text classification libraries"""
    
    # Filter English songs
    df = df.loc[df['language'] == 'en']
    
    # Filter relevant columns
    df = df[['lyrics', 'playlist_genre']]
    
    # Change column names for model
    df = df.rename(columns={'lyrics': 'text', 'playlist_genre': 'label'})
    
    # Lables str->numeric
    df['label'] = pd.factorize(df['label'])[0]
    
    return(df)


def preprocess(data):
    """Function that preprocesses a dataframe before feeding it to text classification libraries.
    Inputs:
    - data. Dataframe containing text fields and gold classification labels-
    
    Output:
    - Dataframe containing only the 6 largest classes of the data. The relevant text fields have been joint to a single string column. The classes are changed to integers.
    - Dictionary with the mapping of classes to integers. 
    """
    
    # Join relevant text info
    data['text'] = data['headline'] + ' ' + data['short_description']

    # Filter relevant columns
    data = data[['text', 'category']]

    # Change column names for model
    data = data.rename(columns={'category': 'label'})
    
    # Get only 6 largest classes / Create dictionary to map classes to integers
    data, genres = largest_classes(data, 6)   

    # Lables str->numeric
    data['label'] = pd.factorize(data['label'], sort=True)[0]
    
    return data, genres

def largest_classes(data, n_class):
    """
    Select only the n_class largest classes of a pandas dataframe.
    """
    largest = data.groupby('label').count().nlargest(n_class,'text')
    classes = [i[0] for i in largest.itertuples()] # value of largest classes
    data = data[data['label'].isin(classes)]
    
    # Create dictionary to map classes to integers
    genres = {key:classes.index(key) for key in classes}  
    
    return data, genres

def model_performance(model, data, n_obs, n_class):
    """
    Function that given a classification model and data returns the performance of said model. 
    
    Inputs:
    - model: ML model used for classification. Should it be fit?
    - data: Dataframe object containg the text field and the gold-label for classification.
    - n_obs: number of observations used.
    - n_class: number of classes used.
    
    Outputs:
    - Precision: Integer.
    - Recall: Integer.
    - F1-score: Integer.
    """
    # Filter the n_class largest classes.
    data,_ = largest_classes(data, n_class)
    
    # Select n_obs from the data. Stratified by class.
    train_df, test_df = train_test_split(data, train_size= n_obs, test_size=2000)
    
    # Fit Model
    model.fit(train_df['text'], train_df['label'])
    
    # Predict
    preds = model.predict(test_df['text'])   
    
    # Compute Performance Measures   
    precision, recall, f1 = performance_measures(test_df['label'], preds)
    
    return precision, recall, f1


#from simpletransformers.classification import ClassificationModel, ClassificationArgs

import matplotlib.pyplot as plt

def transformer_performance(model, data, n_obs, n_class):
    
    # Filter the n_class largest classes.
    data,_ = largest_classes(data, n_class)
    
    # Factorize classes for transformer class
    data['label'] = pd.factorize(data['label'], sort=True)[0]
    
    # Select n_obs from the data. Stratified by class.
    train_df, test_df = train_test_split(data, train_size= n_obs, test_size=2000)
    
    # Model train
    model.train_model(train_df)
    
    # Model predict
    test_l = list(test_df['text'])
    preds,_ = model.predict(test_l)
    
    # Compute Performance Measures   
    precision, recall, f1 = performance_measures(test_df['label'], preds)
    
    return precision, recall, f1



def performance_measures(test_data, predictions):
    
    measures = classification_report(test_data, predictions, zero_division = 0, output_dict=True)
    precision = measures['macro avg']['precision']
    recall = measures['macro avg']['recall']
    f1 = measures['macro avg']['f1-score']
    
    return precision, recall, f1



def plot_comp(f1_array, time_array, n_class):
    """
    Function that shows and stores a multiplot. 
    The plot has 2 level.
    - The upper level compares the f1-score of 3 models.
    - The lower level compares the training time - in minutes - of such models.
    Each level compares 8 different sample size.
    
    Inputs:
    - f1_array: Numpy array with values of f1-score of each model. Shape = (3,8).
    - time_array: Numpy array with values of training time of each model. Shape = (3,8)
    
    Outputs:
    - matplotlib.pyplot.show(): Shows the plot.
    - Stores the plot in a folder calles 'Plots' in .png format.
    """
    
    title = 'Text Classification - ' + str(n_class) + ' Classes'

    # Create plot
    fig, (ax1, ax2) = plt.subplots(2, figsize=(15,5))
    fig.suptitle(title)
    
    X = np.arange(8) #number of different sample sizes
    
    # F1-Score
    ax1.bar(X-0.25 , f1_array[0], color = 'lightgreen', width = 0.25)
    ax1.bar(X, f1_array[1], color = 'darkblue', width = 0.25)
    ax1.bar(X +0.25, f1_array[2], color = 'yellow', width = 0.25)
    ax1.set(ylabel='F1-score')
    ax1.set_xticks([])
    
    # Training Time
    ax2.bar(X-0.25 , time_array[0]/60, color = 'lightgreen', width = 0.25)
    ax2.bar(X, time_array[1]/60, color = 'darkblue', width = 0.25)
    ax2.bar(X +0.25, time_array[2]/60, color = 'yellow', width = 0.25)
    ax2.legend(labels=['LR', 'SVM', 'BERT'], loc=2)
    ax2.set(ylabel='Time (min)')
    
    # Xticks
    sample_size = ['1000', '2000', '4000', '6000', '10000', '15000', '20000', '40000']
    plt.xticks(ticks=np.arange(8),labels=sample_size, ha='center')
    plt.xlabel("Sample Size")
    
    # Save plot
    root = 'Plots/'+str(n_class)+'_Classes.png'

    plt.savefig(root)
    plt.show()
    