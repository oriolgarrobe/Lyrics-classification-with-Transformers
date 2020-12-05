# Libraries needed
import pandas as pd
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
    
    # Get only 6 larger classes
    largest = data.groupby('label').count().nlargest(6,'text')
    classes = [i[0] for i in largest.itertuples()] # value of largest classes
    data = data[data['label'].isin(classes)]
    
    # Create dictionary to map classes to integers
    genres = {key:classes.index(key) for key in classes}    

    # Lables str->numeric
    data['label'] = pd.factorize(data['label'], sort=True)[0]
    
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
    largest = data.groupby('label').count().nlargest(n_class,'text')
    classes = [i[0] for i in largest.itertuples()] # value of largest classes
    data = data[data['label'].isin(classes)]
    
    # Select n_obs from the data. Stratified by class.
    train_df, test_df = train_test_split(data, train_size= n_obs, test_size=2000)
    
    # Fit Model
    model.fit(train_df['text'], train_df['label'])
    
    # Predict
    preds = model.predict(test_df['text'])   
    
    # Compute Performance Measures
    measures = classification_report(test_df['label'], preds, zero_division = 0, output_dict=True)
    precision = measures['macro avg']['precision']
    recall = measures['macro avg']['recall']
    f1 = measures['macro avg']['f1-score']
    
    return precision, recall, f1