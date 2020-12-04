import pandas as pd

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