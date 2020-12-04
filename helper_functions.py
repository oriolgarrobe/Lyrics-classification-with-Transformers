import pandas as pd

def preprocess(data):
    """Function that preprocesses a dataframe before feeding it to text classification libraries.
    Inputs:
    - data. Dataframe containing text fields and gold classification labels-
    
    Output:
    - Dataframe with the text fields containing relevant information joint and the classes as integers.    
    """
    
    # Join relevant text info
    data['text'] = data['headline'] + ' ' + data['short_description']

    # Filter relevant columns
    data = data[['text', 'category']]

    # Change column names for model
    data = data.rename(columns={'category': 'label'})

    # Lables str->numeric
    data['label'] = pd.factorize(data['label'])[0]
    
    return(data)