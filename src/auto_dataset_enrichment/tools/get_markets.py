import pandas as pd
from pathlib import Path


market_file = Path('/Users/sandeepkumar/auto_dataset_enrichment/markets_with_history.csv')

def _iteration_for_market(num:int):

    df = pd.read_csv(market_file)

    result = []

    for _,row in df.head(num).iterrows():
        pair = []
        question = row['question']
        description = row['description']
        pair.append(question)
        pair.append(description)
        result.append(pair)
        
    
    return result


	
    

