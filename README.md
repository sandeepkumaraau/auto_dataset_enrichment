# AutoDatasetEnrichment 

Welcome to the AutoDatasetEnrichment project, This project is indetend to show how to turn raw market data from Polymarket API, that had no prior publicly available dataset into a AI/ML redy dataset  Then using AI agents by providing them the market info to find more information about the market,so that i can train an AI agent to do sentiment analysis and market analysis to help optimize trading/decision strategy.

## Polymarket Dataset collection pipeline
This part is about the custom pipeline to collect raw market data from Polymarket GAMMA API, the data collected is:
1. Genral market information 
2. Hourly market price data 
3. Market volume and liqudity
4. Trades for each day 
5. Number of holders and thier postions 
6. Order book for each day of the market

Then using them to create additional finical scoring features for each market into the dataset.








