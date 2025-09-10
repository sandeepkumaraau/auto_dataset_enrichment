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
The complete dataset is available on [kaggle](https://www.kaggle.com/datasets/sandeepkumarfromin/full-market-data-from-polymarket/data).

## Automated relevance discovery

This part is about creation of 'search_strategist Agent' using [crewai](https://github.com/crewAIInc/crewAI) to taken as input "market_question" ,"market_description" and use tools like [tavily search](https://github.com/tavily-ai/tavily-python) to automatically genrate the most relevent queries about the market to search for and do relevence scoring by checking Source credibility,Direct Relevance,Timeline check,Data-Driven Analysis to  rank the most relevent website for each market, leveraging the search  results snippts to ensure high-quality sources and then output top 10 website to be scraped.

1. The [Agent](src/auto_dataset_enrichment/config/agents.yaml) used is the 'search_strategist'.
2. The [Task](src/auto_dataset_enrichment/config/tasks.yaml) given is the 'url_discovery_task'.
3. The [Tool](src/auto_dataset_enrichment/tools/tavily_search.py) is used by the Agent.
   
## Example Output

```text
╭──────────────────────────────────────────────────────────────────────────── 🔧 Agent Tool Execution ─────────────────────────────────────────────────────────────────────────────╮
│                                                                                                                                                                                  │
│  Agent: Search Strategist and URL Analyst                                                                                                                                        │
│                                                                                                                                                                                  │
│  Thought: I have a comprehensive plan to analyze the market question regarding the 2024 Taiwanese presidential election. I've generated a list of targeted search queries to     │
│  gather relevant information. Now, I need to execute these queries using the Tavily Search Tool to collect the URLs and their content for analysis.                              │
│                                                                                                                                                                                  │
│  Using Tool: Tavily Search Tool                                                                                                                                                  │
│                                                                                                                                                                                  │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─────────────────────────────────────────────────────────────────────────────────── Tool Input ───────────────────────────────────────────────────────────────────────────────────╮
│                                                                                                                                                                                  │
│  "{\"queries\": [{\"query\": \"2024 Taiwan election polls Hou Yu-ih Ko Wen-je\"}, {\"query\": \"Taiwan presidential election Hou Yu-ih vs Ko Wen-je\"}, ... ]}"                   │
│                                                                                                                                                                                  │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭────────────────────────────────────────────────────────────────────────────────── Tool Output ───────────────────────────────────────────────────────────────────────────────────╮
│                                                                                                                                                                                  │
│  {                                                                                                                                                                               │
│    "results": [                                                                                                                                                                  │
│      { "url": "https://apnews.com/article/taiwan-election-hou-talks...", "title": "Taiwan presidential hopeful Hou promises...", "score": 0.98524 },                             │
│      { "url": "https://international.thenewslens.com/feature/2024-taiwan-election...", "title": "Ko Overtakes Lai in Presidential Race...", "score": 0.98205 },                  │
│      ...                                                                                                                                                                         │
│    ]                                                                                                                                                                             │
│  }                                                                                                                                                                               │
│                                                                                                                                                                                  │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯

```
## Agent Driven data extraction
This part is about creation of 'structured extractor agent' that takes as input top 10 websites from the 'search_strategist' and uses WebsiteSearchTool from the crewai_tools library, configured to use Google's Gemini model for conducting semantic searches within the content of websites using Retrieval-Augmented Generation (RAG) to navigate and extract information from specified URLs efficiently. Then the Agent extract a rich set of fields including headline, date, author, summary, main text, named entities,key events, implied probabilities, market mentions, causal statements, and (for Wikipedia) edit info and return a list of json objects,each with full schema, one per url.

1. The [Agent](src/auto_dataset_enrichment/config/agents.yaml) used is the 'structured_extractor'.
2. The [Task](src/auto_dataset_enrichment/config/tasks.yaml) given is the 'extract_structured_articles_task'.
3. The [Tool](src/auto_dataset_enrichment/tools/rag_extract.py) is used by the Agent.









