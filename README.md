# AutoDatasetEnrichment 

**Turning raw Polymarket data into AI/ML-ready datasets using automated multi-agent pipelines.**  

This project demonstrates how to:  
1. Collect raw market data from the **Polymarket GAMMA API**, where no public dataset previously existed.  
2. Transform it into a **machine-learning-ready dataset** with financial features.  
3. Use **AI agents** to automatically discover and extract relevant external information about each market.  
4. Enable downstream **ML tasks such as sentiment analysis, market analysis, and decision optimization**.  

---

## 📊 Polymarket Dataset Collection Pipeline  

A custom pipeline gathers comprehensive raw market data, including:  

1. General market information  
2. Hourly market price data  
3. Market volume and liquidity  
4. Daily trades  
5. Number of holders and their positions  
6. Daily order books  

From this, additional **financial scoring features** are engineered for each market.  

### 📅 Dataset Time Range

The dataset included in this repository (`polymarket_final_dataset.csv.zip`) contains historical data spanning approximately **1.5 years**:

- **Start Date**: January 3, 2024  
- **End Date**: July 23, 2025  

The ingestion pipeline uses `start_date_max="2025-01-01T00:00:00Z"` to fetch markets that started before 2025, ensuring comprehensive historical coverage.

➡️ The complete dataset is available on **[Kaggle](https://www.kaggle.com/datasets/sandeepkumarfromin/full-market-data-from-polymarket/data)**.  

---

## 🔎 Automated Relevance Discovery  

A **Search Strategist Agent** (powered by [crewAI](https://github.com/crewAIInc/crewAI)) takes as input a `market_question` and `market_description`, then:  

- Generates optimized queries using [Tavily Search](https://github.com/tavily-ai/tavily-python).  
- Scores results by **source credibility, direct relevance, timeline, and data quality**.  
- Ranks and outputs the **Top 10 high-quality URLs** for each market.  

📌 Config details:  
- Agent: [`search_strategist`](src/auto_dataset_enrichment/config/agents.yaml)  
- Task: [`url_discovery_task`](src/auto_dataset_enrichment/config/tasks.yaml)  
- Tool: [`tavily_search.py`](src/auto_dataset_enrichment/tools/tavily_search.py)  
   
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
│  "{\"queries\": [{\"query\": \"2024 Taiwan election polls Hou Yu-ih Ko Wen-je\"}, {\"query\": \"Taiwan presidential election Hou Yu-ih vs Ko Wen-je\"}, ... ]}"                  │
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
## 📑 Agent-Driven Data Extraction
A **Structured Extractor Agent** takes the `top URLs` from the `Search Strategist` and:

- Uses the **WebsiteSearchTool + Google’s Gemini (via RAG)** to extract structured insights.  
- Returns a rich JSON schema per article, including:
    - **Headline, date, author, summary, main text**
    - **Named entities, key events**
    - **Market mentions, causal statements, Wikipedia edit info**  
 

📌 Config details:  
- Agent: [`structured_extractor`](src/auto_dataset_enrichment/config/agents.yaml)  
- Task: [`extract_structured_articles_task`](src/auto_dataset_enrichment/config/tasks.yaml)  
- Tool: [`rag_extract.py`](src/auto_dataset_enrichment/tools/tavily_search.py)  

## Example Output

```text
╭────────────────────────────────────────────────────────────────────────────── 🔧 Agent Tool Execution ──────────────────────────────────────────────────────────────────────────────╮
│                                                                                                                                                                                     │
│  Agent: Structured Data Extractor                                                                                                                                                   │
│                                                                                                                                                                                     │
│  Thought: I will attempt to extract the information from the second URL, as the first one returned a forbidden error. I will use the tool to search for the relevant information    │
│                                                                                                                                                                                     │
│  Using Tool: Search in a specific website                                                                                                                                           │
│                                                                                                                                                                                     │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭──────────────────────────────────────────────────────────────────────────────────── Tool Input ─────────────────────────────────────────────────────────────────────────────────────╮
│  "{\"search_query\": \"2024 Taiwanese Presidential Election Hou Yu-ih vs. Ko Wen-je\", \"website\": \"https://www.csis.org/...\"}"                                                  │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭──────────────────────────────────────────────────────────────────────────────────── Tool Output ────────────────────────────────────────────────────────────────────────────────────╮
│  Relevant Content: Taiwan’s 2024 Presidential Election ... A three-way race between Lai (DPP), Hou Yu-ih (KMT), and Ko Wen-je (TPP).                                                │
│  Hou has little diplomatic experience; election framed as “war vs peace.”                                                                                                           │
│  ...                                                                                                                                                                                │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯

🚀 Crew: crew
├── 📋 Task: url_discovery_task → ✅ Completed using Tavily Search Tool
└── 📋 Task: extract_structured_articles_task → Executing (used Search in a specific website)

╭───────────────────────────────────────────────────────────────────────────── ✅ Agent Final Answer ────────────────────────────────────────────────────────────────────────────────╮
│  Taiwan’s 2024 Elections: Results and Implications                                                                                                                                 │
│  - William Lai (DPP) won with ~40% of the vote.                                                                                                                                    │
│  - Legislative Yuan split: DPP 51, KMT 52, TPP 8 → no majority.                                                                                                                    │
│  - Next four years: challenges in cross-Strait relations, U.S. support needed, domestic divisions persist.                                                                         │
│  ...                                                                                                                                                                               │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

## Future Work

- Add sentiment analysis module for enriched market datasets.
- Build predictive ML models for price forecasting and strategy optimization.
- Extend pipeline to other market APIs beyond Polymarket.









