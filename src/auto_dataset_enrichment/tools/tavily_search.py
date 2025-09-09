import os
import json
from crewai.tools import BaseTool
from pydantic import BaseModel, Field,ConfigDict
from typing import Type, List, Optional ,Dict
from tavily import TavilyClient



class TavilySearchInput(BaseModel):
    queries: List[Dict] = Field(..., description="A list of dictionaries, where each dictionary is a separate search query .")
    
    


class TavilySearchTool(BaseTool):

    name: str = "Tavily Search Tool "

    description: str = (
        "Executes a list of search queries for a topic  using the Tavily API. "
        "For best results, each query in the list should be as specific and detailed as possible. "

    )

    args_schema: Type[BaseModel] = TavilySearchInput
    model_config = ConfigDict(extra='allow', arbitrary_types_allowed=True)

    

    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            raise ValueError("TAVILY_API_KEY environment variable is not set.")
        self.tavily_client = TavilyClient(api_key = api_key)
    

    def _run(self,queries:List[Dict]) -> str:

        all_results = []
        try:
            
            for query_dict in queries:
                # Combine base parameters with query-specific parameters from the list
                search_params = {
                    "search_depth": "advanced",
                    "include_raw_content": False,
                    "include_answer": False,
                    **query_dict
                }
                
                
                
                query_text = search_params.get('query', 'No query text provided')
                print(f"  - Searching for: {query_text}")
                response = self.tavily_client.search(**search_params)
                
                # Add the results from this query to the main list
                if response.get("results"):
                    all_results.extend(response["results"])

            final_responce = {"results":all_results}

            
            return json.dumps(final_responce,indent=2)
        except Exception as e:
            return f"An error occurred: {e}"
            


