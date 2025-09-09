from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
import os
from dotenv import load_dotenv 
from .tools import(
    search_tool,
    rag_extract
)
# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators

google_api_key = os.getenv('GOOGLE_API_KEY')
from crewai import LLM

gemini_llm = LLM(

    model='gemini/gemini-2.0-flash',
    api_key= google_api_key,
    
    temperature=0.5

)


@CrewBase
class AutoDatasetEnrichment():
    """AutoDatasetEnrichment crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    # Learn more about YAML configuration files here:
    # Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
    # Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
    
    # If you would like to add tools to your agents, you can learn more about it here:
    # https://docs.crewai.com/concepts/agents#agent-tools
    @agent
    def search_strategist(self) -> Agent:
        return Agent(
            config=self.agents_config['search_strategist'], # type: ignore[index]
            verbose=True,
            reasoning=True,
            tools= [search_tool],
            llm=gemini_llm
        )

    @agent
    def structured_extractor(self) -> Agent:
        return Agent(
            config=self.agents_config['structured_extractor'], # type: ignore[index]
            verbose=True,
            reasoning=True,
            tools=[rag_extract],
            llm=gemini_llm
        )

    # To learn more about structured task outputs,
    # task dependencies, and task callbacks, check out the documentation:
    # https://docs.crewai.com/concepts/tasks#overview-of-a-task
    @task
    def url_discovery_task(self) -> Task:
        return Task(
            config=self.tasks_config['url_discovery_task'] # type: ignore[index]

        )

    @task
    def extract_structured_articles_task(self) -> Task:
        return Task(
            config=self.tasks_config['extract_structured_articles_task'], # type: ignore[index]
            context=[self.url_discovery_task()],
            markdown=True,
            output_file='report.md'
        )

    @crew
    def crew(self) -> Crew:
        """Creates the AutoDatasetEnrichment crew"""
        # To learn how to add knowledge sources to your crew, check out the documentation:
        # https://docs.crewai.com/concepts/knowledge#what-is-knowledge

        return Crew(
            agents=self.agents, # Automatically created by the @agent decorator
            tasks=self.tasks, # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            
        )
