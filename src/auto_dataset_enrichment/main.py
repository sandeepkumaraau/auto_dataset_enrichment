#!/usr/bin/env python
import sys
import warnings

from datetime import datetime

from auto_dataset_enrichment.crew import AutoDatasetEnrichment

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

from .tools import _iteration_for_market

# This main file is intended to be a way for you to run your
# crew locally, so refrain from adding unnecessary logic into this file.
# Replace with inputs you want to test with, it will automatically
# interpolate any tasks and agents information




def run():
    """
    Run the crew.
    """

    
    try:
        markets = _iteration_for_market(10)
        for pair in markets:
            input = {
               'question': pair[0],
               'description':pair[1]
            }

            print(f" Kicking off crew for market: {input['question']}")
            AutoDatasetEnrichment().crew().kickoff(inputs=input)
    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}")


def train():
    """
    Train the crew for a given number of iterations.
    """
    inputs = {
        "topic": "AI LLMs",
        'current_year': str(datetime.now().year)
    }
    try:
        AutoDatasetEnrichment().crew().train(n_iterations=int(sys.argv[1]), filename=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")

def replay():
    """
    Replay the crew execution from a specific task.
    """
    try:
        AutoDatasetEnrichment().crew().replay(task_id=sys.argv[1])

    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")

def test():
    """
    Test the crew execution and returns the results.
    """
    inputs = {
        "topic": "AI LLMs",
        "current_year": str(datetime.now().year)
    }
    
    try:
        AutoDatasetEnrichment().crew().test(n_iterations=int(sys.argv[1]), eval_llm=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while testing the crew: {e}")
