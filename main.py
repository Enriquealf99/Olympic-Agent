import pandas as pd
from dotenv import load_dotenv
import os
from llama_index.experimental.query_engine import PandasQueryEngine
from prompts import new_prompt, instruction_str, context
from note_engine import note_engine
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI
from pdf import olympics_engine

load_dotenv()

medallists_path = os.path.join("data", "medallists.csv")
medallists_df = pd.read_csv(medallists_path)

medallists_query_engine = PandasQueryEngine(df = medallists_df, verbose = True, instruction_str = instruction_str)
medallists_query_engine.update_prompts({"pandas_prompt": new_prompt})



medals_path = os.path.join("data", "medals.csv")
medals_df = pd.read_csv(medals_path)

medals_query_engine = PandasQueryEngine(df = medals_df, verbose = True,  instruction_str = instruction_str )
medals_query_engine.update_prompts({"pandas_prompt": new_prompt})




medals_total_path = os.path.join("data", "medals_total.csv")
medals_total_df = pd.read_csv(medals_total_path)

medals_total_query_engine = PandasQueryEngine(df = medals_total_df, verbose = True,  instruction_str = instruction_str)
medals_total_query_engine.update_prompts({"pandas_prompt": new_prompt})

tools = [
    note_engine,
    QueryEngineTool(query_engine = medals_query_engine, 
                    metadata = ToolMetadata(
                        name = "medals_data",
                        description = "this gives information about the 2024 Olympics medals",
                    ),
                    ),
    QueryEngineTool(query_engine = medallists_query_engine, 
                    metadata = ToolMetadata(
                        name = "medallists_data",
                        description = "this gives information about the 2024 Olympics medallists",
                    ),
                    ),
    QueryEngineTool(query_engine = medals_total_query_engine, 
                    metadata = ToolMetadata(
                        name = "medals_total_data",
                        description = "this gives information about the 2024 Olympics total medals",
                    ),
                    ),
    QueryEngineTool(query_engine = olympics_engine, 
                    metadata = ToolMetadata(
                        name = "olympics_data",
                        description = "this give relevant information about the 2024 Summer Olympics",
                    ),
                    ),
]

llm = OpenAI(model = "gpt-4o-mini")

agent = ReActAgent.from_tools(tools, llm = llm, verbose = True, context = context)

while(prompt := input("Enter a prompt  (q to quit): ")) != "q":
    result = agent.query(prompt)
    print(result)

