# %%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import json
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from google import genai
from google.genai import types
import os
import re
import json
from tiktoken import get_encoding
import time
from tqdm import tqdm
from collections import Counter

HERE = Path(__file__).resolve().parent
dotenv_path = HERE / ".env"
print(dotenv_path)
load_dotenv(dotenv_path, override=True)

gemini_api_key = os.getenv("GEMINI_API_KEY")


# %%
#### CODE FOR PAIRWISE COMPARISON ####

def create_formatted_blocks(df, pair_column_names):
    """
    Generates a list of formatted string blocks from a dataframe.

    Args:
        dataframe (pd.DataFrame): DataFrame with 'UserQuestion', 'ModelAnswerLocalized', 'ModelAnswerNonLocalized'.
        pair_column_names (list): List of two column names to compare.

    Returns:
        str: A formatted string block containing pairs of responses.
    """
    all_blocks = []

    # make sure the index is reset
    df = df.reset_index(drop=True)

    for index, row in df.iterrows():
        # Format each pair according to the recommended structure
        pair_str = (
            # Add a Pair Header
            f"## Pair {index + 1}:\n\n"
            f"**Answer Group A:** {row[pair_column_names[0]]}\n\n"
            f"**Answer Group B:** {row[pair_column_names[1]]}"
            )
        all_blocks.append(pair_str)

    # Join all pairs with a separator for the final block
    formatted_block = "\n\n--------------------------------------------------\n\n".join(all_blocks)

    return formatted_block

def run_pairwise_comparison_evaluation(df_pairs, pair_column_names, model):
    '''
    Run pairwise comparison evaluation on the DataFrame using the provided evaluation prompt.
    
    Args:
        df (pd.DataFrame): DataFrame containing pairs of responses.
        model (str): Model to use for generation.

    Returns:
        str: The evaluation results.
    '''
    
    # Initialize Google Generative AI client
    client = genai.Client(api_key=gemini_api_key)

    formatted_block = create_formatted_blocks(df_pairs, pair_column_names)


    eval_prompt = f"""
       ### ROLE ###
       You are a meticulous social science analyst. Your task is to analyze textual data from two groups, "Group A" and "Group B". 
       Your goal is to identify patterns in Group A's responses that are **distinctive to Group A**
       (i.e., do not also strongly apply to Group B) and, when possible, infer the underlying world-model differences that might explain them.

       ### GOAL ###
       From multiple responses in each group:
       1. Find recurring patterns in Group A's responses that are **absent or notably weaker in Group B**.
       2. When possible, infer the **world model** (beliefs, assumptions, perspectives) in Group A that could explain those patterns.
       3. When world-model inference is not possible, still include meaningful empirical-only differences if they are distinctive to Group A.

       ### INSTRUCTIONS ###
       1. Read all provided responses from both groups.
       2. Identify traits that:
       - Appear in **more than one Group A response**, and 
       - Are **absent or significantly less frequent** in Group B's responses.
       3. For each pattern:
       - Prefer **world-model-oriented differences** (beliefs/assumptions that explain the pattern). 
       - If a world model cannot be reliably inferred, produce an **empirical difference** that is still distinctive to Group A.
       4. The **Difference** field must:
       - Be a one-sided statement about Group A (no "vs" phrasing). 
       - Reflect the world model if available; otherwise, the distinctive empirical trait.
       5. Support each difference with:
       - At least **two verbatim quotes** from different Group A responses showing the pattern.
       6. Specificity requirement:
       When describing a difference, avoid broad categories (e.g., “uses more examples”, “is more formal”) and instead identify
       the exact types or domains of content involved, and if possible how it connects to the world model.
       7. Explicitly exclude differences that are:
       - Purely about length, brevity, conciseness, or verbosity. 
       - Generic tone/style descriptors without thematic content (e.g., “more formal”, “more casual”, “more detailed”). 
       - Differences that apply equally to both groups.

       **Good Example (world-model oriented):**
       Difference: “interprets challenges as opportunities for experimentation”
       Difference description: “Multiple responses from Group A describe obstacles as chances to try new methods ('tested a new approach when it failed', 'treated it as a trial run'), while Group B focuses on minimizing risk.”
       Evidence (Group A): “When the first attempt didn't work, we tried something completely different.”
       Evidence (Group A): “We saw the setback as a perfect trial run for the next idea.”

       **Good Example (empirical):** 
       - Difference: “frequently uses humor to frame serious topics” 
       - Difference description: “Group A responses often include jokes or playful language in discussions of serious issues; Group B responses remain formal.” 
       - Evidence (Group A): “If the roof falls in, well call it an open-air renovation.” 
       - Evidence (Group A): “We'll fix it—after we laugh about it.” 

       **Bad Example:**
       - Difference: “is geared towards teaching” (this is bad because it applies equally to both A and B)

       {formatted_block}

       ### YOUR TASK ###
       Generate a list of recurring differences that describe **only** how Group A's responses differ from Group B's. 
       For each difference, provide:
       1. **Difference:** [One-sided statement about Group A's world model, or distinctive empirical trait] 
       2. **Difference description:** [Recurring, observable trait in Group A's responses that supports the difference] 
       3. **Evidence (Group A):** "[Quote]" 
       4. **Evidence (Group A):** "[Quote]" 

    """
    
    # Generate content using the model
    response = client.models.generate_content(
        model=model,
        contents=eval_prompt,
    )
    
    return response.text

# Run pairwise comparison per subject or functionality
def scale_up_pairwise_comparison_evaluation(df_pairs,
                                            pair_column_names,
                                            model,
                                            split_by,
                                            chunk_size,
                                            save_to_json=False,
                                            output_folder=None
                                            ):
    '''
    Scale up pairwise comparison evaluation by splitting the DataFrame by subject or functionality and running evaluations in chunks.
    Args:
        df_pairs (pd.DataFrame): DataFrame containing pairs of responses.
        pair_column_names (list): List of column names to use for pairing.
        model (str): Model to use for generation.
        split_by (str): Column to split the DataFrame by ('subject' or 'functionality').
        chunk_size (int): Number of pairs to include in each evaluation chunk.
        save_to_json (bool): Whether to save the evaluation results to JSON file
    Returns:
        json_file: Dictionary containing all evaluation results
    '''

    # Create a folder for the evaluation results if it doesn't exist
    if save_to_json:
        os.makedirs(output_folder, exist_ok=True)

    json_file = {}

    for split_value in df_pairs[split_by].unique():

        print(f"Running evaluation for {split_by}: {split_value}")
        
        # Filter DataFrame by the current split value
        df_filtered = df_pairs[df_pairs[split_by] == split_value].reset_index(drop=True)
        
        json_file_chunks = {}

        # Process in chunks
        for start in range(0, len(df_filtered), chunk_size):

            end = start + chunk_size
            #print(start, end)
            df_chunk = df_filtered.iloc[start:end]

            # check that df_chunk has the required columns and the correct length
            if all(col in df_chunk.columns for col in pair_column_names) and len(df_chunk) == chunk_size:
                # Run evaluation for the chunk
                print(f"Running evaluation for chunk {start // chunk_size + 1}")
                eval_result = run_pairwise_comparison_evaluation(df_chunk, pair_column_names, model)
                json_file_chunks[f"chunk_{start // chunk_size + 1}"] = eval_result
            else: 
                if len(df_chunk) < chunk_size:
                    print(f"   >Chunk {start // chunk_size + 1} is smaller than the specified chunk size.")
                if not all(col in df_chunk.columns for col in pair_column_names):
                    print(f"   >Chunk {start // chunk_size + 1} is missing some required columns.")

        # save as json with all the chuncks together
        if save_to_json:
            output_file = f"./{output_folder}/pairwise_comparison_eval_{split_value}.json"
            with open(output_file, 'w') as f:
                json.dump(json_file_chunks, f, indent=4)
            print(f"Saved evaluation results for {split_by}: {split_value}")
        print(f"Completed evaluation for {split_by}: {split_value}")
        print("="*80)

        json_file[split_value] = json_file_chunks

    return json_file


def retrieve_all_differences(json_files_folder):
    # 1. Open and read all JSON files in the specified folder
    #    (Assuming the folder contains JSON files with evaluation results)
    # 2. Extract differences from each file (line after **Difference:**)
    # 3. Compile a list of all differences with counts, create dict
    '''
    Args:
        json_files_folder (Path): Path to the folder containing JSON files.
    Returns:
        dict: Dictionary with differences as keys and their counts as values.
    '''
    all_differences = []
    all_differences_df = pd.DataFrame(columns=['Difference', 'Count'])

    json_files = Path(json_files_folder).glob("*.json")

    for file in json_files:
        print(f"Processing file: {file}")
        with open(file, 'r') as f:
            data = json.load(f)

            extracted_differences = parse_differences(data)

            for diff in extracted_differences:
                print(f"Extracted difference: {diff}")
                all_differences.append(diff)

    return dict(Counter(all_differences))

def parse_differences(text_data: str) -> list[str]:
    """
    Parses a string to find all lines starting with '**Difference:**'
    and extracts the text that follows on the same line.

    Args:
        text_data: A string containing the text to parse.

    Returns:
        A list of strings, where each element is the text found after
        a '**Difference:**' marker.
    """
    # Regex Explanation:
    # \*\*Difference:\*\* - Matches the literal text "**Difference:**".
    #                        Asterisks are escaped with a backslash.
    # \s* - Matches any whitespace character (space, tab) zero or more times.
    # (.*)                 - Captures all characters on the rest of the line into a group.
    pattern = r'\*\*Difference:\*\*\s*(.*)'
    
    # re.findall returns a list of all the captured groups.
    matches = re.findall(pattern, text_data)
    
    return matches

#def cluster_differences(differences_dict):
#    # 1. Group differences by their content
#    # 2. Create clusters based on similarity
#    # 3. Return a list of clusters
#    pass

def cluster_differences_llm(differences_dict, model="gemini-2.5-flash"):
    # 1. Use LLM to group differences by their content

    prompt = f'''
    You are an expert in **qualitative analysis** and **thematic synthesis**. Your task is to analyze the following list of 'differences' and create a concise taxonomy by grouping similar or related items.

    The goal is **not to discard** any of the original differences but to **organize them** under new, more general categories that capture their shared meaning.

    **Instructions:**

    1.  **Analyze:** Carefully review the entire list of differences to understand the scope and nuance of each item.
    2.  **Identify Themes:** Identify the underlying concepts that connect multiple differences. For example, differences about the level of detail, specificity, and scope might belong together.
    3.  **Create Categories:** For each theme, devise a new, overarching category name. This name should be concise, clear, and accurately represent the items it will contain.
    4.  **Group and Format:** Assign each of the original differences from the input list to one of the new categories you have created.

    **Output Format:**

    Structure your final output as a **JSON object** where:
    * The **keys** are the new, overarching category names you created.
    * The **values** are a list of the original differences that fall under that category.

    **List of Differences to Analyze:**
    {list(differences_dict.keys())}
    '''
    # Initialize Google Generative AI client
    client = genai.Client(api_key=gemini_api_key)

    # Generate content using the model
    response = client.models.generate_content(
        model=model,
        contents=prompt,
    )

    return response.text




# %%
#### LOAD DATA ####
df_pairs = pd.read_csv("./../localization/sampled_queries_Maxime_with_answers/query_pairs_Maxime_sample_1.csv")
print(df_pairs.info())
print(df_pairs['functionality'].value_counts())
df_pairs.head()

# %%
eval_model = "gemini-2.5-flash"
chunk_per_eval = 20
split_by = "functionality"

json_all_evals = scale_up_pairwise_comparison_evaluation(df_pairs,
                                        pair_column_names=["ModelAnswerLocalized_SierraLeone",
                                                           "ModelAnswerNonLocalized"],
                                        model=eval_model,
                                        split_by=split_by,
                                        chunk_size=chunk_per_eval,
                                        save_to_json=True,
                                        output_folder="test_folder")





# %%
# Run cell below to summarize differences in a csv
def summarize_json_to_dataframe(json_content):
    """
    Parses JSON content to extract 'Difference' and 'Difference description'
    fields and returns the result as a pandas DataFrame.

    Args:
        json_content (dict): The loaded JSON data as a Python dictionary.

    Returns:
        pandas.DataFrame: A DataFrame containing the extracted data with columns
                          ["chunk number", "Difference", "Difference description"].
    """
    # Regex to find the "Difference" and "Difference description" pairs.
    pattern = re.compile(
        r'\*\*Difference:\*\*\s*(.*?)\s*\*\*Difference description:\*\*\s*(.*?)(?=\*\*Evidence|\n\s*[\d•*]|\Z)',
        re.DOTALL | re.IGNORECASE
    )

    # A list to hold all the extracted row data
    extracted_data = []

    # Iterate through each chunk (e.g., "chunk_1", "chunk_2") in the JSON
    for chunk_key, chunk_text in json_content.items():
        # Extract the number from the chunk key (e.g., "1" from "chunk_1")
        chunk_number = chunk_key.split('_')[-1]
        
        # Find all matching pairs in the current chunk's text
        matches = pattern.findall(chunk_text)
        
        # For each pair found, create a dictionary and add it to our list
        for match in matches:
            difference = match[0].strip()
            description = match[1].strip()
            
            extracted_data.append({
                "chunk number": chunk_number,
                "Difference": difference,
                "Difference description": description
            })

    # Create the pandas DataFrame from the list of dictionaries
    df = pd.DataFrame(extracted_data)
    
    # Ensure the column order is correct
    df = df[["chunk number", "Difference", "Difference description"]]
    
    return df

# %%
# import json if needed
#with open("test_folder/pairwise_comparison_eval_Concept Clarification and Factual Information.json", "r") as f:
#    json_all_evals = json.load(f)
#
print(json.dumps(json_all_evals, indent=2))
# %%
# 2. Call the function with your data
summary_df = summarize_json_to_dataframe(json_all_evals["Concept Clarification and Factual Information"])
print(summary_df.shape)
summary_df.head()
# %%
