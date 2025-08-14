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
LOCALIZATION_FOLDER = HERE
dotenv_path = HERE / ".env"
print(dotenv_path)
load_dotenv(dotenv_path, override=True)

gemini_api_key = os.getenv("GEMINI_API_KEY")

# %%
#### CODE FOR PAIRWISE COMPARISON ####

def create_formatted_blocks(df, chunk_size):
    """
    Generates a list of formatted string blocks from a dataframe.

    Args:
        dataframe (pd.DataFrame): DataFrame with 'UserQuestion', 'ModelAnswerLocalized', 'ModelAnswerNonLocalized'.
        chunk_size (int): The number of pairs to include

    Returns:
        str: A formatted string block containing pairs of responses.
    """
    all_blocks = []

    if chunk_size >= len(df):
        print(f"Warning: chunk_size ({chunk_size}) is greater than or equal to the number of rows in the DataFrame ({len(df)}). Using the entire DataFrame.")

    df_chunked = df.head(chunk_size).reset_index(drop=True)  # Limit to the first 'chunk_size' rows for each block


    for index, row in df_chunked.iterrows():
        # Format each pair according to the recommended structure
        pair_str = (
            # Add a Pair Header
            f"## Pair {index + 1}:\n\n"
            f"**Answer Group A:** {row['ModelAnswerLocalized']}\n\n"
            f"**Answer Group B:** {row['ModelAnswerNonLocalized']}"
            )
        all_blocks.append(pair_str)

    # Join all 20 formatted pairs with a separator for the final block
    formatted_block = "\n\n--------------------------------------------------\n\n".join(all_blocks)

    return formatted_block

def run_pairwise_comparison_evaluation(df_pairs, model="gemini-2.5-flash-lite", chunk_size=20):
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

    # Prepare the formatted block of data
    #formatted_block = "\n\n".join(
    #    [f"**Query:** {row['UserQuestion']}\n\n**Group A Response:** {row['ModelAnswerLocalized']}\n\n**Group B Response:** {row['ModelAnswerNonLocalized']}" 
    #     for _, row in df_pairs.iterrows()]
    #)
    formatted_block = create_formatted_blocks(df_pairs, chunk_size=chunk_size)

    print(formatted_block)

    eval_prompt = f'''
    ### ROLE ###
    You are a meticulous social science analyst. Your task is to analyze textual data from two groups, "Group A" and "Group B", to identify the characteristic behavioral and content differences between them.

    ### GOAL ###
    To perform a pairwise comparison of responses from Group A and Group B for each query. Based on these comparisons, you will identify and describe noteworthy differences that appear to distinguish the two groups. This includes both widespread patterns and more subtle or emerging distinctions.

    ### INSTRUCTIONS ###
    1.  For each query, directly compare the response from Group A with the response from Group B.
    2.  Identify specific differences in their content, tone, perspective, assumptions, and reasoning.
    3.  After analyzing the individual pairs, synthesize your findings. Group similar observations together to describe general patterns of difference. **Note these patterns even if they don't appear in every single pair.**
    4.  Prioritize meaningful semantic differences over superficial formatting ones (e.g., focus on *what* is said and *how* it is framed, rather than the simple use of bolding or bullet points).

    ### INPUT DATA ###
    The data below contains pairs of responses to the same query.

    {formatted_block}

    ### YOUR TASK ###
    Based on your comparative analysis, generate a list of characteristic differences between Group A and Group B. Your list should capture the most significant distinctions, but also **include more nuanced or less frequent patterns that appear to be indicative of a difference.**

    For each characteristic difference you identify, you must provide:
    1.  **Difference:** A concise title for the difference (e.g., "Approach to Risk", "Level of Formality").
    2.  **Analysis:** A brief explanation of the difference, describing how Group A and Group B contrast on this point.
    3.  **Evidence:** A pair of short, verbatim snippets—one from Group A and one from Group B from the same query—that clearly illustrates this contrast.

    ### OUTPUT FORMAT ###
    Use the following format for each identified difference:

    **Difference:** [Concise title of the difference]
    * **Analysis:** [Your explanation of how the two groups differ on this point.]
    * **Evidence (Group A):** "[Verbatim quote from a Group A response]"
    * **Evidence (Group B):** "[Verbatim quote from the corresponding Group B response that shows the contrast]"

    ---
    Your analysis:
    '''

    # Generate content using the model
    response = client.models.generate_content(
        model=model,
        contents=eval_prompt,
    )
    
    return response.text




# %%

#### LOAD DATA ####

df_pairs = pd.read_csv(LOCALIZATION_FOLDER / "query_pairs_by_subject_gemini_2p5_flash.csv")
df_pairs.head()

# %%
#df_pairs_subject = df_pairs[df_pairs['subject'] == '[[Natural Sciences]]'].reset_index(drop=True)
#
#formatted_block = create_formatted_blocks(df_pairs, chunk_size=5)
#print("Formatted block for evaluation:")
#print(formatted_block)


# %%
# Run pairwise comparison evaluation subject per subject

eval_results = {}

for subject in df_pairs['subject'].unique():
    print(f"Running evaluation for subject: {subject}")
    df_pairs_subject = df_pairs[df_pairs['subject'] == subject].reset_index(drop=True)
    
    # Run evaluation
    eval_result = run_pairwise_comparison_evaluation(df_pairs_subject, model="gemini-2.5-flash", chunk_size=20)
    
    # Store results
    eval_results[subject] = eval_result
    break

# %%
# Print evaluation results for each subject
for subject, result in eval_results.items():
    print(f"Evaluation results for subject: {subject}")
    print(result)
    print("\n" + "="*80 + "\n")

# %%
# save as json
output_file = LOCALIZATION_FOLDER / "json_evals/pairwise_comparison_evaluation_by_subject_gemini_2p5_flash.json"
with open(output_file, 'w') as f:
    json.dump(eval_results, f, indent=4)
print(f"Saved evaluation results to {output_file}")





# %%
# Scale up the experiment, process the json files

# Run pairwise comparison per subject or functionality
def scale_up_pairwise_comparison_evaluation(df_pairs,
                                            model,
                                            split_by,
                                            chunk_size=20,
                                            save_to_json=False):
    '''
    Scale up pairwise comparison evaluation by splitting the DataFrame by subject or functionality and running evaluations in chunks.
    Args:
        df_pairs (pd.DataFrame): DataFrame containing pairs of responses.
        model (str): Model to use for generation.
        split_by (str): Column to split the DataFrame by ('subject' or 'functionality').
        chunk_size (int): Number of pairs to include in each evaluation chunk.
        save_to_json (bool): Whether to save the evaluation results to JSON file for each subject or functionality.
    Returns:
        None: The function saves the evaluation results to JSON files if `save_to_json` is True.
    '''

    for split_value in df_pairs[split_by].unique():

        # Create a folder for the evaluation results if it doesn't exist
        if save_to_json:
            output_folder = LOCALIZATION_FOLDER / f"json_evals/pairwise_comparison_eval_{model}_{split_by}_{split_value}"
            output_folder.mkdir(parents=True, exist_ok=True)

        print(f"Running evaluation for {split_by}: {split_value}")
        
        # Filter DataFrame by the current split value
        df_filtered = df_pairs[df_pairs[split_by] == split_value].reset_index(drop=True)
        
        # Process in chunks
        for start in range(0, len(df_filtered), chunk_size):
            end = start + chunk_size
            #print(start, end)
            df_chunk = df_filtered.iloc[start:end]

            # Break if df does not have the full chunk size
            if len(df_chunk) < chunk_size:
                print(f"Skipping last chunk from {start} to {end} as it has less than {chunk_size} pairs.")
                continue
            
            # Run evaluation for the chunk
            eval_result = run_pairwise_comparison_evaluation(df_chunk, model=model, chunk_size=chunk_size)

            # save as json each chunk
            if save_to_json:
                output_file = output_folder / f"pairwise_comparison_eval_{split_value}_chunk_{start//chunk_size}.json"
                with open(output_file, 'w') as f:
                    json.dump(eval_result, f, indent=4)
                print(f"Saved evaluation results for {split_by}: {split_value} chunk {start//chunk_size} to {output_file}")
        print(f"Completed evaluation for {split_by}: {split_value}")
        print("="*80)
        break


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
eval_model = "gemini-2.5-flash"
chunk_per_eval = 10
split_by = "subject"
#split_by = "functionality"

subjects_list = df_pairs['subject'].unique().tolist()
#functionality_list = df_pairs['functionality'].unique().tolist()

print(f"Subjects: {subjects_list}")
#print(f"Functionality: {functionality_list}")

scale_up_pairwise_comparison_evaluation(df_pairs,
                                        model=eval_model,
                                        split_by=split_by,
                                        chunk_size=chunk_per_eval,
                                        save_to_json=True)
# %%
# Retrieve all differences from the JSON files
json_files_folder = "./json_evals/pairwise_comparison_eval_gemini-2.5-flash_subject_[[Pedagogy & Instruction]]"

all_differences = retrieve_all_differences(json_files_folder)

# %%
diff_clusters = cluster_differences_llm(all_differences, model="gemini-2.5-flash")
print(diff_clusters)