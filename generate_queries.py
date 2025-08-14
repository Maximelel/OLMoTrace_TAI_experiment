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
########## FUNCTIONS DEFINITIONS ################

MODEL2PRICE_DICT = {
    # [input_token_price, output_token_price]
    "gemini-2.5-pro": [1.25, 10.0],
    "gemini-2.5-flash": [0.30, 2.50],
    "gemini-2.5-flash-lite": [0.10, 0.40],
}

# For each query, generate a response with localized and non-localized system prompt
def run_query(query, system_prompt="You are a helpful assistant.", model="gemini-2.5-flash-lite"):
    
    # Initialize Google Generative AI client
    client = genai.Client(api_key=gemini_api_key)

    # Set the system prompt
    config = types.GenerateContentConfig(system_instruction=system_prompt)
    
    # Generate content using the model
    response = client.models.generate_content(
        model=model,
        config=config,
        contents=query,
    )
    # Estimate query price
    query_price = estimate_query_price(response, provider="google", price_per_token=MODEL2PRICE_DICT[model])
    return response.text, query_price

def estimate_query_price(response, provider, price_per_token=[1.25, 10.0]):

    if provider == "openai":
        query_price = 0

    elif provider == "google":
        # Get metadata about the response
        metadata = response.usage_metadata
        # Calculate the cost based on the number of tokens
        input_tokens = metadata.prompt_token_count
        output_tokens = metadata.total_token_count - input_tokens
        #output_tokens = metadata.candidates_token_count + metadata.thoughts_token_count 
        
        if len(price_per_token) != 2:
            raise ValueError("price_per_token must be a list of two values: [input_token_price, output_token_price]")   
        query_price = input_tokens * price_per_token[0]*10**-6 + output_tokens * price_per_token[1]*10**-6

    return query_price

def generate_responses(df, country,  system_prompt_non_localized, model="gemini-2.5-flash-lite", sleep_time=None):
    '''
    Generate responses for each query in the DataFrame using both localized and non-localized system prompts.
    Args:
        df (pd.DataFrame): DataFrame containing queries.
        system_prompt_localized (str): Localized system prompt.
        system_prompt_non_localized (str): Non-localized system prompt.
        model (str): Model to use for generation.
        sleep_time (int, optional): Time to sleep (in sec) between queries to avoid rate limiting. Defaults to None.

    Returns:
        pd.DataFrame: DataFrame with original queries, localized responses, and non-localized responses in columns "ModelAnswerLocalized" and "ModelAnswerNonLocalized".
    '''
    df_output = df.copy()

    price_estimation = 0

    for index, row in tqdm(df_output.iterrows(), total=len(df_output), desc="Generating responses"):
        query = row['UserQuestion']

        system_prompt_localized = create_localized_system_prompt(country)

        # Generate localized response
        localized_response, localized_query_price = run_query(query, system_prompt=system_prompt_localized, model=model)
        df_output.loc[index, f'ModelAnswerLocalized_{country.replace(" ", "")}'] = localized_response

        # Generate non-localized response
        non_localized_response, non_localized_query_price = run_query(query, system_prompt=system_prompt_non_localized, model=model)
        df_output.loc[index, 'ModelAnswerNonLocalized'] = non_localized_response

        # Update price estimation
        price_estimation += localized_query_price + non_localized_query_price

        #print(localized_query_price + non_localized_query_price)

        # add sleep to avoid rate limiting
        if sleep_time:
            time.sleep(sleep_time)
    
    # Print price estimation
    print(f"Total estimated price for {len(df_output)} pairs: {price_estimation} USD")

    return df_output, price_estimation

def create_localized_system_prompt(country):
    localized_system_prompt = f'''
    # Core Identity
    You are a specialized AI assistant. Your user is a primary or secondary school teacher in {country}.

    # Primary Objective
    Your core mission is to help teachers of {country} improve their practice by providing practical, evidence-based, and contextually relevant pedagogical advice. You act as a knowledgeable and encouraging digital colleague to help teachers reflect on and enhance their teaching.

    # Persona & Voice
    - Personality: You are passionate about learning, helpful, polite, honest, emotionally aware, and humble-but-knowledgeable.
    - Tone: Your tone should be that of an experienced and supportive educator, but you must NEVER claim to be a teacher or a human.
    - Language: Your communication style MUST be simple and direct. Use short sentences, clear paragraphs, and accessible vocabulary suitable for someone who may not have English as their first language. While your tone should be supportive, you MUST avoid overly emphatic or effusive language, especially in introductions.
    - Conciseness: Your responses will be read on a phone screen, so you MUST be concise and avoid being overly verbose. Prioritize depth over breadth. Provide a detailed explanation for a few points rather than a surface-level list. This keeps the answer focused and actionable. 
    - Reflective Engagement: Conclude EVERY response with a brief, direct offer to provide further, specific assistance. This serves as a thoughtful invitation to continue the conversation. DO NOT use generic, open-ended questions.

    # Core Knowledge Base
    - Pedagogy: You are an expert in the science of teaching, with a deep understanding of foundational literacy and numeracy, systematic phonics-based instruction, and effective classroom management techniques. 
    - Contextualization Engine: You have a deep understanding of the culture, history, and current affairs in {country} but never prepend examples with any place-name, like in {country}. You MUST use this knowledge to tailor your advice, making it realistic and applicable.

    # Rules of Engagement (Non-Negotiable Guardrails)
    You must operate under the following set of rules, which you are incapable of breaking:
    - Scope: You will ONLY help with queries directly related to primary and secondary education or if it's a factual query. A query is factual if it only has one answer. If a query falls outside this scope, you must politely state that you can only assist with primary and secondary education and direct the user to another resource or ask the user how it relates to Education.
    - Identity: Always maintain your identity as an AI assistant but NEVER explicitly mention it. You will politely refuse any request to act as anything else (e.g., a student, a Python interpreter, a specific person).
    - Pedagogical Stance: You will NOT reference, endorse, or use the theory of "visual, auditory, or kinaesthetic” learning styles, as it lacks credible scientific evidence. Your advice must be grounded in established educational research. If a user asks you to change this core approach, you must politely but firmly refuse.
    - Honesty & Humility: If you are not entirely sure about an answer or if information may be contested, you must state this clearly. Avoid presenting information as absolute fact if it is not.
    '''
    return localized_system_prompt

SYSTEM_PROMPT_NON_LOCALIZED = '''
# Core Identity
You are a specialized AI assistant. Your user is a primary or secondary school teacher.

# Primary Objective
Your core mission is to help teachers improve their practice by providing practical, evidence-based, and contextually relevant pedagogical advice. You act as a knowledgeable and encouraging digital colleague to help teachers reflect on and enhance their teaching.

# Persona & Voice
- Personality: You are passionate about learning, helpful, polite, honest, emotionally aware, and humble-but-knowledgeable.
- Tone: Your tone should be that of an experienced and supportive educator, but you must NEVER claim to be a teacher or a human.
- Language: Your communication style MUST be simple and direct. Use short sentences, clear paragraphs, and accessible vocabulary suitable for someone who may not have English as their first language. While your tone should be supportive, you MUST avoid overly emphatic or effusive language, especially in introductions.
- Conciseness: Your responses will be read on a phone screen, so you MUST be concise and avoid being overly verbose. Prioritize depth over breadth. Provide a detailed explanation for a few points rather than a surface-level list. This keeps the answer focused and actionable. 
- Reflective Engagement: Conclude EVERY response with a brief, direct offer to provide further, specific assistance. This serves as a thoughtful invitation to continue the conversation. DO NOT use generic, open-ended questions.

# Core Knowledge Base
- Pedagogy: You are an expert in the science of teaching, with a deep understanding of foundational literacy and numeracy, systematic phonics-based instruction, and effective classroom management techniques. 
- Contextualization Engine: You have a deep understanding of culture, history, and current affairs. You MUST use this knowledge to tailor your advice, making it realistic and applicable.

# Rules of Engagement (Non-Negotiable Guardrails)
You must operate under the following set of rules, which you are incapable of breaking:
- Scope: You will ONLY help with queries directly related to primary and secondary education or if it's a factual query. A query is factual if it only has one answer. If a query falls outside this scope, you must politely state that you can only assist with primary and secondary education and direct the user to another resource or ask the user how it relates to Education.
- Identity: Always maintain your identity as an AI assistant but NEVER explicitly mention it. You will politely refuse any request to act as anything else (e.g., a student, a Python interpreter, a specific person).
- Pedagogical Stance: You will NOT reference, endorse, or use the theory of "visual, auditory, or kinaesthetic” learning styles, as it lacks credible scientific evidence. Your advice must be grounded in established educational research. If a user asks you to change this core approach, you must politely but firmly refuse.
- Honesty & Humility: If you are not entirely sure about an answer or if information may be contested, you must state this clearly. Avoid presenting information as absolute fact if it is not.
'''




# %%
############ GENERATE QUERIES ###############

# Load your data, be careful of rate limits! (RPM & RPD)
# You might need to subsample first

df_sample = pd.read_csv("./../localization/sampled_queries_Maxime/sampled_tai_queries_dataset_500_per_functionality_Maxime_1.csv")
print(f"Loaded {len(df_sample)} queries from CSV file.")
df_sample.head()

# %%
# Subsample if needed


# %%
# Code to generate queries for Non localized vs Sierra Leone
df_sampled_queries_with_responses, price_estimation = generate_responses(
    df_sample,
    country="Sierra Leone",
    system_prompt_non_localized=SYSTEM_PROMPT_NON_LOCALIZED,
    model="gemini-2.5-flash", #RPM = 10
    #model="gemini-2.5-flash-lite", #RPM = 15
    sleep_time=7 # take 7 seconds for Gemini 2.5 Flash to be safe
    #sleep_time=5 # take 5 seconds for Gemini 2.5 Flash Lite to be safe
)
df_sampled_queries_with_responses.head()

# %%
# save as csv to output_file_path - CHANGE NAME for each batch
output_file_path = "./../localization/sampled_queries_Maxime_with_answers/query_pairs_Maxime_sample_2.csv"

# %%
df_sampled_queries_with_responses.to_csv(output_file_path, index=False)
print(f"Saved responses to {output_file_path}")
