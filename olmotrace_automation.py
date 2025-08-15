# %%
import numpy as np
import pandas as pandas
# Import necessary classes from selenium
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.action_chains import ActionChains
import time
import re
import json


# %%
def complete_setup(driver, short_wait):

    # --- Step 1: Click through the FIRST dialog ("Limitations") ---
    print("Handling the Limitations dialog...")
    acknowledge_label = short_wait.until(
        EC.element_to_be_clickable((By.XPATH, "//span[text()='Acknowledge']"))
    )
    acknowledge_label.click()

    next_button_1 = short_wait.until(
        EC.element_to_be_clickable((By.XPATH, "//button[text()='Next']"))
    )
    next_button_1.click()
    print("Handled the Limitations dialog")

    # --- Step 2: Check the first box in the SECOND dialog using its position ---
    print("Handling the 'Terms of Use' dialog...")

    # This XPath finds the first <input type="checkbox"> inside the dialog box.
    # We then move to its parent ('/..') which is the actual clickable element.
    checkbox_1 = short_wait.until(
        EC.element_to_be_clickable((By.XPATH, "(//div[@role='dialog']//input[@type='checkbox'])[1]/.."))
    )
    checkbox_1.click()

    checkbox_2 = short_wait.until(
        EC.element_to_be_clickable((By.XPATH, "(//div[@role='dialog']//input[@type='checkbox'])[2]/.."))
    )
    checkbox_2.click()

    checkbox_3 = short_wait.until(
        EC.element_to_be_clickable((By.XPATH, "(//div[@role='dialog']//input[@type='checkbox'])[3]/.."))
    )
    checkbox_3.click()

    checkbox_4 = short_wait.until(
        EC.element_to_be_clickable((By.XPATH, "(//div[@role='dialog']//input[@type='checkbox'])[4]/.."))
    )
    checkbox_4.click()

    next_button_2 = short_wait.until(
        EC.element_to_be_clickable((By.XPATH, "//button[text()='Next']"))
    )
    next_button_2.click()
    print("Handled the 'Terms of Use' dialog")

    # --- Step 3: Handle the "Contribute to Datasets" Dialog (REVISED) ---
    print("Handling 'Contribute to Datasets' dialog...")
    # REVISED STRATEGY: Click the visible text label instead of the hidden input.
    # This is more reliable for complex UI components.
    opt_out_label = short_wait.until(
        EC.element_to_be_clickable((By.XPATH, "//span[contains(text(), 'OPT-OUT of dataset publication')]"))
    )
    opt_out_label.click()
    print("Handling 'Contribute to Datasets' dialog (Selected 'OPT-OUT' option)")

    lets_go_button = short_wait.until(
        EC.element_to_be_clickable((By.XPATH, "//button[text()=\"Let's Go!\"]"))
    )
    lets_go_button.click()
    #short_wait.until(EC.element_to_be_clickable((By.XPATH, "//button[text()=\"Let's Go!\"]"))).click()
    print("Setup complete! Ready to use the Playground.\n")


def query_model(driver, question, short_wait, long_wait):

    # Query the OLMo model
    text_box = short_wait.until(
        EC.element_to_be_clickable((By.CSS_SELECTOR, "textarea[placeholder^='Message OLMo']"))
    )

    # Below, good practices to make sure the query is typed in correctly
    # 1. Click the element to ensure it has focus.
    text_box.click()
    # 2. Clear the field to remove any default text (good practice).
    text_box.clear()
    # 3. Type the text
    #text_box.send_keys("How to teach fractions to Grade 5 students?")
    #text_box.send_keys("What is the capital of France?")
    text_box.send_keys(question)

    # 4. Find the submit button and click it.
    submit_button = short_wait.until(
        EC.element_to_be_clickable((By.CSS_SELECTOR, "button[aria-label='Submit prompt']"))
    )
    submit_button.click()

    print("Successfully sent the question.")

    print("Waiting for response and scrolling down to the button...")
    olmo_trace_button = long_wait.until(
        EC.element_to_be_clickable((By.XPATH, "//button[@aria-label='Copy']/following-sibling::button[2]"))
    )
    return olmo_trace_button

def retrieving_answer_text(driver, short_wait):

    print("\nRetrieving the answer text...")
    # MODIFIED LOCATOR: Find all assertive spans and select the LAST one.
    response_element = short_wait.until(
        EC.presence_of_element_located((By.XPATH, "(//span[@aria-live='assertive'])[last()]"))
    )

    # Get the clean text from the 'aria-label' attribute.
    answer_text = response_element.get_attribute('aria-label')

    return answer_text


def open_olmotrace_window(driver, olmo_trace_button):

    #olmo_trace_button = long_wait.until(
    #    EC.element_to_be_clickable((By.XPATH, "//button[@aria-label='Copy']/following-sibling::button"))
    #)
    # 2. Use JavaScript to scroll the button into the visible area of the window.
    driver.execute_script("arguments[0].scrollIntoView(true);", olmo_trace_button)
    time.sleep(0.5)  # A brief pause to allow any scrolling animation to finish.

    olmo_trace_button.click()
    print("Successfully opened the OLMoTrace window.")

def select_model(driver, model_name, short_wait):
    # --- Click the Model Selector Button ---
    print("Clicking the model selector button...")

    # Find the button by its unique role and wait for it to be clickable.
    model_selector_button = short_wait.until(
        EC.element_to_be_clickable((By.XPATH, "//div[@role='combobox']"))
    )

    # Click the button to open the dropdown menu.
    model_selector_button.click()
    print("Model selector dropdown opened.")

    print(f"Attempting to select model: {model_name}")
    try:
        # This XPath finds the list item ('li') that contains a paragraph ('p') with the exact model name.
        model_locator = (By.XPATH, f"//li[@role='option' and .//p[text()='{model_name}']]")

        model_option = short_wait.until(
            EC.element_to_be_clickable(model_locator)
        )
        model_option.click()
        print(f"Successfully selected {model_name}.")
    except Exception as e:
        print(f"Could not select model '{model_name}'. Error: {e}")


def query_olmo_pipeline(question, model, save_output, save_olmo_trace, max_documents=999):
    
    driver = webdriver.Chrome()
    driver.get("https://playground.allenai.org/")
    driver.maximize_window()
    
    # Create wait objects ONCE and pass them to the functions
    very_short_wait = WebDriverWait(driver, 2)
    short_wait = WebDriverWait(driver, 10)
    long_wait = WebDriverWait(driver, 60)

    final_dict = {"Question": question, "Model": model}
    
    try:
        complete_setup(driver, short_wait)

        if model != "OLMo 2 32B Instruct":
            select_model(driver, model, short_wait)

        olmo_trace_button = query_model(driver, question, short_wait, long_wait)

        if save_output:
            final_dict["Answer"] = retrieving_answer_text(driver, short_wait)

        if save_olmo_trace:
            open_olmotrace_window(driver, olmo_trace_button)

            # Recording data inside OLMoTrace docs
            olmo_trace_docs = record_olmo_trace(driver = driver,
                                               extract_url=True,
                                               very_short_wait=very_short_wait,
                                               short_wait=short_wait,
                                               long_wait=long_wait,
                                               max_documents=max_documents)
            
            final_dict["OLMoTrace docs"] = olmo_trace_docs

    finally:
        print("\n--- Pipeline finished ---")
        time.sleep(3)
        driver.quit()
    
    return final_dict

def reading_url(driver, very_short_wait):

    try:
        link_element = very_short_wait.until(
            EC.presence_of_element_located((By.XPATH, "//span[contains(text(), 'URL:')]"))
        )
        url = link_element.get_attribute('href')
        print(url)
        if url:
            print(f"  > Found URL: {url}")
    
    except TimeoutException:
        # If the element isn't found after 5 seconds, this block will run.
        print("  > No URL found on this page.")
    
    return url


def record_olmo_trace(driver, extract_url, very_short_wait, short_wait, long_wait, max_documents = 999):

    # for each document, record
    #   - document name (e.g., olmo-mix-1124)
    #   - training (pre, mid, post)
    #   - url
    #   - exact match
    #   - document text
    #   - relevance (if possible)

    # Loop Through Documents and Extract Data
    # First, find out how many "View Document" buttons there are.
    view_document_buttons = long_wait.until(
        EC.presence_of_all_elements_located((By.XPATH, "//button[text()='View Document']"))
    )
    num_buttons = len(view_document_buttons)
    print(f"Found {num_buttons} documents to process.")

    olmo_trace_docs = {}

    # Use an index-based loop to avoid stale element errors
    for i in range(num_buttons):

        olmo_trace_doc_i = {}

        print(f"Processing document {i + 1}/{num_buttons}...")
        # Re-find the buttons on each iteration to get a fresh list
        view_document_buttons_all = short_wait.until(
            EC.presence_of_all_elements_located((By.XPATH, "//button[text()='View Document']"))
        )

        # Scroll the button into view and click it
        button_to_click = view_document_buttons_all[i]
        driver.execute_script("arguments[0].scrollIntoView(true);", button_to_click)
        time.sleep(0.5) # Small pause to ensure scroll completes
        button_to_click.click()
        #actions = ActionChains(driver)
        #actions.move_to_element(button_to_click).click().perform()

        # --- Extract Information from the Document Title: Training type and document source ---
        print("   Extracting title information...")

        # Wait for the document dialog to be visible
        title_element = short_wait.until(
            EC.visibility_of_element_located((By.ID, "modal-title"))
        )

        # 1) Get the training type from the first span inside the title
        training_type_element = title_element.find_element(By.XPATH, ".//span[1]")
        olmo_trace_doc_i['Training type'] = training_type_element.text.replace(" document from:", "")
        #print(training_type_element.text.replace(" document from:", ""))

        # 2) Get the document source from the link (a tag) inside the title
        document_source_element = title_element.find_element(By.TAG_NAME, "a")
        olmo_trace_doc_i['Doc name'] = document_source_element.text
        #print(document_source_element.text)

        # 3) Get the URL from the 'href' attribute of the SAME link element
        document_url = document_source_element.get_attribute('href')
        olmo_trace_doc_i['URL doc'] = document_url

        # 4) Get the URL of training corpus
        if extract_url:
            pattern = r"URL:\s*(.+)"
            match = re.search(pattern, title_element.text)

            if match:
                url = match.group(1)
                print(f"    >Found URL: {url}")
                olmo_trace_doc_i['URL'] = url
            else:
                olmo_trace_doc_i['URL'] = None

        # 5) Extract the Text from the Blockquote ---
        print("Extracting text from the document...")

        # Use the (//...)[last()] syntax to get the last blockquote
        blockquote_element = long_wait.until(
            EC.visibility_of_element_located((By.XPATH, "(//blockquote)[last()]"))
        )

        # Use the .text property to get all the clean, visible text
        all_document_text = blockquote_element.text

        # add to dict if text
        if all_document_text:
            olmo_trace_doc_i['Full text'] = all_document_text

        # --- 2. Extract Text ONLY from the <strong> tags ---
        print("\nExtracting only the verbatim text...")
        # Use find_elements (plural) to get a list of all <strong> tags within the blockquote
        strong_elements = blockquote_element.find_elements(By.TAG_NAME, "strong")

        # Loop through the list of <strong> elements and get the text from each one
        verbatim_texts = [strong.text for strong in strong_elements]

        if len(verbatim_texts) > 0:
            olmo_trace_doc_i['Verbatim text'] = verbatim_texts

        # Closes the window of Doc i
        #print("  > Navigating back to the list...")
        back_button = very_short_wait.until(
            EC.element_to_be_clickable((By.XPATH, "//button[contains(@aria-label, 'close')]"))
        )
        back_button.click()
        # Wait for the list to be visible again before the next loop
        short_wait.until(EC.visibility_of_element_located((By.XPATH, "//button[text()='View Document']")))

        olmo_trace_docs[f"Doc {i+1}"] = olmo_trace_doc_i

        # Stop if more than max_documents document
        if i == max_documents - 1:
            print(f"Stopping after reading {max_documents}")
            break

    return olmo_trace_docs

# %%
#olmo_model = [
#    "OLMo 2 32B Instruct",
#    "OLMo 2 13B Instruct",
#]
#
#question = "Give me 3 ideas to teach fractions to Grade 5 students"
##question="Where was Napoleon born?"
#
#final_dict = query_olmo_pipeline(question,
#                                 model=olmo_model[1],
#                                 save_output=True,
#                                 save_olmo_trace=True,
#                                 max_documents=3
#                                 )
#
## %%
## print as json with indent 4
#print(json.dumps(final_dict, indent=4))
#
## %%
## save json
#path_file = "./../localization/OLMoTrace experiments/output_complete.json"
#json.dump(final_dict, open(path_file, "w"), indent=4)
## %%
