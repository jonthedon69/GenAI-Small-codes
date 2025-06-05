!pip install langchain langchain-community cohere google-auth google-auth-oauthlib google-auth-httplib2 google-api-python-client

import os
from cohere import Client
from langchain.prompts import PromptTemplate

# Step 1: Set Cohere API Key
os.environ["COHERE_API_KEY"] = "wqIHv9O46Q1Rx8I2Y05ved6wuLg3SYr2nE8Vd6Wa"
co = Client(os.getenv("COHERE_API_KEY"))

# Step 2: Load Text Document (Option 1: Local File)
with open("C:/Users/THANVITHA/GenAI/APIKey.txt", "r", encoding="utf-8") as file:
    text_document = file.read()

# Step 3: Create a Prompt Template
template = """
You are an expert summarizer. Summarize the following text in a concise manner:
Text: {text}
Summary:
"""
prompt_template = PromptTemplate(input_variables=["text"], template=template)
formatted_prompt = prompt_template.format(text=text_document)

# Step 4: Send Prompt to Cohere API
response = co.generate(
    model="command",
    prompt=formatted_prompt,
    max_tokens=50
)

# Step 5: Display Output
print("Summary:")
print(response.generations[0].text.strip())
