# Install required packages (use this in Jupyter or Colab)
!pip install cohere pypdf ipywidgets

# Import necessary libraries
import cohere
from pypdf import PdfReader
import getpass

# Initialize Cohere client with API key
co = cohere.Client("3JE437qflNp5M7TbvoQXMmeDuRrRWgylGCq09x29")

# Read and extract text from the PDF
reader = PdfReader(r"C:\Users\THANVITHA\Downloads\indian-penal-code.pdf")
text = ""
for page in reader.pages:
    text += page.extract_text()

# Define a function to ask IPC-related questions
def ask_ipc(question):
    prompt = f"""
    You are a legal assistant specialized in the Indian Penal Code (IPC).
    Use the following content to answer the user's question:
    {text[:10000]}
    User Question: {question}
    Respond with a relevant IPC section (if any) and a clear explanation.
    """
    
    response = co.generate(
        prompt=prompt,
        model="command-r-plus",
        max_tokens=300
    )

    print("\n" + response.generations[0].text.strip())

# Take user input and call the function
user_input = input("Ask your IPC question: ")
ask_ipc(user_input)
