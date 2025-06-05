import gensim.downloader as api

# Load GloVe embeddings directly
model = api.load("glove-wiki-gigaword-50")

# Function to construct a short paragraph
def construct_paragraph(seed_word, similar_words):
    # Create a simple template-based paragraph
    paragraph = (
        f"In the spirit of {seed_word}, one might embark on an unforgettable {similar_words[0][0]} "
        f"to distant lands. Every {similar_words[1][0]} brings new challenges and opportunities for {similar_words[2][0]}. "
        f"Through perseverance and courage, the {similar_words[3][0]} becomes a tale of triumph, much like an {similar_words[4][0]}."
    )
    return paragraph

# Generate a paragraph for "adventure"
seed_word = input("Enter a seed word: ")
similar_words = model.most_similar(seed_word, topn=5)

# Construct a paragraph
paragraph = construct_paragraph(seed_word, similar_words)

print(paragraph)
