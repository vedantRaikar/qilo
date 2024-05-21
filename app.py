import streamlit as st
import requests
from bs4 import BeautifulSoup
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain import OpenAI, LLMChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from transformers import pipeline

llm = OpenAI(openai_api_key='Your API KEY', temperature=0)
qa_model = pipeline("question-answering")

def get_wikipedia_content(topic):
    try:
        url = 'https://en.wikipedia.org/w/api.php' #Wikipedia API URL
        params = {
            'action': 'parse', #Action to be performed on the Wikipedia API, indicating that the request is for parsing content.
            'format': 'json', #Indicating that the response should be in JSON format.
            'page': topic, #Specifies the title of the Wikipedia page for which content is being requested. The value is dynamic and is determined by the topic variable.
            'prop': 'text', #Specifies the properties of the page that should be included in the response.
            'redirects': '' #An empty string indicates that redirects should not be followed, and the response should include information from the original page.
        }

        response = requests.get(url, params=params).json() #Sends an HTTP GET request to the specified URL (url) with the given parameters (params). The requests.get() function is part of the requests library and is used to make HTTP requests. The .json() method is then called on the response to parse it as JSON.
        raw_html = response['parse']['text']['*'] #Extracts the raw HTML content of the Wikipedia page from the JSON response. The JSON structure likely has a nested dictionary, where response['parse'] retrieves the parsed content of the page, and ['text']['*'] accesses the actual HTML content
        soup = BeautifulSoup(raw_html, 'html.parser') #BeautifulSoup object is created to parse and navigate the HTML content. The BeautifulSoup class, from the bs4 (Beautiful Soup) library, provides tools for web scraping by transforming a complex HTML document into a tree of Python objects.

        # Extract text from paragraphs
        paragraphs = soup.find_all('p') #This line uses the find_all method of the BeautifulSoup object (soup) to locate all <p> (paragraph) tags in the HTML content.
        text = ' '.join([p.get_text() for p in paragraphs]) #Creates a list comprehension to iterate through each paragraph (<p> tag) in the ResultSet (paragraphs). For each paragraph, p.get_text() is called to extract the text content inside the paragraph tag. The resulting list of text from all paragraphs is then joined into a single string using the join method with a space ' ' as the separator.

        return text

#If an exception occurs, it prints an error message and returns None. 
    except Exception as e:
        print(f"Topic '{topic}' not found on Wikipedia")
        return None
    
    #‘CharacterTextSplitter’ class is used for splitting text into smaller chunks based on specific parameters
def summarize_wikipedia_content(wikipedia_content):
    text_splitter = CharacterTextSplitter(
        separator="\n",  #Indicates that the text will be split at each occurrence of a newline character.
        chunk_size=1000, #Parameter sets the maximum size of each text chunk after splitting. The text will be divided into chunks, and each chunk will contain, at most, 1000 characters.
        chunk_overlap=200, #This parameter specifies the overlap between consecutive chunks. In other words, the chunks won't be completely independent; there will be a 200-character overlap between each chunk.
        length_function=len, #Returns the length of a string.
    )

    wiki = text_splitter.split_text(wikipedia_content) #split the wikipedia_content into chunks based on the specified parameters
    docs = [Document(page_content=t) for t in wiki]  #Each Document object is initialized with one of the text chunks (t) from the wiki list.
    chain = load_summarize_chain(llm, chain_type="map_reduce") #It takes the OpenAI model (llm) and specifies the chain type as "map_reduce". The "map_reduce" chain type is likely a summarization technique that involves mapping and reducing operations.
    summarized_text = chain.run(docs) #Executes the summarization chain on the provided documents

    return summarized_text

# Streamlit App
def main():

    st.title("Wikipedia Summarizer")

    # User input for the Wikipedia topic
    topic = st.text_input("Enter the Wikipedia topic:")
    if not topic:
        st.warning("Please enter a topic.")
        st.stop()

    # Retrieve Wikipedia content based on the input topic
    wikipedia_content = get_wikipedia_content(topic)
    if wikipedia_content is None:
        st.warning(f"No Wikipedia page found for '{topic}'. Please enter a valid topic.")
        st.stop()

    # Display Wikipedia content
    st.subheader(f"Content for '{topic}':")
    st.text_area("Wikipedia Content", wikipedia_content, height=300)

    # Sidebar with buttons
    page = st.sidebar.radio("Choose a page:", ["Summarize", "Get Answer"])

    if page == "Summarize":
        # Button to Summarize Wikipedia content
        summarize_button = st.button("Summarize")

        # Summarize Wikipedia content if the Summarize button is clicked
        if summarize_button:
            st.subheader("Summary:")
            summarized_text = summarize_wikipedia_content(wikipedia_content)

             # Display the summarized text
            st.write(summarized_text)

    elif page == "Get Answer":
        # Input for User's Question
        user_question = st.text_input("Ask a Question")

        # Button to Get Answer
        get_answer_button = st.button("Get Answer")

        # Get and display the answer if the Get Answer button is clicked
        if get_answer_button and user_question:
            answer = qa_model(question=user_question, context=wikipedia_content)
            st.subheader("Answer:")
            st.write(answer["answer"])

if __name__ == "__main__":
    main()