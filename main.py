# import os
# import streamlit as st
# import pickle
# import time
# from langchain import OpenAI
# from langchain.chains import RetrievalQAWithSourcesChain
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.document_loaders import UnstructuredURLLoader
# from langchain.embeddings import OpenAIEmbeddings
# # from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.vectorstores import FAISS

# from dotenv import load_dotenv
# load_dotenv()  # take environment variables from .env (especially openai api key)

# # print("API KEY IS: ")
# # print(" ")
# # print(os.getenv("OPENAI_API_KEY"))
# # print(" ")

# st.title("StockSage: News Research Analyzer 📈")
# st.sidebar.title("News Article URLs")

# urls = []
# for i in range(3):
#     url= st.sidebar.text_input(f"URL {i+1}")
#     urls.append(url)
#     # print(url)

# process_url_clicked = st.sidebar.button("Inspect URLs")
# file_path = "faiss-store-openai.pkl"

# main_placeholder = st.empty()
# llm = OpenAI(temperature=0.9, max_tokens=500)

# if process_url_clicked:
#     # load data
#     loader = UnstructuredURLLoader(urls=urls)
#     main_placeholder.text("Data Loading >>> Started >>> ✅✅✅")
#     data = loader.load()
#     # split data
#     text_splitter = RecursiveCharacterTextSplitter(
#         separators=['\n\n', '\n', '.', ','],
#         chunk_size=500
#     )
#     main_placeholder.text("Text Splitter >>> Started >>> ✅✅✅")
#     print(data)
#     docs = text_splitter.split_documents(data)
#     print(docs)
#     if not docs:
#         main_placeholder.text("No documents found or invalid data.")
#     else:
#         if os.path.exists(file_path):
#             with open(file_path, "rb") as f:
#                 vectorstore_openai = pickle.load(f)
#         else:
#             # create embeddings and save it to FAISS index
#             embeddings = OpenAIEmbeddings()
#             # embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY")) changes made

#             # Example with a lightweight model
#             # embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            
#             # Process and save the Faiss index into a pickle file
#             vectorstore_openai = FAISS.from_documents(docs, embeddings)
#             with open(file_path, "wb") as f:
#                 pickle.dump(vectorstore_openai, f)
#         main_placeholder.text("Embedding Vector Creation >>> ✅✅✅")
#     time.sleep(2)

# query = main_placeholder.text_input("Question: ")
# if query:
#     if os.path.exists(file_path):
#         with open(file_path, "rb") as f:
#             vectorstore = pickle.load(f)
#             chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
#             result = chain({"question": query}, return_only_outputs=True)
#             # result will be a dictionary of this format --> {"answer": "", "sources": [] }
#             st.header("Answer")
#             st.write(result["answer"])

#             # Display sources, if available
#             sources = result.get("sources", "")
#             if sources:
#                 st.subheader("Sources:")
#                 sources_list = sources.split("\n")  # Split the sources by newline
#                 for source in sources_list:
#                     st.write(source)


# import os
# import streamlit as st
# import pickle
# import time
# from langchain import OpenAI
# from langchain.chains import RetrievalQAWithSourcesChain
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.document_loaders import UnstructuredURLLoader
# from langchain.embeddings import OpenAIEmbeddings
# from langchain.vectorstores import FAISS

# from dotenv import load_dotenv
# load_dotenv()  # take environment variables from .env (especially openai api key)

# st.title("RockyBot: News Research Tool 📈")
# st.sidebar.title("News Article URLs")

# urls = []
# for i in range(3):
#     url = st.sidebar.text_input(f"URL {i+1}")
#     urls.append(url)

# process_url_clicked = st.sidebar.button("Process URLs")
# file_path = "faiss_store_openai.pkl"

# main_placeholder = st.empty()
# llm = OpenAI(temperature=0.9, max_tokens=500)

# if process_url_clicked:
#     # load data
#     loader = UnstructuredURLLoader(urls=urls)
#     main_placeholder.text("Data Loading...Started...✅✅✅")
#     data = loader.load()
#     print(data)
#     # split data
#     text_splitter = RecursiveCharacterTextSplitter(
#         separators=['\n\n', '\n', '.', ','],
#         chunk_size=1000
#     )
#     main_placeholder.text("Text Splitter...Started...✅✅✅")
#     docs = text_splitter.split_documents(data)
#     # create embeddings and save it to FAISS index
#     embeddings = OpenAIEmbeddings()
#     vectorstore_openai = FAISS.from_documents(docs, embeddings)
#     main_placeholder.text("Embedding Vector Started Building...✅✅✅")
#     time.sleep(2)

#     # Save the FAISS index to a pickle file
#     with open(file_path, "wb") as f:
#         pickle.dump(vectorstore_openai, f)

# query = main_placeholder.text_input("Question: ")
# if query:
#     if os.path.exists(file_path):
#         with open(file_path, "rb") as f:
#             vectorstore = pickle.load(f)
#             chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
#             result = chain({"question": query}, return_only_outputs=True)
#             # result will be a dictionary of this format --> {"answer": "", "sources": [] }
#             st.header("Answer")
#             st.write(result["answer"])

#             # Display sources, if available
#             sources = result.get("sources", "")
#             if sources:
#                 st.subheader("Sources:")
#                 sources_list = sources.split("\n")  # Split the sources by newline
#                 for source in sources_list:
#                     st.write(source)



# import os
# import streamlit as st
# import pickle
# import time
# from langchain import OpenAI
# from langchain.chains import RetrievalQAWithSourcesChain
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.document_loaders import UnstructuredURLLoader
# from langchain.embeddings import OpenAIEmbeddings
# from langchain.vectorstores import FAISS
# from dotenv import load_dotenv

# load_dotenv()  # take environment variables from .env (especially openai api key)

# st.title("StockSage: News Research Analyzer 📈")
# st.sidebar.title("News Article URLs")

# urls = []
# for i in range(3):
#     url = st.sidebar.text_input(f"URL {i+1}")
#     urls.append(url)

# process_url_clicked = st.sidebar.button("Inspect URLs")
# file_path = "faiss-store-openai.pkl"

# main_placeholder = st.empty()
# llm = OpenAI(temperature=0.9, max_tokens=500)

# if process_url_clicked:
#     # Load data from URLs
#     loader = UnstructuredURLLoader(urls=urls)
#     main_placeholder.text("Data Loading >>> Started >>> ✅✅✅")
#     try:
#         data = loader.load()
#         print(f"Fetched Data: {data}")  # Debugging line to check data
#         if not data:
#             main_placeholder.text("No data fetched from the URLs.")
#         else:
#             main_placeholder.text("Data Loaded Successfully >>> ✅✅✅")

#             # Split data into documents
#             text_splitter = RecursiveCharacterTextSplitter(
#                 separators=['\n\n', '\n', '.', ','],
#                 chunk_size=500
#             )
#             docs = text_splitter.split_documents(data)
#             print(f"Docs after splitting: {docs}")  # Debugging line to see if docs are split properly

#             if not docs:
#                 main_placeholder.text("No documents found after splitting.")
#             else:
#                 # Create FAISS index and save to pickle file
#                 embeddings = OpenAIEmbeddings()
#                 vectorstore_openai = FAISS.from_documents(docs, embeddings)
#                 print(f"Vectorstore created: {vectorstore_openai}")
#                 with open(file_path, "wb") as f:
#                     pickle.dump(vectorstore_openai, f)
#                 main_placeholder.text("Embedding Vector Creation >>> ✅✅✅")

#     except Exception as e:
#         main_placeholder.text(f"Error fetching data: {e}")
#         print(f"Error: {e}")

#     time.sleep(2)

# query = main_placeholder.text_input("Question: ")
# if query:
#     if os.path.exists(file_path):
#         with open(file_path, "rb") as f:
#             vectorstore = pickle.load(f)
#             chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
#             result = chain({"question": query}, return_only_outputs=True)
#             st.header("Answer")
#             st.write(result["answer"])

#             # Display sources if available
#             sources = result.get("sources", "")
#             if sources:
#                 st.subheader("Sources:")
#                 sources_list = sources.split("\n")  # Split the sources by newline
#                 for source in sources_list:
#                     st.write(source)


import os
import streamlit as st
import pickle
import time
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

load_dotenv()  # load environment variables from .env file (especially OPENAI_API_KEY)

st.title("StockSage: News Research Analyzer 📈")
st.sidebar.title("News Article URLs")

# Get URLs from sidebar input
urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i + 1}")
    if url:
        urls.append(url)

process_url_clicked = st.sidebar.button("Inspect URLs")
file_path = "faiss-store-openai.pkl"

main_placeholder = st.empty()
llm = OpenAI(temperature=0.9, max_tokens=500)

# Initialize Selenium WebDriver
def fetch_page_with_selenium(url):
    options = Options()
    options.headless = True  # Run browser in headless mode (no GUI)
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    driver.get(url)
    
    # Wait for page to fully load (you can adjust this based on your page load times)
    driver.implicitly_wait(10)

    # Extract the page content (you can use other selectors if needed)
    page_content = driver.find_element(By.TAG_NAME, 'body').text
    driver.quit()  # Close the browser once done
    return page_content

if process_url_clicked:
    # Ensure that URLs are provided
    if not urls:
        main_placeholder.text("Please provide at least one URL.")
    else:
        # Load data from the provided URLs using Selenium
        all_data = []
        for url in urls:
            try:
                page_content = fetch_page_with_selenium(url)  # Fetch the content using Selenium
                print(f"Fetched Content from {url}: {page_content[:200]}...")  # Print the first 200 chars for debugging
                if page_content:
                    all_data.append(page_content)
            except Exception as e:
                main_placeholder.text(f"Error fetching data from {url}: {e}")
                print(f"Error: {e}")
        
        if not all_data:
            main_placeholder.text("No data fetched from the URLs.")
        else:
            main_placeholder.text("Data Loaded Successfully >>> ✅✅✅")

            # Split the data into chunks for further processing
            text_splitter = RecursiveCharacterTextSplitter(
                separators=['\n\n', '\n', '.', ','],
                chunk_size=500
            )
            docs = text_splitter.split_documents(all_data)
            print(f"Docs after splitting: {docs}")  # Debugging line to check the split documents

            if not docs:
                main_placeholder.text("No documents found after splitting.")
            else:
                # Create FAISS index for document embeddings
                embeddings = OpenAIEmbeddings()
                vectorstore_openai = FAISS.from_documents(docs, embeddings)
                print(f"Vectorstore created: {vectorstore_openai}")

                # Save the FAISS index to disk
                with open(file_path, "wb") as f:
                    pickle.dump(vectorstore_openai, f)
                main_placeholder.text("Embedding Vector Creation >>> ✅✅✅")

    time.sleep(2)

# Handle the query and provide answers using the stored FAISS index
query = main_placeholder.text_input("Question: ")
if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
            result = chain({"question": query}, return_only_outputs=True)
            st.header("Answer")
            st.write(result["answer"])

            # Display sources if available
            sources = result.get("sources", "")
            if sources:
                st.subheader("Sources:")
                sources_list = sources.split("\n")  # Split the sources by newline
                for source in sources_list:
                    st.write(source)

