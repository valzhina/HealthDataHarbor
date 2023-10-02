# os module provides a way to work with Operating System(win/mac..)
import os


# Used to buils the app
import streamlit as st

# Used to build LLM Workflow
from langchain.llms import OpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings


from langchain.agents.agent_toolkits import (
    create_vectorstore_agent,
    VectorStoreToolkit,
    VectorStoreInfo
)

# Sets an environment variable named OPENAI_API_KEY to the value of the apikey
# os.environ['OPENAI_API_KEY'] = apikey 


#App Framework
st.title('👩‍⚕️ Personal Health GPT Blood Tests Collector')
 
#Creates an instance of OpenAI service(Llms) temperature sets creativity level
llm = OpenAI(temperature=0.9) 
embeddings = OpenAIEmbeddings()

# uploaded_file = st.file_uploader("Choose a file", type=["pdf"])

# Place for prompt
prompt = st.text_input('Your request goes here')

#Creates PDF loader
loader = PyPDFLoader('annualtestreport.pdf') 

#Splits pages
pages = loader.load_and_split()

#Load PDFs into ChromaDB
store = Chroma.from_documents(pages, collection_name='annualtestreport.pdf', embedding=embeddings)

vectorStore_info = VectorStoreInfo(
    name="annual_testreport",
    description="a blood work annual report as PDF",
    vectorstore=store
)

#Converts teh doc into a langchain kit
toolkit = VectorStoreToolkit(vectorstore_info=vectorStore_info)

#Adds the toolkit to an LC
agent_executor = create_vectorstore_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True
)


#Screen response 
if prompt:

    #Passes the prompt to the LLM
    response = agent_executor.run(prompt)
    st.write(response)

    with st.expander('Document Similarity Search'):
        #finds the relevante pages
        search = store.similarity_search_with_score(prompt)

