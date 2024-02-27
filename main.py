import boto3
import streamlit as st
from PIL import Image
import numpy as np
from langchain_community.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock
from langchain.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA


import boto3

# Specify your AWS credentials
aws_access_key_id = ''
aws_secret_access_key = ''
#aws_session_token = 'YOUR_SESSION_TOKEN'  # Only required if using temporary security credentials

# Specify the AWS region
region_name = 'us-east-1'

# Create the boto3 client with your credentials
aws_bedrock = boto3.client(
    service_name="bedrock-runtime",
    region_name=region_name,
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    #aws_session_token=aws_session_token  # Omit this if not using temporary security credentials
)



embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1",client=aws_bedrock)

def read_csv():
    loader = CSVLoader(file_path='messages1.csv', encoding="utf-8", csv_args={
                    'delimiter': ','})
    data = loader.load()
    return data

def get_vector_store(data):
    vectorstore_faiss=FAISS.from_documents(
        data,
        embeddings
    )
    vectorstore_faiss.save_local('vector_store')

def llama2_model():
    llm=Bedrock(model_id="meta.llama2-70b-chat-v1",client=aws_bedrock,
                model_kwargs={'max_gen_len':512})
    
    return llm

prompt_template = """

System Prompt: Use the following pieces of context to provide a 
short answer to the question. Answer within 1 or 2 statements. If you don't know the answer, 
just say that you don't know, don't try to make up an answer. 
You are Praneeth Korukonda. Answer how Praneeth Korukonda would answer it.
<context>
{context}
</context

Question: {question}

Praneeth:"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

def get_response_llm(llm,vectorstore,query):
    qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": 3}
    ),
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)
    answer=qa({"query":query})
    return answer['result']

def main():
    st.set_page_config("Linkedin Chatbot")
    
    st.header("AI clone Of Praneeth")

    user_question = st.text_input("Ask a Question for Praneeth")

    with st.sidebar:
        st.title("Update Or Create Vector Store:")
        
        if st.button("Vectors Update"):
            with st.spinner("Processing..."):
                data = read_csv()
                print(data)
                get_vector_store(data)
                st.success("Done")

    if st.button("Ask"):
        with st.spinner("Processing..."):
            vector_store = FAISS.load_local("vector_store", embeddings)
            llm=llama2_model()
            
            #faiss_index = get_vector_store(docs)
            st.write(get_response_llm(llm,vector_store,user_question))
            st.success("Done")


if __name__ == "__main__":
    main()