import streamlit as st
from PyPDF2 import PdfReader #libreria para leer pdf
from langchain.text_splitter import RecursiveCharacterTextSplitter#libreria para splitear textos de archivos pdf
import os

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI #para encrustar texto
import google.generativeai as genai

from langchain_community.vectorstores import FAISS #Para los embeddings / vectores
from langchain_google_genai import ChatGoogleGenerativeAI #
from langchain.chains.question_answering import load_qa_chain #para encadenar prompts
from langchain.prompts import PromptTemplate #para crear templates del prompt
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key = os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    text = ""
    # recorre todos los archivos PDF cargados
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        # recorre todas las páginas en un PDF
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    # crear  objeto RecursiveCharacterTextSplitter con un chunk size especficio 
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 10000, chunk_overlap = 1000)
    # separar el texto en chunks
    chunks = text_splitter.split_text(text)

    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001") # google embeddings
    vector_store = FAISS.from_texts(text_chunks,embeddings) # utiliza el objeto de embedding en el texto dividido de los documentos PDF
    vector_store.save_local("faiss_index") # guardar los embeddings en local

def get_conversation_chain():

    # definir un prompt para el chatbot y darle instrucciones
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model = "gemini-pro", temperature = 0.3) # crear objeto de gemini pro /

    prompt = PromptTemplate(template = prompt_template, input_variables= ["context","question"])

    chain = load_qa_chain(model,chain_type="stuff",prompt = prompt)

    return chain

def user_input(user_question):
    # user_question es la pregunta dentro del input
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    #db cargar la db faiss en local
    new_db = FAISS.load_local("faiss_index", embeddings)

    #usando busqueda por similitud tener la respuesta basada en el input
    docs = new_db.similarity_search(user_question)

    chain = get_conversation_chain()

    
    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)

    print(response)
    st.write("Reply: ", response["output_text"])

def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using Gemini")

    user_question = st.text_input("Ask a Question:")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")


if __name__ == "__main__":
    main()
