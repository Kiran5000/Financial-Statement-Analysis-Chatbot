import streamlit as st
from PyPDF2 import PdfReader
from pdfminer.high_level import extract_text
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Function to extract text from PDF files
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        try:
            text += extract_text(pdf)
        except Exception as e:
            st.error(f"Error processing PDF file: {e}")
    return text

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create vector store
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Function to handle user input and generate response
def user_input(user_question, show_additional_message):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

        docs = new_db.similarity_search(user_question)
        chain = get_conversational_chain()
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        
        # Analyze response and add additional points
        analyzed_response = response["output_text"]
        if show_additional_message:
            analyzed_response = analyze_response(analyzed_response)
        
        # Display the analyzed response to the user
        st.write("Reply:", analyzed_response)
    except Exception as e:
        st.error(f"Error processing user input: {e}")

# Function to analyze response and add additional points
def analyze_response(response):
    # Perform analysis here and add additional points
    # For example, you can extract key insights, highlight important information, or provide recommendations based on the response
    
    # Sample analysis: Adding a generic message
    additional_points = "Additional analysis: This is a generic analysis message."
    
    # Combine the original response with additional points
    analyzed_response = f"{response}\n\n{additional_points}"
    
    return analyzed_response

# Function to load conversational chain
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Main function
def main():
    st.set_page_config("Financial Statements Chat")
    st.header("Chat with Financial Statements PDF using Gemini💁")

    # User input for PDF files
    pdf_docs = st.file_uploader("Upload Financial Statements PDF Files", accept_multiple_files=True)
    if pdf_docs:
        raw_text = get_pdf_text(pdf_docs)
        text_chunks = get_text_chunks(raw_text)
        get_vector_store(text_chunks)
        st.success("Financial Statements PDF Files Processed Successfully")

    # User input for questions
    user_question = st.text_input("Ask a Question related to Financial Statements")
    show_additional_message = st.checkbox("Show Additional Analysis", value=False)
    if user_question:
        if st.button("Get Result"):
            user_input(user_question, show_additional_message)

    # Additional interaction prompts
    if user_question.lower().strip() in ["keywords", "topics", "description"]:
        st.info("Please provide some keywords, topics, or description related to the financial information you are looking for.")
    elif user_question.lower().strip() in ["questions", "assistance", "discuss"]:
        st.info("Feel free to ask any questions or specify how I can assist you further with the financial statements.")
        
# Entry point of the application
if __name__ == "__main__":
    main()
