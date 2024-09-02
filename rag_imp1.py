
import os
import streamlit as st
import bs4
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np
from urllib.parse import urlparse

load_dotenv("keys.env")

# Function to validate URL
def is_valid_url(url):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False

# Polishing the user input if incorrect
def polishing(result):
    prefix = "Corrected Query: "
    if result.startswith(prefix):
        return result[len(prefix):]
    else:
        return result

# Initialize the model
with st.sidebar:
    llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
    bs4_strainer = bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))
    url = st.text_input("ENTER LINK: ")
    image_file = st.file_uploader("Upload Image: ")

if url:
    if not is_valid_url(url):
        st.error("Invalid URL. Please enter a valid URL starting with http:// or https://")
    else:
        with st.spinner("Processing URL..."):
            loader = WebBaseLoader(
                web_paths=(url,),
                bs_kwargs={"parse_only": bs4_strainer}
            )
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
        all_splits = text_splitter.split_documents(docs)
        vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 2})

        template_to_correct_spells = """
            You are an intelligent assistant designed to help users by correcting any errors in their queries. Your task is to read the user's query, identify any spelling or grammatical errors, and return the corrected version of the query. Do not provide an answer, only return the corrected query.
            Corrected Query:
            Original Query: {question}
        """
        prompt_to_correct_spells = ChatPromptTemplate.from_template(template_to_correct_spells)
        chain_to_correct_spells = prompt_to_correct_spells | llm | StrOutputParser()

        contextualize_q_system_prompt = """Given a chat history and the latest user question \
            which might reference context in the chat history, formulate a standalone question \
            which can be understood without the chat history. Do NOT answer the question, \
            just reformulate it if needed and otherwise return it as is."""
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        history_aware_retriever = create_history_aware_retriever(
            llm, retriever, contextualize_q_prompt
        )

        qa_system_prompt = """You are a thoughtful assistant. \
                When asked a question, you generate a step-by-step chain of thought to arrive at the answer. \
                Please provide a detailed response with your reasoning.
            {context}"""
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", qa_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        st.title("Chatbot")

        if 'chat_history' not in st.session_state:
            st.session_state['chat_history'] = []

        for message in st.session_state['chat_history']:
            if message['role'] == 'user':
                st.chat_message("user").write(message['content'])
            else:
                st.chat_message("assistant").write(message['content'])

        if query := st.chat_input("Say something"):
            corrected_query = polishing(chain_to_correct_spells.invoke({"question": query}))
            st.session_state['chat_history'].append({'role': 'user', 'content': corrected_query})
            st.chat_message("user").write(corrected_query)
            with st.spinner("Fetching data..."):
                answer = rag_chain.invoke({"input": corrected_query, "chat_history": st.session_state['chat_history']})
                st.session_state['chat_history'].append({'role': 'assistant', 'content': answer["answer"]})
                st.chat_message("assistant").write(answer["answer"])

elif image_file:
    model = tf.keras.models.load_model('document_classification_model.h5')

    IMG_SIZE = (224, 224)  # Same as used during training

    def prepare_image(img_path):
        """Load and preprocess the image for prediction."""
        img = image.load_img(img_path, target_size=IMG_SIZE)  # Load the image and resize it
        img_array = image.img_to_array(img)  # Convert the image to a numpy array
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array = preprocess_input(img_array)  # Preprocess the image for ResNet50
        return img_array

    def classify_image(img):
        """Classify the image and return the predicted class."""
        img_array = prepare_image(img)
        predictions = model.predict(img_array)  # Predict the class probabilities
        predicted_class = np.argmax(predictions, axis=1)  # Get the index of the highest probability
        class_labels = ['bank', 'invoice']  # Adjust according to your classes
        return class_labels[predicted_class[0]]

    # Example usage
    result = classify_image(image_file)
    st.write(f"Classified Image is: {result}")
