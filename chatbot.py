import os
import torch
from transformers import pipeline
from langchain.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import HuggingFacePipeline

DB_FAISS_PATH = 'vectorstore/db_faiss'

custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

def set_custom_prompt():
    prompt = PromptTemplate(
        template=custom_prompt_template, 
        input_variables=['context', 'question']
    )
    return prompt


def load_llm():
    model_path = '/home/haridoss/Models/llama-models/models/llama3/meta-llama/Meta-Llama-3-8B-Instruct'
    
    try:
        hf_pipeline = pipeline('text-generation', model=model_path, device=0) 
        generator = HuggingFacePipeline(pipeline=hf_pipeline)
        return generator
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

generator = load_llm()

def chaatBot_Mode_chain(generator, prompt, db, memory):
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=generator,
        retriever=db.as_retriever(search_kwargs={'k': 2}),
        return_source_documents=True,
        memory=memory,
        combine_docs_chain_kwargs={'prompt': prompt}
    )
    return qa_chain

def handle_user_message(message, chat_history):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cuda:0'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    qa_prompt = set_custom_prompt()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")
    qa = chaatBot_Mode_chain(generator, qa_prompt, db, memory)
    response = qa({'question': message})
    helpful_answer_marker = "Helpful answer:\n"
    answer_start = response['answer'].find(helpful_answer_marker)
    if answer_start != -1:
        helpful_answer = response['answer'][answer_start + len(helpful_answer_marker):].strip()
    else:
        helpful_answer = "Helpful answer not found."
    print("Helpful Answer:", helpful_answer)
    chat_history.append((message, helpful_answer))
    
    return "", chat_history