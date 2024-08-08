from langchain_core.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import ConversationalRetrievalChain
from huggingface_hub import hf_hub_download
from langchain.memory import ConversationBufferMemory

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

def chaatBot_Mode_chain(llm, prompt, db, memory):
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=db.as_retriever(search_kwargs={'k': 2}),
        return_source_documents=True,
        memory=memory,
        combine_docs_chain_kwargs={'prompt': prompt}
    )
    return qa_chain

def load_llm():
    model_path = hf_hub_download(repo_id="TheBloke/Llama-2-7B-Chat-GGML", filename="llama-2-7b-chat.ggmlv3.q8_0.bin")
    llm = CTransformers(
        model=model_path,
        model_type="llama",
        max_new_tokens=512,
        temperature=0.5
    )
    return llm

def qa_bot():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")
    qa = chaatBot_Mode_chain(llm, qa_prompt, db, memory)
    return qa

def respond(message, chat_history):
    qa = qa_bot()
    response = qa.invoke({'question': message})
    answer = response['answer']
    sources = response.get('source_documents', [])
    
    if sources:
        answer += "\n\nSources:"
        for i, source in enumerate(sources, 1):
            answer += f"\n{i}. {source.metadata.get('source', 'Unknown')}"
    else:
        answer += "\n\nNo sources found."
    
    chat_history.append((message, answer))
    return "", chat_history