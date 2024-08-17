import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModelForCausalLM, LlamaTokenizer, pipeline
from langchain_core.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import HuggingFacePipeline
import chainlit as cl
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

DB_FAISS_PATH = 'vectorstore/db_faiss'
LOCAL_MODEL_PATH = '/home/haridoss/Models/llama-models/models/llama3/meta-llama/Meta-Llama-3-8B-Instruct'

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def set_custom_prompt():
    custom_prompt_template = """Leverage the information provided below to deliver a comprehensive and accurate response to the user's inquiry.
    If the answer is not known, clearly state that you do not know, and refrain from providing speculative information.

    Context: {context}
    Question: {question}

    Please provide the source where you find the resource of the URL.
    Source URL : 
    """
    prompt = PromptTemplate(
        template=custom_prompt_template, 
        input_variables=['context', 'question']
    )
    return prompt

def load_llm(rank):
    tokenizer = LlamaTokenizer.from_pretrained(LOCAL_MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(LOCAL_MODEL_PATH, torch_dtype=torch.bfloat16)
    model.to(rank)
    model = DDP(model, device_ids=[rank])  # Wrap with DDP
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=256, temperature=0.6, top_p=0.9, device=rank)
    llm = HuggingFacePipeline(pipeline=pipe)
    return llm, tokenizer

def initialize_chain(rank, world_size):
    setup(rank, world_size)
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': f'cuda:{rank}'})
        db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
        llm, _ = load_llm(rank)
        qa_prompt = set_custom_prompt()
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")
        qa = chaatBot_Mode_chain(llm, qa_prompt, db, memory)
        logging.info(f"Chain initialized and stored in session for rank {rank}.")
        return qa
    except Exception as e:
        logging.error(f"Error initializing chain for rank {rank}: {e}")
    finally:
        cleanup()

def chaatBot_Mode_chain(llm, prompt, db, memory):
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=db.as_retriever(search_kwargs={'k': 2}),
        return_source_documents=True,
        memory=memory,
        combine_docs_chain_kwargs={'prompt': prompt}
    )
    return qa_chain

@cl.on_chat_start
async def start():
    world_size = torch.cuda.device_count()
    chain = initialize_chain(0, world_size)

    msg = cl.Message(content="Starting the bot...")
    await msg.send()
    msg.content = "Hi, Welcome to the North Texas University, How can I help you?"
    await msg.update()
    cl.user_session.set("chain", chain)

@cl.on_message
async def main(message):
    chain = cl.user_session.get("chain")
    if chain is None:
        await cl.Message(content="Error: Chain not initialized.").send()
        return

    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = True
    
    res = await chain.acall({'question': message.content}, callbacks=[cb])
    
    answer = res["answer"]
    sources = res["source_documents"]

    if sources:
        answer += f"\nSources:" + str(sources)
    else:
        answer += "\nNo sources found"

    await cl.Message(content=answer).send()

if __name__ == "__main__":
    # Run the Chainlit application
    cl.run()