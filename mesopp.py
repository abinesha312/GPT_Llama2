import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_huggingface import HuggingFacePipeline
from langchain.schema import BaseRetriever, Document
from typing import List
import env as en
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from pydantic import BaseModel, Field
import os

global_llm = None
global_db = None
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

CUSTOM_PROMPT_TEMPLATE = """
You are an expert assistant for the University of North Texas (UNT).
Use the following pieces of information to answer the user's question.

Here's how you should approach answering:

1. Analyze the question:
   - Determine if it's asking for a list, types, or detailed information.
   - Identify the key points and subject matter of the question.

2. Formulate your response:
   - For list or type requests: Create a numbered or bulleted list of items.
   - For detailed information: Provide a comprehensive, paragraph-style answer.

3. Information gathering:
   - Create a list of essential information needed to answer the question.
   - Search UNT official sources for relevant information.
   - Align the gathered information with your list of essentials.

4. Construct your answer:
   - For lists: Present each item clearly, with a brief explanation if necessary.
   - For detailed answers: Organize information logically, using paragraphs for clarity.
   - The answer should feel like a natural interaction between users, not in a question-and-answer format.
   - Ensure all information is specific to the Question which asked.
   - The response should be related to University of North Texas and its departments.
   - Generate the answer only once. Do not repeat information or rephrase the same points.

5. Source citation:
    - Include the source URL for every piece of information you provide.
    - Only use the exact URLs provided in the list below. Do not modify or generate new URLs.
    - Offer up to five most relevant URLs at the end of your answer.
    - Only include the URLs, if any, from the below Context.
    - If no URL is available in the Context then do NOT output any URL.
    - Cite the sentence before it.
   
If the information is not available in the provided context, state:
"I'm unable to find that information from the available resources."

Context: {context}
Question: {question}

Helpful answer:
"""

class GradualLoadingRetriever(BaseRetriever, BaseModel):
    base_retriever: BaseRetriever = Field(exclude=True)
    chunk_size: int = 8  # Number of documents to process at a time

    class Config:
        arbitrary_types_allowed = True

    def get_relevant_documents(self, query: str) -> List[Document]:
        all_docs = self.base_retriever.get_relevant_documents(query)
        processed_docs = []
        for i in range(0, len(all_docs), self.chunk_size):
            chunk = all_docs[i:i+self.chunk_size]
            for doc in chunk:
                url = doc.metadata.get('source_url', '')
                print(f"Processing document with URL: {url}")
            processed_docs.extend(chunk)
        return processed_docs

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        return await self.base_retriever.aget_relevant_documents(query)

def free_gpu_memory():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

def set_custom_prompt():
    return PromptTemplate(
        template=CUSTOM_PROMPT_TEMPLATE,
        input_variables=['context', 'question']
    )

def load_llm():
    global global_llm
    if global_llm is None:
        model_path = '/home/haridoss/Models/Models/llama-models/models/llama3_1/Meta-Llama-3.1-70B-Instruct'
        try:
            with init_empty_weights():
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True,
                    device_map="auto"
                )
            
            model = load_checkpoint_and_dispatch(
                model, 
                model_path, 
                device_map="auto",
                no_split_module_classes=["LlamaDecoderLayer"],
                dtype=torch.float16
            )
            print("Enabling gradient checkpointing...")
            model.gradient_checkpointing_enable()

            tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            hf_pipeline = pipeline(
                'text-generation',
                model=model,
                tokenizer=tokenizer,
                device_map="auto",
                max_new_tokens=1028
            )
            
            global_llm = HuggingFacePipeline(pipeline=hf_pipeline)
        except Exception as e:
            raise
    return global_llm

def load_db():
    global global_db
    if global_db is None:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 1})
        global_db = FAISS.load_local(en.DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    return global_db

class PrintingRetriever(BaseRetriever, BaseModel):
    base_retriever: BaseRetriever = Field(exclude=True)

    class Config:
        arbitrary_types_allowed = True

    def get_relevant_documents(self, query: str) -> List[Document]:
        docs = self.base_retriever.get_relevant_documents(query)
        for doc in docs:
            doc.metadata.get('source_url', '')
        return docs

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        return await self.base_retriever.aget_relevant_documents(query)


def create_chatbot_chain(llm, prompt, db, memory):
    base_retriever = db.as_retriever(search_kwargs={'k': 10})
    gradual_retriever = GradualLoadingRetriever(base_retriever=base_retriever)
    
    res = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=gradual_retriever,
        return_source_documents=True,
        memory=memory,
        combine_docs_chain_kwargs={'prompt': prompt}
    )
    return res

class ChatMessage:
    def __init__(self, role, content):
        self.role = role
        self.content = content

    @classmethod
    def from_dict(cls, data):
        return cls(role=data['role'], content=data['content'])

    def to_dict(self):
        return {"role": self.role, "content": self.content}

def handle_user_message(message, chat_history):
    if chat_history is None:
        chat_history = []

    db = load_db()
    qa_prompt = set_custom_prompt()
    free_gpu_memory()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")
    llm = load_llm()
    qa = create_chatbot_chain(llm, qa_prompt, db, memory)

    with torch.amp.autocast('cuda'):
        response = qa.invoke({
            'question': message,
            'chat_history': chat_history
        })
    answer_marker = "Helpful answer:\n"
    answer_start = response['answer'].find(answer_marker)
    helpful_answer = response['answer'][answer_start + len(answer_marker):].strip() if answer_start != -1 else response['answer']
    source_urls = list(set([doc.metadata.get("source_url", "") for doc in response['source_documents'] if doc.metadata.get("source_url")]))
    if source_urls:
        helpful_answer += "\n\nSources:\n" + "\n".join(source_urls[:5])

    chat_history.append(ChatMessage(role="user", content=message))
    chat_history.append(ChatMessage(role="assistant", content=helpful_answer))
    
    return "", chat_history