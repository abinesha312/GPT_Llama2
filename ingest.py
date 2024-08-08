import requests
from bs4 import BeautifulSoup
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import time

DATA_URL = 'https://studentaffairs.unt.edu/news/dungeons-dragons-and-therapy-an-adventure-begins.html'  # Replace with the actual URL
DB_FAISS_PATH = 'vectorstore/db_faiss'
SCRAPED_DATA_FILE = 'scraped_data.txt'
SCRAPED_URLS_FILE = 'scraped_urls.txt'
TIME_LOG_FILE = 'time_log.txt'
MAX_WEBSITES = 200

scraped_urls = set()

def scrape_website(url, depth=1, max_depth=2):
    """
    Scrape the website for text and URLs recursively up to a specified depth.
    """
    global scraped_urls
    if len(scraped_urls) >= MAX_WEBSITES:
        return [], []

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Failed to scrape {url}: {e}")
        return [], []

    soup = BeautifulSoup(response.text, 'html.parser')
    
    paragraphs = soup.find_all('p')
    text_data = [para.get_text() for para in paragraphs]
    
    links = soup.find_all('a', href=True)
    urls = [link['href'] for link in links if link['href'].startswith('http')]
    
    scraped_urls.add(url)
    
    if depth < max_depth:
        for link in urls:
            if len(scraped_urls) >= MAX_WEBSITES:
                break
            try:
                sub_text_data, sub_urls = scrape_website(link, depth + 1, max_depth)
                text_data.extend(sub_text_data)
                urls.extend(sub_urls)
            except Exception as e:
                print(f"Failed to scrape {link}: {e}")
    
    return text_data, urls

def create_vector_db():
    start_time = time.time()
    
    text_data, urls = scrape_website(DATA_URL)
    with open(SCRAPED_DATA_FILE, 'w', encoding='utf-8') as file:
        for text in text_data:
            file.write(text + '\n')
    
    with open(SCRAPED_URLS_FILE, 'w', encoding='utf-8') as file:
        for url in scraped_urls:
            file.write(url + '\n')

    with open(SCRAPED_DATA_FILE, 'r', encoding='utf-8') as file:
        text_data = file.readlines()
    
    documents = [Document(page_content=text.strip()) for text in text_data]
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': 'cpu'})
    db = FAISS.from_documents(texts, embeddings)
    db.save_local(DB_FAISS_PATH)
    
    end_time = time.time() 
    response_time = end_time - start_time
    
    with open(TIME_LOG_FILE, 'w', encoding='utf-8') as file:
        file.write(f"Start time: {time.ctime(start_time)}\n")
        file.write(f"End time: {time.ctime(end_time)}\n")
        file.write(f"Response time: {response_time:.6f} seconds\n")
    
    print(f"Response time: {response_time:.6f} seconds")

if __name__ == "__main__":
    create_vector_db()
