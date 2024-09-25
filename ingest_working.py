import urllib.parse
import urllib3
import json
import aiohttp
import asyncio
import torch
import time
import re
import signal
from bs4 import BeautifulSoup
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

DB_FAISS_PATH = 'vectorstore/db_faiss'
SCRAPED_DATA_FILE = 'scraped_data.json'
SCRAPED_URLS_FILE = 'scraped_urls.json'
TIME_LOG_FILE = 'time_log.json'
MAX_WEBSITES = 20
TIMEOUT = 10
RETRIES = 3
MAX_DEPTH = 5000
scraped_urls = set()
scraped_data = {}

UNT_DOMAINS = [
    'https://www.unt.edu',
    'https://online.unt.edu',
    'https://library.unt.edu',
    'https://digital.library.unt.edu',
    'https://texashistory.unt.edu',
    'https://cybercemetery.unt.edu',
    'https://learn.unt.edu',
    'https://my.unt.edu',
    'https://studentaffairs.unt.edu',
    'https://registrar.unt.edu',
    'https://admissions.unt.edu',
    'https://financialaid.unt.edu',
    'https://housing.unt.edu',
    'https://dining.unt.edu',
    'https://it.unt.edu',
    'https://recsports.unt.edu',
    'https://business.unt.edu',
    'https://coe.unt.edu',
    'https://engineering.unt.edu',
    'https://music.unt.edu',
    'https://cvad.unt.edu',
    'https://cas.unt.edu',
    'https://hsc.unt.edu',
    'https://cob.unt.edu',
    'https://class.unt.edu',
    'https://math.unt.edu',
    'https://physics.unt.edu',
    'https://chemistry.unt.edu',
    'https://biology.unt.edu',
    'https://psychology.unt.edu',
    'https://history.unt.edu',
    'https://english.unt.edu',
    'https://philosophy.unt.edu',
    'https://economics.unt.edu',
    'https://geography.unt.edu',
    'https://sociology.unt.edu',
    'https://politicalscience.unt.edu',
    'https://linguistics.unt.edu',
    'https://worldlanguages.unt.edu',
    'https://communication.unt.edu',
    'https://journalism.unt.edu',
    'https://vpn.unt.edu',
    'https://sso.unt.edu',
    'https://eaglemail.unt.edu',
    'https://jobs.unt.edu',
    'https://careers.unt.edu',
    'https://transportation.unt.edu',
    'https://facilities.unt.edu',
    'https://ipa.unt.edu',
    'https://research.unt.edu',
    'https://ams.unt.edu',
    'https://its.unt.edu',
    'https://aits.unt.edu',
    'https://ecs.unt.edu',
    'https://guides.library.unt.edu',
    'https://discover.library.unt.edu',
    'https://findingaids.library.unt.edu',
    'https://journals.library.unt.edu',
    'https://esports.library.unt.edu',
    'https://untpress.unt.edu',
    'https://northtexan.unt.edu',
    'https://tams.unt.edu',
    'https://honors.unt.edu'
]

def is_valid_response(status_code):
    return 200 <= status_code <= 205

async def fetch_page_content(session, url, timeout=TIMEOUT, retries=RETRIES):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36'
    }
    attempt = 0
    while attempt < retries:
        try:
            encoded_url = urllib.parse.quote(url, safe=':/')
            print(f"Fetching URL: {encoded_url}")
            async with session.get(encoded_url, headers=headers, timeout=timeout) as response:
                if is_valid_response(response.status):
                    html_content = await response.text(errors='replace')
                    if "<html" not in html_content.lower():
                        print(f"Warning: The content fetched from {url} does not appear to be valid HTML.")
                        return None
                    return html_content
                else:
                    print(f"Invalid response for {url}: {response.status}")
                    return None
        except asyncio.TimeoutError:
            attempt += 1
            print(f"Timeout while fetching {url}, attempt {attempt} of {retries}")
            if attempt >= retries:
                print(f"Skipping {url} after {retries} attempts due to timeout.")
                return None
            await asyncio.sleep(2)
        except aiohttp.ClientError as e:
            print(f"Failed to scrape {url}: {e}")
            return None
        
def parse_content(html_content, url):
    soup = BeautifulSoup(html_content, 'lxml')
    body = soup.find('body')
    body_elements = []
    if body:
        for element in body.stripped_strings:
            if element.strip():
                body_elements.append(element.strip())
    
    paragraphs = soup.find_all('p')
    text_data = []
    for para in paragraphs:
        text = para.get_text(strip=True)
        links = para.find_all('a', href=True)
        for link in links:
            link_text = link.get_text(strip=True)
            link_url = urllib.parse.urljoin(url, link['href'])
            text = text.replace(link_text, f'({link_text}, URL:"{link_url}")')
        text_data.append(text)
    
    text_data = " ".join(text_data)
    
    images = soup.find_all('img', src=True)
    image_data = [img['src'] for img in images]
    headings = {f"h{i}": [h.get_text(strip=True) for h in soup.find_all(f'h{i}')] for i in range(1, 7)}
    lists = [ul.get_text(separator=", ", strip=True) for ul in soup.find_all('ul')]
    urls = extract_links(soup, url)
    DataCount = len(text_data)
    
    structured_data = {
        "Source_URL": url,
        "Body": body_elements,
        "text": text_data,
        "images": image_data,
        "headings": headings,
        "lists": lists,
        "URLs": urls,
        "Count": DataCount
    }
    return structured_data

# def parse_content(html_content, url):
#     soup = BeautifulSoup(html_content, 'lxml')
#     body = soup.find('body')
#     body_elements = []
#     if body:
#         for element in body.stripped_strings:
#             if element.strip():  # Only add non-empty strings
#                 body_elements.append(element.strip())
    
#     paragraphs = soup.find_all('p')
#     text_data = " ".join(para.get_text(strip=True) for para in paragraphs)
#     images = soup.find_all('img', src=True)
#     image_data = [img['src'] for img in images]
#     headings = {f"h{i}": [h.get_text(strip=True) for h in soup.find_all(f'h{i}')] for i in range(1, 7)}
#     lists = [ul.get_text(separator=", ", strip=True) for ul in soup.find_all('ul')]
#     urls = extract_links(soup, url)
#     DataCount = len(text_data)
    
#     structured_data = {
#         "Source_URL": url,
#         "Body": body_elements,
#         "text": text_data,
#         "images": image_data,
#         "headings": headings,
#         "lists": lists,
#         "URLs": urls,
#         "Count": DataCount
#     }
#     return structured_data

def extract_links(soup, base_url):
    links = soup.find_all('a', href=True)
    unt_pattern = re.compile(r'^https?://(?:[\w-]+\.)*unt\.edu(?:/[\w-]+)*(?:\?[\w=&]+)?')
    
    extracted_links = []
    for link in links:
        href = link['href']
        full_url = urllib.parse.urljoin(base_url, href)
        if unt_pattern.match(full_url):
            extracted_links.append(full_url)
    
    return extracted_links

def load_scraped_urls():
    global scraped_urls
    try:
        with open(SCRAPED_URLS_FILE, 'r', encoding='utf-8') as file:
            urls_data = json.load(file)
            scraped_urls = set(urls_data.values())
    except FileNotFoundError:
        scraped_urls = set()
    print(f"Loaded {len(scraped_urls)} previously scraped URLs.")

def save_scraped_data():
    numbered_data = {f"URL_{str(i+1).zfill(2)}": data for i, data in enumerate(scraped_data.values())}
    with open(SCRAPED_DATA_FILE, 'w', encoding='utf-8') as file:
        json.dump(numbered_data, file, indent=4, ensure_ascii=False)

def save_scraped_urls():
    urls_data = {f"URL_{str(i+1).zfill(2)}": url for i, url in enumerate(scraped_urls)}
    with open(SCRAPED_URLS_FILE, 'w', encoding='utf-8') as file:
        json.dump(urls_data, file, indent=4, ensure_ascii=False)

def create_documents(scraped_data):
    documents = []
    for url, data in scraped_data.items():
        if data['text']:
            doc = Document(
                page_content=data['text'],
                metadata={"source": data['Source_URL'], "word_count": data['Count']}
            )
            documents.append(doc)
    return documents

def build_vector_store(documents):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    embeddings = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2',
        model_kwargs={'device': device}
    )
    db = FAISS.from_documents(documents, embeddings)
    db.save_local(DB_FAISS_PATH)

def log_time(start_time, end_time):
    response_time = end_time - start_time
    time_log = {
        "start_time": time.ctime(start_time),
        "end_time": time.ctime(end_time),
        "response_time": f"{response_time:.6f} seconds"
    }
    with open(TIME_LOG_FILE, 'w', encoding='utf-8') as file:
        json.dump(time_log, file, indent=4, ensure_ascii=False)
    print(f"Response time: {response_time:.6f} seconds")

async def scrape_domain_and_related(session, start_url, depth=0):
    if start_url in scraped_urls or depth > MAX_DEPTH or len(scraped_urls) >= MAX_WEBSITES:
        return {}

    print(f"Scraping: {start_url} (Depth: {depth})")
    html_content = await fetch_page_content(session, start_url)
    if html_content is None:
        return {}

    structured_data = parse_content(html_content, start_url)
    scraped_urls.add(start_url)
    scraped_data[start_url] = structured_data
    data = {start_url: structured_data}

    if depth < MAX_DEPTH:
        related_urls = extract_links(BeautifulSoup(html_content, 'lxml'), start_url)
        tasks = []
        for url in related_urls:
            if url not in scraped_urls and len(scraped_urls) < MAX_WEBSITES:
                tasks.append(scrape_domain_and_related(session, url, depth + 1))
        
        if tasks:
            results = await asyncio.gather(*tasks)
            for result in results:
                data.update(result)

    return data

async def scrape_all_domains():
    async with aiohttp.ClientSession() as session:
        for domain in UNT_DOMAINS:
            print(f"Starting to scrape domain: {domain}")
            domain_data = await scrape_domain_and_related(session, domain)
            scraped_data.update(domain_data)
            save_scraped_data()
            save_scraped_urls()
            print(f"Completed scraping domain: {domain}")
            print(f"Total URLs scraped: {len(scraped_urls)}")
            if len(scraped_urls) >= MAX_WEBSITES:
                print(f"Reached maximum number of websites ({MAX_WEBSITES}). Stopping.")
                break
            await asyncio.sleep(5)  # Add a small delay between domains

def signal_handler(signum, frame):
    print("Received termination signal. Saving data and exiting...")
    save_scraped_data()
    save_scraped_urls()
    documents = create_documents(scraped_data)
    if documents:
        build_vector_store(documents)
    exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

async def create_vector_db():
    start_time = time.time()
    try:
        load_scraped_urls()
        await asyncio.wait_for(scrape_all_domains(), timeout=25200)  # 2 hour timeout
        documents = create_documents(scraped_data)
        if documents:
            build_vector_store(documents)
        else:
            print("No valid documents to build vector store.")
    except asyncio.TimeoutError:
        print("Scraping process timed out after 2 hours.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        end_time = time.time()
        log_time(start_time, end_time)

if __name__ == "__main__":
    asyncio.run(create_vector_db())


#--------------------------------


# import urllib.parse
# import urllib3
# import json
# import aiohttp
# import asyncio
# import torch
# import time
# import re
# import signal
# from bs4 import BeautifulSoup
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_core.documents import Document

# urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# DB_FAISS_PATH = 'vectorstore/db_faiss'
# SCRAPED_DATA_FILE = 'scraped_data.json'
# SCRAPED_URLS_FILE = 'scraped_urls.json'
# TIME_LOG_FILE = 'time_log.json'
# MAX_WEBSITES = 20
# TIMEOUT = 10
# RETRIES = 3
# MAX_DEPTH = 2
# scraped_urls = set()    
# scraped_data = {}

# UNT_DOMAINS = [
#     'https://www.unt.edu',
#     'https://online.unt.edu',
#     'https://library.unt.edu',
#     'https://digital.library.unt.edu',
#     'https://texashistory.unt.edu',
#     'https://cybercemetery.unt.edu',
#     'https://learn.unt.edu',
#     'https://my.unt.edu',
#     'https://studentaffairs.unt.edu',
#     'https://registrar.unt.edu',
#     'https://admissions.unt.edu',
#     'https://financialaid.unt.edu',
#     'https://housing.unt.edu',
#     'https://dining.unt.edu',
#     'https://it.unt.edu',
#     'https://recsports.unt.edu',
#     'https://business.unt.edu',
#     'https://coe.unt.edu',
#     'https://engineering.unt.edu',
#     'https://music.unt.edu',
#     'https://cvad.unt.edu',
#     'https://cas.unt.edu',
#     'https://hsc.unt.edu',
#     'https://cob.unt.edu',
#     'https://class.unt.edu',
#     'https://math.unt.edu',
#     'https://physics.unt.edu',
#     'https://chemistry.unt.edu',
#     'https://biology.unt.edu',
#     'https://psychology.unt.edu',
#     'https://history.unt.edu',
#     'https://english.unt.edu',
#     'https://philosophy.unt.edu',
#     'https://economics.unt.edu',
#     'https://geography.unt.edu',
#     'https://sociology.unt.edu',
#     'https://politicalscience.unt.edu',
#     'https://linguistics.unt.edu',
#     'https://worldlanguages.unt.edu',
#     'https://communication.unt.edu',
#     'https://journalism.unt.edu',
#     'https://vpn.unt.edu',
#     'https://sso.unt.edu',
#     'https://eaglemail.unt.edu',
#     'https://jobs.unt.edu',
#     'https://careers.unt.edu',
#     'https://transportation.unt.edu',
#     'https://facilities.unt.edu',
#     'https://ipa.unt.edu',
#     'https://research.unt.edu',
#     'https://ams.unt.edu',
#     'https://its.unt.edu',
#     'https://aits.unt.edu',
#     'https://ecs.unt.edu',
#     'https://guides.library.unt.edu',
#     'https://discover.library.unt.edu',
#     'https://findingaids.library.unt.edu',
#     'https://journals.library.unt.edu',
#     'https://esports.library.unt.edu',
#     'https://untpress.unt.edu',
#     'https://northtexan.unt.edu',
#     'https://tams.unt.edu',
#     'https://honors.unt.edu'
# ]

# def is_valid_response(status_code):
#     return 200 <= status_code <= 205

# async def fetch_page_content(session, url, timeout=TIMEOUT, retries=RETRIES):
#     headers = {
#         'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36'
#     }
#     attempt = 0
#     while attempt < retries:
#         try:
#             encoded_url = urllib.parse.quote(url, safe=':/')
#             print(f"Fetching URL: {encoded_url}")
#             async with session.get(encoded_url, headers=headers, timeout=timeout) as response:
#                 if is_valid_response(response.status):
#                     html_content = await response.text(errors='replace')
#                     if "<html" not in html_content.lower():
#                         print(f"Warning: The content fetched from {url} does not appear to be valid HTML.")
#                         return None
#                     return html_content
#                 else:
#                     print(f"Invalid response for {url}: {response.status}")
#                     return None
#         except asyncio.TimeoutError:
#             attempt += 1
#             print(f"Timeout while fetching {url}, attempt {attempt} of {retries}")
#             if attempt >= retries:
#                 print(f"Skipping {url} after {retries} attempts due to timeout.")
#                 return None
#             await asyncio.sleep(2)
#         except aiohttp.ClientError as e:
#             print(f"Failed to scrape {url}: {e}")
#             return None

# def parse_content(html_content, url):
#     soup = BeautifulSoup(html_content, 'lxml')
#     body = soup.find('body')
#     body_elements = []
#     if body:
#         for element in body.stripped_strings:
#             if element.strip():  # Only add non-empty strings
#                 body_elements.append(element.strip())
    
#     paragraphs = soup.find_all('p')
#     text_data = " ".join(para.get_text(strip=True) for para in paragraphs)
#     images = soup.find_all('img', src=True)
#     image_data = [img['src'] for img in images]
#     headings = {f"h{i}": [h.get_text(strip=True) for h in soup.find_all(f'h{i}')] for i in range(1, 7)}
#     lists = [ul.get_text(separator=", ", strip=True) for ul in soup.find_all('ul')]
#     urls = extract_links(soup, url)
#     DataCount = len(text_data)
#     structured_data = {
#         "Source_URL": url,
#         "Body": body_elements,
#         "text": text_data,
#         "images": image_data,
#         "headings": headings,
#         "lists": lists,
#         "URLs": urls,
#         "Count": DataCount
#     }
#     return structured_data

# def extract_links(soup, base_url):
#     links = soup.find_all('a', href=True)
#     unt_pattern = re.compile(r'^https?://(?:[\w-]+\.)*unt\.edu(?:/[\w-]+)*(?:\?[\w=&]+)?')
    
#     extracted_links = []
#     for link in links:
#         href = link['href']
#         full_url = urllib.parse.urljoin(base_url, href)
#         if unt_pattern.match(full_url):
#             extracted_links.append(full_url)
    
#     return extracted_links

# def load_scraped_urls():
#     global scraped_urls
#     try:
#         with open(SCRAPED_URLS_FILE, 'r', encoding='utf-8') as file:
#             urls_data = json.load(file)
#             scraped_urls = set(urls_data.values())
#     except FileNotFoundError:
#         scraped_urls = set()
#     print(f"Loaded {len(scraped_urls)} previously scraped URLs.")

# def save_scraped_data():
#     numbered_data = {f"URL_{str(i+1).zfill(2)}": data for i, data in enumerate(scraped_data.values())}
#     with open(SCRAPED_DATA_FILE, 'w', encoding='utf-8') as file:
#         json.dump(numbered_data, file, indent=4, ensure_ascii=False)

# def save_scraped_urls():
#     urls_data = {f"URL_{str(i+1).zfill(2)}": url for i, url in enumerate(scraped_urls)}
#     with open(SCRAPED_URLS_FILE, 'w', encoding='utf-8') as file:
#         json.dump(urls_data, file, indent=4, ensure_ascii=False)

# def create_documents(scraped_data):
#     documents = []
#     for url, data in scraped_data.items():
#         if data['text']:
#             doc = Document(
#                 page_content=data['text'],
#                 metadata={"source": data['Source_URL'], "word_count": data['Count']}
#             )
#             documents.append(doc)
#     return documents

# def build_vector_store(documents):
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     embeddings = HuggingFaceEmbeddings(
#         model_name='sentence-transformers/all-MiniLM-L6-v2',
#         model_kwargs={'device': device}
#     )
#     db = FAISS.from_documents(documents, embeddings)
#     db.save_local(DB_FAISS_PATH)

# def log_time(start_time, end_time):
#     response_time = end_time - start_time
#     time_log = {
#         "start_time": time.ctime(start_time),
#         "end_time": time.ctime(end_time),
#         "response_time": f"{response_time:.6f} seconds"
#     }
#     with open(TIME_LOG_FILE, 'w', encoding='utf-8') as file:
#         json.dump(time_log, file, indent=4, ensure_ascii=False)
#     print(f"Response time: {response_time:.6f} seconds")

# async def scrape_domain_and_related(session, start_url, depth=0):
#     if start_url in scraped_urls or depth > MAX_DEPTH or len(scraped_urls) >= MAX_WEBSITES:
#         return {}

#     print(f"Scraping: {start_url} (Depth: {depth})")
#     html_content = await fetch_page_content(session, start_url)
#     if html_content is None:
#         return {}

#     structured_data = parse_content(html_content, start_url)
#     scraped_urls.add(start_url)
#     scraped_data[start_url] = structured_data
#     data = {start_url: structured_data}

#     if depth < MAX_DEPTH:
#         related_urls = extract_links(BeautifulSoup(html_content, 'lxml'), start_url)
#         tasks = []
#         for url in related_urls:
#             if url not in scraped_urls and len(scraped_urls) < MAX_WEBSITES:
#                 tasks.append(scrape_domain_and_related(session, url, depth + 1))
        
#         if tasks:
#             results = await asyncio.gather(*tasks)
#             for result in results:
#                 data.update(result)

#     return data

# async def scrape_all_domains():
#     async with aiohttp.ClientSession() as session:
#         for domain in UNT_DOMAINS:
#             print(f"Starting to scrape domain: {domain}")
#             domain_data = await scrape_domain_and_related(session, domain)
#             scraped_data.update(domain_data)
#             save_scraped_data()
#             save_scraped_urls()
#             print(f"Completed scraping domain: {domain}")
#             print(f"Total URLs scraped: {len(scraped_urls)}")
#             if len(scraped_urls) >= MAX_WEBSITES:
#                 print(f"Reached maximum number of websites ({MAX_WEBSITES}). Stopping.")
#                 break
#             await asyncio.sleep(5)  # Add a small delay between domains

# def signal_handler(signum, frame):
#     print("Received termination signal. Saving data and exiting...")
#     save_scraped_data()
#     save_scraped_urls()
#     documents = create_documents(scraped_data)
#     if documents:
#         build_vector_store(documents)
#     exit(0)

# signal.signal(signal.SIGINT, signal_handler)
# signal.signal(signal.SIGTERM, signal_handler)

# async def create_vector_db():
#     start_time = time.time()
#     try:
#         load_scraped_urls()
#         await asyncio.wait_for(scrape_all_domains(), timeout=7200)  # 2 hour timeout
#         documents = create_documents(scraped_data)
#         if documents:
#             build_vector_store(documents)
#         else:
#             print("No valid documents to build vector store.")
#     except asyncio.TimeoutError:
#         print("Scraping process timed out after 2 hours.")
#     except Exception as e:
#         print(f"An error occurred: {e}")
#     finally:
#         end_time = time.time()
#         log_time(start_time, end_time)

# if __name__ == "__main__":
#     asyncio.run(create_vector_db())

#-----------------------------------------


# import urllib.parse
# import urllib3
# import json
# import aiohttp
# import asyncio
# import torch
# import time
# import re
# import signal
# from bs4 import BeautifulSoup
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_core.documents import Document

# urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# DATA_URL = 'https://studentaffairs.unt.edu/events/index.html'
# DB_FAISS_PATH = 'vectorstore/db_faiss'
# SCRAPED_DATA_FILE = 'scraped_data.json'
# SCRAPED_URLS_FILE = 'scraped_urls.json'
# TIME_LOG_FILE = 'time_log.json'
# MAX_WEBSITES = 2000
# TIMEOUT = 10
# RETRIES = 3
# scraped_urls = set()
# scraped_data = {}

# def is_valid_response(status_code):
#     return 200 <= status_code <= 205

# async def fetch_page_content(session, url, timeout=TIMEOUT, retries=RETRIES):
#     headers = {
#         'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36'
#     }
#     attempt = 0
#     while attempt < retries:
#         try:
#             encoded_url = urllib.parse.quote(url, safe=':/')
#             print(f"Fetching URL: {encoded_url}")
#             async with session.get(encoded_url, headers=headers, timeout=timeout) as response:
#                 if is_valid_response(response.status):
#                     html_content = await response.text(errors='replace')
#                     if "<html" not in html_content.lower():
#                         print(f"Warning: The content fetched from {url} does not appear to be valid HTML.")
#                         return None
#                     return html_content
#                 else:
#                     print(f"Invalid response for {url}: {response.status}")
#                     return None
#         except asyncio.TimeoutError:
#             attempt += 1
#             print(f"Timeout while fetching {url}, attempt {attempt} of {retries}")
#             if attempt >= retries:
#                 print(f"Skipping {url} after {retries} attempts due to timeout.")
#                 return None
#             await asyncio.sleep(2)
#         except aiohttp.ClientError as e:
#             print(f"Failed to scrape {url}: {e}")
#             return None

# def parse_content(html_content, url):
#     soup = BeautifulSoup(html_content, 'lxml')
#     body = soup.find('body')
#     body_elements = []
#     if body:
#         for element in body.stripped_strings:
#             if element.strip():  # Only add non-empty strings
#                 body_elements.append(element.strip())
    
#     paragraphs = soup.find_all('p')
#     text_data = " ".join(para.get_text(strip=True) for para in paragraphs)
#     images = soup.find_all('img', src=True)
#     image_data = [img['src'] for img in images]
#     headings = {f"h{i}": [h.get_text(strip=True) for h in soup.find_all(f'h{i}')] for i in range(1, 7)}
#     lists = [ul.get_text(separator=", ", strip=True) for ul in soup.find_all('ul')]
#     urls = extract_links(soup, url)
#     DataCount = len(text_data)
#     structured_data = {
#         "Source_URL": url,
#         "Body": body_elements,
#         "text": text_data,
#         "images": image_data,
#         "headings": headings,
#         "lists": lists,
#         "URLs": urls,
#         "Count": DataCount
#     }
#     return structured_data

# def extract_links(soup, base_url):
#     links = soup.find_all('a', href=True)
#     unt_pattern = re.compile(r'^https?://(?:[\w-]+\.)*unt\.edu(?:/[\w-]+)*(?:\?[\w=&]+)?')
    
#     extracted_links = []
#     for link in links:
#         href = link['href']
#         full_url = urllib.parse.urljoin(base_url, href)
#         if unt_pattern.match(full_url):
#             extracted_links.append(full_url)
    
#     return extracted_links

# def load_scraped_urls():
#     global scraped_urls
#     try:
#         with open(SCRAPED_URLS_FILE, 'r', encoding='utf-8') as file:
#             urls_data = json.load(file)
#             scraped_urls = set(urls_data.values())
#     except FileNotFoundError:
#         scraped_urls = set()
#     print(f"Loaded {len(scraped_urls)} previously scraped URLs.")

# def save_scraped_data():
#     numbered_data = {f"URL_{str(i+1).zfill(2)}": data for i, data in enumerate(scraped_data.values())}
#     with open(SCRAPED_DATA_FILE, 'w', encoding='utf-8') as file:
#         json.dump(numbered_data, file, indent=4, ensure_ascii=False)

# def save_scraped_urls():
#     urls_data = {f"URL_{str(i+1).zfill(2)}": url for i, url in enumerate(scraped_urls)}
#     with open(SCRAPED_URLS_FILE, 'w', encoding='utf-8') as file:
#         json.dump(urls_data, file, indent=4, ensure_ascii=False)

# def create_documents(scraped_data):
#     documents = []
#     for url, data in scraped_data.items():
#         if data['text']:
#             doc = Document(
#                 page_content=data['text'],
#                 metadata={"source": data['Source_URL'], "word_count": data['Count']}
#             )
#             documents.append(doc)
#     return documents

# def build_vector_store(documents):
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     embeddings = HuggingFaceEmbeddings(
#         model_name='sentence-transformers/all-MiniLM-L6-v2',
#         model_kwargs={'device': device}
#     )
#     db = FAISS.from_documents(documents, embeddings)
#     db.save_local(DB_FAISS_PATH)

# def log_time(start_time, end_time):
#     response_time = end_time - start_time
#     time_log = {
#         "start_time": time.ctime(start_time),
#         "end_time": time.ctime(end_time),
#         "response_time": f"{response_time:.6f} seconds"
#     }
#     with open(TIME_LOG_FILE, 'w', encoding='utf-8') as file:
#         json.dump(time_log, file, indent=4, ensure_ascii=False)
#     print(f"Response time: {response_time:.6f} seconds")

# async def scrape_single_url(session, url, depth, max_depth):
#     global scraped_urls, scraped_data
#     if url in scraped_urls:
#         print(f"Already visited {url}, skipping.")
#         return {}
#     html_content = await fetch_page_content(session, url)
#     if html_content is None:
#         return {}
#     structured_data = parse_content(html_content, url)
#     scraped_urls.add(url)
#     scraped_data[url] = structured_data
#     data = {url: structured_data}
#     if depth < max_depth:
#         unt_links = extract_links(BeautifulSoup(html_content, 'lxml'), url)
#         tasks = [scrape_single_url(session, link, depth + 1, max_depth) for link in unt_links if link not in scraped_urls and len(scraped_urls) < MAX_WEBSITES]
#         results = await asyncio.gather(*tasks, return_exceptions=True)
#         for result in results:
#             if isinstance(result, dict):
#                 data.update(result)
#     return data

# async def scrape_website(url, depth=1, max_depth=50):
#     async with aiohttp.ClientSession() as session:
#         return await scrape_single_url(session, url, depth, max_depth)

# def signal_handler(signum, frame):
#     print("Received termination signal. Saving data and exiting...")
#     save_scraped_data()
#     save_scraped_urls()
#     documents = create_documents(scraped_data)
#     if documents:
#         build_vector_store(documents)
#     exit(0)

# signal.signal(signal.SIGINT, signal_handler)
# signal.signal(signal.SIGTERM, signal_handler)

# async def create_vector_db():
#     start_time = time.time()
#     try:
#         load_scraped_urls()
#         global scraped_data
#         scraped_data = await asyncio.wait_for(scrape_website(DATA_URL), timeout=3600)  # 1 hour timeout
#         save_scraped_data()
#         save_scraped_urls()
#         documents = create_documents(scraped_data)
#         if documents:
#             build_vector_store(documents)
#         else:
#             print("No valid documents to build vector store.")
#     except asyncio.TimeoutError:
#         print("Scraping process timed out after 1 hour.")
#     except Exception as e:
#         print(f"An error occurred: {e}")
#     finally:
#         end_time = time.time()
#         log_time(start_time, end_time)

# if __name__ == "__main__":
#     asyncio.run(create_vector_db())