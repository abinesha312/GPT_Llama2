import os
import redis

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

DB_FAISS_PATH = 'vectorstore/db_faiss'
redis_client = redis.Redis(host='localhost', port=6379, db=0)

JSON_FILE_PATH = '/home/haridoss/Gradio/scraped_data.json'
WORLD_SIZE = 8

export TF_ENABLE_ONEDNN_OPTS=0
