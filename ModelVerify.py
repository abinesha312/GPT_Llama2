import os

# Specify the model directory path
model_directory = '/home/haridoss/Models/llama-models/models/llama3/meta-llama/Meta-Llama-3-8B-Instruct'

# Check if the directory exists and list its contents
if os.path.exists(model_directory):
    contents = os.listdir(model_directory)
    print("Directory exists. Contents:")
    print(contents)
else:
    print("Directory does not exist")