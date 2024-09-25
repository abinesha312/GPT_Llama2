from huggingface_hub import try_to_load_from_cache

model_id = "nvidia/NV-Embed-v2"

# Try to load the model configuration file from cache
cached_file = try_to_load_from_cache(model_id, "config.json")

if cached_file:
    print(f"The model {model_id} is cached at: {cached_file}")
else:
    print(f"The model {model_id} is not cached on this machine.")