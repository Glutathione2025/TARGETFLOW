import os
import requests
from transformers import AutoTokenizer, AutoModelForTokenClassification
from huggingface_hub import try_to_load_from_cache

try:
    requests.get('https://hf-mirror.com', timeout=5)
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
except:
    os.environ['HF_ENDPOINT'] = 'https://huggingface.co' 

# Secure token issuance
token = os.getenv('HF_TOKEN') or input("请输入HF token: ")

def download_model():
    try:
        # Cache lookup
        cache_status = try_to_load_from_cache(
            repo_id="pruas/BENT-PubMedBERT-NER-Gene",
            filename="pytorch_model.bin"
        )
        
        # Model retrieval
        tokenizer = AutoTokenizer.from_pretrained(
            "pruas/BENT-PubMedBERT-NER-Gene",
            token=token,
            resume_download=True,
            local_files_only=cache_status is not None
        )
        
        model = AutoModelForTokenClassification.from_pretrained(
            "pruas/BENT-PubMedBERT-NER-Gene",
            token=token,
            local_files_only=cache_status is not None
        )
        
        # Model saving
        save_path = "./bert"
        os.makedirs(save_path, exist_ok=True)
        tokenizer.save_pretrained(save_path)
        model.save_pretrained(save_path)
        print(f"The model has been saved to: {os.path.abspath(save_path)}")
        
    except Exception as e:
        print(f"Model Loading Error: {str(e)}")
        if "401" in str(e):
            print("Please verify the validity of the token")
        elif "404" in str(e):
            print("The model name may be incorrect")

if __name__ == "__main__":
    download_model()
