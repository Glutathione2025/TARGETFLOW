import os
import requests
from huggingface_hub import try_to_load_from_cache
from Bio import Entrez
import configparser
import pandas as pd
import time
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import re
from collections import Counter
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification

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

def fetch_disease_abstracts(max_results=200, output_file='disease_abstracts.csv'):
    config = configparser.ConfigParser()
    config.read('config.ini')

    Entrez.email = config['pubmed']['email']
    
    disease_name = config['pubmed']['disease_name']
    query = f"({disease_name}) AND (target OR biomarker OR therapeutic)"
    print(f"Searching PubMed: {query}")

    handle = Entrez.esearch(db="pubmed", retmax=max_results, term=query)
    record = Entrez.read(handle)
    handle.close()
    id_list = record['IdList'] 
    if not id_list:
        print("No relevant literature found")
        return
  
    print(f"Found {len(id_list)} disease target-related literature")
    
    # Fetch abstract content
    abstracts = []
    for pmid in id_list:
        try:
            handle = Entrez.efetch(db="pubmed", id=pmid, retmode="xml")
            record = Entrez.read(handle)
            handle.close()
            
            if 'Abstract' in record['PubmedArticle'][0]['MedlineCitation']['Article']:
                abstract = " ".join(
                    str(text) for text in 
                    record['PubmedArticle'][0]['MedlineCitation']['Article']['Abstract']['AbstractText']
                )
                abstracts.append(abstract)
            time.sleep(0.34)  # Comply with PubMed API restrictions
        except Exception as e:
            print(f"Error processing PMID {pmid}: {str(e)}")
    
    # Save as CSV
    df = pd.DataFrame({'Abstract': abstracts})
    df.to_csv(output_file, index=False, encoding='utf-8-sig') 
    print(f"Saved {len(df)} literature abstracts to {output_file}")

if __name__ == "__main__":
    fetch_disease_abstracts(max_results=200)

def clean_text(text):
    # Text cleaning
    text = re.sub(r'-', '', text)
    text = re.sub(r'[^a-zA-Z0-9αβγΔ\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def is_biomedical_target(word):
    word_lower = word.lower()
    greek_chars = re.findall(r'[αβγδεζηθικλμνξοπρστυφχψω]', word_lower) 
    digits = re.findall(r'\d', word_lower)
    latin_chars = re.findall(r'[a-z]', word_lower)
    
    # Rule 1: Must contain at least one Greek letter or digit
    if not (greek_chars or digits):
        return False
    
    # Rule 2: Latin alphabet count must be ≥2 
    if len(latin_chars) < 2:
        return False
    
    # Rule 3: Greek letter count must be ≤1 
    if len(greek_chars) > 1:
        return False
    
    # All rules satisfied
    return True

def analyze_targets():
    try:
        df = pd.read_csv('disease_abstracts.csv', encoding='utf-8-sig')
    except:
        df = pd.read_csv('disease_abstracts.csv', encoding='latin1')
    
    text_col = 'Abstract' if 'Abstract' in df.columns else 'abstract'
    df['cleaned_text'] = df[text_col].apply(clean_text)
    
    try:
        model_path = "./bert"
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForTokenClassification.from_pretrained(model_path)
        tokenizer.model_max_length = 512
    except:
        tokenizer = AutoTokenizer.from_pretrained("pruas/BENT-PubMedBERT-NER-Gene")
        model = AutoModelForTokenClassification.from_pretrained("pruas/BENT-PubMedBERT-NER-Gene")
    
    nlp = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

    targets = []
    for abstract in df['cleaned_text'].head(200):
        entities = nlp(abstract)
        article_targets = set()  # Deduplicate each document
        for entity in entities:
            word = entity.get('word', '').strip().lower()
            if entity.get('entity_group', '') in ['B','I']:
                if not word.startswith('##'):
                    # Split compound entities by space, then apply rule-based filtering to each part
                    for part in word.split():
                        if is_biomedical_target(part):
                            article_targets.add(part)
        targets.extend(list(article_targets))
    
    if targets:
        top_targets = Counter(targets).most_common(10)  # Change the number as you want
        result_df = pd.DataFrame(top_targets, columns=['Target', 'Frequency'])
        print("High-potential target statistics:")
        print(result_df.to_string(index=False))
    else:
        print("No valid targets identified")

if __name__ == "__main__":
    analyze_targets()
