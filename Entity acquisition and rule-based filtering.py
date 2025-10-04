import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import pandas as pd
import re
from collections import Counter
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification

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
