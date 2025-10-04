from Bio import Entrez
import configparser
import pandas as pd
import time

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
