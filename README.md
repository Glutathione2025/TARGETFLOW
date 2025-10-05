# TARGETFLOW
## Overview
TARGETFLOW is a modular Python pipeline for automated target acquisition from biomedical literature. The workflow begins with automated literature retrieval to download relevant abstracts and construct a database. After selective text cleaning and data processing, it employs large language models (LLMs) for intelligent literature scanning, followed by code-based whitespace tokenization. This yields abundant biomedical entity samples. Finally, rule-based filtering is applied to output high-potential therapeutic targets for the specified disease.
## Innovation
By integrating whitespace tokenization with domain-knowledge-based rule filtering, the method significantly increases the proportion of usable entities. This approach is particularly effective for target discovery in diseases lacking target databases (e.g., rare and emerging diseases), as well as for target updates in well-established diseases.
# Quick Reproducibility
## System Requirements
Python 3.8+ (Author's environment: 3.9.5)<br>
Hugging Face account: Access Tokens-Create new token<br>
4GB RAM (8GB recommended)<br>
Internet connection (for model download and literature retrieval)
## Implementation Steps
1.Install dependencies: pip install -r requirements.txt<br>
2.Configure parameters in config.ini:<br>
[pubmed]<br>
email = your email@example.com<br>
disease_name = full name of your target disease<br>
3.Execute the full workflow: main.py
## Expected Output
200-abstract database: disease_abstracts.csv<br>
High-potential targets and their frequencies for the target disease: in DataFrame format<br>
# Architecture Design
## Workflow Overview
Model retrieval → Literature retrieval and database creation → Data preprocessing → Model selection and loading → Entity recognition and whitespace tokenization → Rule filtering → Frequency ranking and result output
## Components (Modular and Independently Executable)
1.Model Retrieval (Model retrieval.py)‌: Separately load the tokenizer (AutoTokenizer) and the model body (AutoModelForTokenClassification) from the remote repository (pruas/BENT-PubMedBERT-NER-Gene).<br>
2‌.Literature Retrieval (Literature retrieval.py)‌: Download the 200 most relevant article abstracts from PubMed and saves them as a local file (disease_abstracts.csv), serving as the TARGETFLOW temporary database.<br>
3‌.Entity Acquisition and Rule-Based Filtering (Entity acquisition and rule-based filtering.py)‌: Intelligently identify gene and protein entities in the 200 abstracts using the model, split entities at whitespace, apply rule-based filtering, and finally output high-potential targets ranked by frequency.
# License
MIT License
