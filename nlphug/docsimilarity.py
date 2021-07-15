import pandas as pd
import numpy as np
import nltk
nltk.download('stopwords', download_dir='/workspace/conda/hugface/nltk_data')
from nltk.corpus import stopwords
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import torch
# import docx2txt

# # extract text
# text = docx2txt.process("file.docx")

# nltk.download('stopwords', download_dir='/workspace/data/nltk/')

documents = ['Machine learning is the study of computer algorithms that improve automatically through experience.\
Machine learning algorithms build a mathematical model based on sample data, known as training data.\
The discipline of machine learning employs various approaches to teach computers to accomplish tasks \
where no fully satisfactory algorithm is available.',
'Machine learning is closely related to computational statistics, which focuses on making predictions using computers.\
The study of mathematical optimization delivers methods, theory and application domains to the field of machine learning.',
'Machine learning involves computers discovering how they can perform tasks without being explicitly programmed to do so. \
It involves computers learning from data provided so that they carry out certain tasks.',
'Machine learning approaches are traditionally divided into three broad categories, depending on the nature of the "signal"\
or "feedback" available to the learning system: Supervised, Unsupervised and Reinforcement',
'Software engineering is the systematic application of engineering approaches to the development of software.\
Software engineering is a computing discipline.',
'A software engineer creates programs based on logic for the computer to execute. A software engineer has to be more concerned\
about the correctness of the program in all the cases. Meanwhile, a data scientist is comfortable with uncertainty and variability.\
Developing a machine learning application is more iterative and explorative process than software engineering.'
]
documents_df=pd.DataFrame(documents,columns=['documents'])

# removing special characters and stop words from the text
stop_words_l=stopwords.words('english')
documents_df['documents_cleaned']=documents_df.documents.apply(lambda x: " ".join(re.sub(r'[^a-zA-Z]',' ',w).lower() for w in x.split() if re.sub(r'[^a-zA-Z]',' ',w).lower() not in stop_words_l) )

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    # First element of model_output contains all token embeddings
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(
        -1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def most_similar(doc_id,similarity_matrix,matrix):
    print (f'Document: {documents_df.iloc[doc_id]["documents"]}')
    print ('\n')
    print (f'Similar Documents using {matrix}:')
    if matrix=='Cosine Similarity':
        similar_ix=np.argsort(similarity_matrix[doc_id])[::-1]
    elif matrix=='Euclidean Distance':
        similar_ix=np.argsort(similarity_matrix[doc_id])
    for ix in similar_ix:
        if ix==doc_id:
            continue
        print('\n')
        print (f'Document: {documents_df.iloc[ix]["documents"]}')
        print (f'{matrix} : {similarity_matrix[doc_id][ix]}')


# Sentences we want sentence embeddings for later replace this with text from job and resume
sentences = ['This is an example sentence', 'Each sentence is converted']

# sbert_model = SentenceTransformer('bert-base-nli-mean-tokens')
sbert_model = SentenceTransformer('paraphrase-mpnet-base-v2')

document_embeddings = sbert_model.encode(documents_df['documents_cleaned'])

pairwise_similarities = cosine_similarity(document_embeddings)
pairwise_differences = euclidean_distances(document_embeddings)

print(pairwise_similarities)
print("##"*50)
print(pairwise_differences)
print("##"*50)
most_similar(0,pairwise_differences,'Euclidean Distance')
print("##"*50)
most_similar(0,pairwise_similarities,'Cosine Similarity')
print("##"*50)
