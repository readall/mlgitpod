from transformers import ReformerConfig, PyTorchBenchmark, PyTorchBenchmarkArguments
from transformers import ReformerModelWithLMHead, ReformerTokenizer
import torch

import pandas as pd
import numpy as np
import nltk
nltk.download('stopwords', download_dir='/workspace/conda/hugface/nltk_data')
nltk.download('punkt', download_dir='/workspace/conda/hugface/nltk_data')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
from sentence_transformers import SentenceTransformer

from transformers import AutoTokenizer, AutoModel
import torch
import io

from sentence_transformers import SentenceTransformer, util


JD_FILE = "/workspace/mlgitpod/nlphug/jd.txt"
RESUME_FILE = "/workspace/mlgitpod/nlphug/resume.txt"
unwanted_chars = ['\n', '\n\n', '\n\n\n', '\t','\t\t', '\t\t\t']

def load_file(name=JD_FILE):
    content_list = []

    with io.open(name, 'rt', newline='\r\n') as f:
        content_list = f.readlines()
    return content_list


def profile_matching_v1():
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    input_jd = load_file(JD_FILE)
    input_resume = load_file(RESUME_FILE)
    joined_list = []
    joined_list.append(input_jd)
    joined_list.append(input_resume)
    final_list = []
    
    for line in joined_list:
        # print(line)
        # print("#"*50)
        text_tokens = word_tokenize(line[0])
        # print("*"*50)
        # print(text_tokens)
        # print("*"*50)
        tokens_without_ = [word for word in text_tokens if not word in stopwords.words() ]
        tokens_without_sw = [word for word in tokens_without_ if not word in unwanted_chars ]
        sentence_wo_sw = ' '.join(tokens_without_sw)
        # print("#"*50)
        # print(sentence_wo_sw)
        # print("#"*50)
        final_list.append(sentence_wo_sw)
    
    paraphrases = util.paraphrase_mining(model,final_list )

    for paraphrase in paraphrases[0:10]:
        score, i, j = paraphrase
        print("{} \t\t {} \t\t Score: {:.4f}".format(final_list[i][:10], final_list[j][:10], score))

    # #keyword extraction
    # print("#"*50)
    # jd_sentence_wo_sw = ""
    # for line in input_jd:
    #     jd_text_tokens = word_tokenize(line[0])
    #     jd_tokens_without_ = [word for word in text_tokens if not word in stopwords.words() ]
    #     jd_tokens_without_sw = [word for word in tokens_without_ if not word in unwanted_chars ]
    #     jd_sentence_wo_sw = ' '.join(tokens_without_sw)

    # model = SentenceTransformer('sentence-transformers/distilbert-base-nli-mean-tokens')
    # embeddings = model.encode(jd_sentence_wo_sw)
    # print(embeddings, embeddings.size)  
    # print("#"*50, "End Embeddings")
    print("#$"*30)
    extract_keywords(input_jd)
    print("#$"*30)
    extract_keywords(input_resume)
    print("#$"*30)


def extract_keywords(input_list):
    from sklearn.feature_extraction.text import CountVectorizer
    text_all = ""
    sentence_wo_sw = ""
    for line in input_list:
        tokens_without_ = [word for word in line[0] if not word in stopwords.words() ]
        tokens_without_sw = [word for word in tokens_without_ if not word in unwanted_chars ]
        sentence_wo_sw = ' '.join(tokens_without_sw)

    n_gram_range = (1, 1)
    stop_words = "english"

    # Extract candidate words/phrases
    count = CountVectorizer(ngram_range=n_gram_range, stop_words=stop_words).fit([sentence_wo_sw])
    candidates = count.get_feature_names()
    print(candidates)


profile_matching_v1()

