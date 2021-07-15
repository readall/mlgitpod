from transformers import ReformerConfig, PyTorchBenchmark, PyTorchBenchmarkArguments
from transformers import ReformerModelWithLMHead, ReformerTokenizer
import torch

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


from transformers import AutoTokenizer, AutoModelWithLMHead

JD_FILE = "/workspace/mlgitpod/nlphug/jd.txt"
RESUME_FILE = "/workspace/mlgitpod/nlphug/resume.txt"

def load_file(name=JD_FILE):
    my_file = open(name)
    content_list = my_file.readlines()
    return content_list


# config = ReformerConfig.from_pretrained("google/reformer-enwik8", 
#                                     lsh_attn_chunk_length=16386,
#                                     local_attn_chunk_length=16386,
#                                     lsh_num_chunks_before=0,
#                                     local_num_chunks_before=0
#                                     )

# config = ReformerConfig.from_pretrained("google/reformer-enwik8")

# benchmark_args = PyTorchBenchmarkArguments(sequence_lengths=[2048, 4096, 8192, 16386],
#                                 batch_sizes=[1],
#                                 models=["Reformer"]
#                                 )
# benchmark = PyTorchBenchmark(configs=[config], args=benchmark_args)
# result = benchmark.run()
# print(result)

# Encoding
def encode(list_of_strings, pad_token_id=0):
    max_length = max([len(string) for string in list_of_strings])

    # create emtpy tensors
    attention_masks = torch.zeros((len(list_of_strings), max_length), dtype=torch.long)
    input_ids = torch.full((len(list_of_strings), max_length), pad_token_id, dtype=torch.long)

    for idx, string in enumerate(list_of_strings):
        # make sure string is in byte format
        if not isinstance(string, bytes):
            string = str.encode(string)

        input_ids[idx, :len(string)] = torch.tensor([x + 2 for x in string])
        attention_masks[idx, :len(string)] = 1

    return input_ids, attention_masks

# Decoding
def decode(outputs_ids):
    decoded_outputs = []
    for output_ids in outputs_ids.tolist():
        # transform id back to char IDs < 2 are simply transformed to ""
        decoded_outputs.append("".join([chr(x - 2) if x > 1 else "" for x in output_ids]))
    return decoded_outputs

stop_words_l=stopwords.words('english')

model = ReformerModelWithLMHead.from_pretrained("google/reformer-enwik8")
input_jd = load_file(JD_FILE)
document_jd = pd.DataFrame(input_jd,columns=['documents'])
document_jd['documents_cleaned'] = document_jd.documents.apply(lambda x: " ".join(re.sub(r'[^a-zA-Z]',' ',w).lower() for w in x.split() if re.sub(r'[^a-zA-Z]',' ',w).lower() not in stop_words_l) )
encoded_jd, attention_masks_jd = encode(document_jd['documents_cleaned'])
print("JD encoded dimension: ", encoded_jd.size())

input_resume = load_file(RESUME_FILE)
document_resume = pd.DataFrame(input_resume,columns=['documents'])
document_resume['documents_cleaned'] = document_resume.documents.apply(lambda x: " ".join(re.sub(r'[^a-zA-Z]',' ',w).lower() for w in x.split() if re.sub(r'[^a-zA-Z]',' ',w).lower() not in stop_words_l) )

encoded_resume, attention_masks_jd = encode(document_resume['documents_cleaned'])
print("Resume encoded dimension: ", encoded_resume.size())



# print(encoded, attention_masks)
# decoded_output = decode(model.generate(encoded, do_sample=True, max_length=512))
# print(decoded_output)

