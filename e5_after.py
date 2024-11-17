import datasets
from datasets import load_dataset
import os
from eval.retrieval.bm25 import BM25
from utils import utils
from eval.retrieval.kv_store import KVStore
from typing import List
import argparse
from tqdm import tqdm


query_data = load_dataset("princeton-nlp/LitSearch", "query", split="full")
corpus_clean_data = load_dataset("princeton-nlp/LitSearch", "corpus_clean", split="full")
corpus_s2orc_data = load_dataset("princeton-nlp/LitSearch", "corpus_s2orc", split="full")

args = argparse.Namespace(
    index_type="e5",  # Simulate the --index_type argument
    key="title_abstract",  # Simulate the --key argument
    dataset_path="princeton-nlp/LitSearch",  # Default value (or you can customize)
    index_root_dir="retrieval_indices"  # Default value (or you can customize)
)


corpus_data = corpus_clean_data

def get_index_name(args: argparse.Namespace) -> str:
    return os.path.basename(args.dataset_path) + "." + args.key

index_name = get_index_name(args)


kv_pairs = {sentence: utils.get_clean_corpusid(record) for record in corpus_clean_data for sentence in utils.get_sentence_from_t_a(record)}

from eval.retrieval.e5 import E5

index = E5(index_name)

index.create_index(kv_pairs)

index_name = get_index_name(args)
index.save(args.index_root_dir)


def load_index(index_path: str) -> KVStore:
    index_type = os.path.basename(index_path).split(".")[-1]
    if index_type == "bm25":
        from eval.retrieval.bm25 import BM25
        index = BM25(None).load(index_path)
    elif index_type == "instructor":
        from eval.retrieval.instructor import Instructor
        index = Instructor(None, None, None).load(index_path)
    elif index_type == "e5":
        from eval.retrieval.e5 import E5
        index = E5(None).load(index_path)
    elif index_type == "gtr":
        from eval.retrieval.gtr import GTR
        index = GTR(None).load(index_path)
    elif index_type == "grit":
        from eval.retrieval.grit import GRIT
        index = GRIT(None, None).load(index_path)
    else:
        raise ValueError("Invalid index type")
    return index


args = argparse.Namespace(
    index_name="LitSearch.title_abstract.e5", 
    top_k=40,  
    retrieval_results_root_dir="results/retrieval", 
    index_root_dir="retrieval_indices",  
    dataset_path="princeton-nlp/LitSearch" 
)




index = load_index(os.path.join(args.index_root_dir, args.index_name))
query_set = [query for query in datasets.load_dataset(args.dataset_path, "query", split="full")]
for query in tqdm(query_set):
    query_text = query["query"]
    top_k = index.query(query_text, args.top_k)
    unique = []
    seen = []
    for key, value in top_k:
      if value not in seen:  
        unique.append((key, value))  
        seen.append(value)  
    
    query["retrieved"] = unique[:20]
'''
for i in results:
        if i not in seen:
            unique.append(i)
            seen.append(i)
    query["retrieved"] = unique[ :20] 

    '''
os.makedirs(args.retrieval_results_root_dir, exist_ok=True)
output_path = os.path.join(args.retrieval_results_root_dir, f"{args.index_name}.jsonl")
utils.write_json(query_set, output_path)