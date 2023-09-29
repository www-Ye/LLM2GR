import pickle
from tqdm import tqdm
import json
from trie import MarisaTrie
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
)

tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-13b-hf")

with open('cache/w1002full_wikititle2id.json', 'r', encoding='utf-8') as f:
	title2id = json.load(f)

titles = []

for t in tqdm(title2id.keys()):
    tokens = tokenizer(t, add_special_tokens=False)["input_ids"] + [2]
    titles.append(tokens)

print(t)
print(titles[-1])
print(len(titles))
new_trie = MarisaTrie(titles)

with open("cache/llama_kilt_full_titles_trie.pkl", "wb") as f:
    pickle.dump(new_trie, f)