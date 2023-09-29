#!/bin/sh

python src/title2fmid.py

python src/build_trie.py

python src/build_fm_index.py cache/100w_passage_kilt_knowledgesource_full.jsonl cache/kilt_w1002full_corpus.fm_index \
	--hf_model meta-llama/Llama-2-13b-hf \
	--format kilt