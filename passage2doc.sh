#!/bin/sh

python src/convert_kilt_100w_passage2full_tsv_to_jsonl.py \
	--input "kilt_w100_title.tsv" \
	--mapping "mapping_KILT_title.p" \
	--output-dir cache