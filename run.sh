#!/bin/sh

python src/llm2gr.py --dataset hotpotqa \
	--inference_mode recite --model_path meta-llama/Llama-2-13b-hf \
	--isngrams 1 --isretrieval 1 --title_gen_beams 15 --re_beams 10 \
	--lam 0.9 --title_return_nums 2 --gen_len 16