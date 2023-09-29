#
# Pyserini: Reproducible IR research with sparse and dense representations
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import json
import argparse
import pickle
import csv
from tqdm import tqdm
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert KILT 100 words passage tsv into a Document-level JSONL that can be processed by Pyserini')
    parser.add_argument('--input', required=True, help='Path to the kilt_w100_title.tsv file')
    parser.add_argument('--mapping', required=True, help='Path to the mapping_KILT_title.p file')
    parser.add_argument('--output-dir', required=True, help='Path to the output directory')
    parser.add_argument('--concat-title', action="store_true", default=False, help='Concatenate the title into each paragraph')

    args = parser.parse_args()

    # Map of title -> wikipedia id
    KILT_mapping = pickle.load(open(args.mapping, "rb"))

    not_found = set()
    with open(args.input, 'r') as f, open(os.path.join(args.output_dir, '100w_passage_kilt_knowledgesource_full.jsonl'), 'w') as outp:
        tsv = csv.reader(f, delimiter="\t")
        next(tsv)  # Get rid of headers
        
        texts = []
        titles = []
        old_title = None
        for row in tqdm(tsv, mininterval=10.0, maxinterval=20.0):
            i = row[0]
            text = row[1]
            title = row[2]
            
            if title not in KILT_mapping:
                not_found.add(f"{title}#{i}")
                continue
            
            if old_title is not None:
                if title == old_title:
                    texts.append(text)
                else:
                    wikipedia_id = str(KILT_mapping[old_title])
                    
                    doc = {}

                    full_text = ' '.join(texts)
                    
                    doc["id"] = f"{wikipedia_id}#{i}"
                    doc["wikipedia_title"] = old_title
                    doc["wikipedia_id"] = wikipedia_id
                    doc["contents"] = f"{old_title}\n{full_text}" if args.concat_title else full_text

                    _ = outp.write(json.dumps(doc))
                    _ = outp.write('\n')
                    
                    old_title = title
                    texts = [text]
                    titles.append(title)
                    
            else:
                texts.append(text)
                titles.append(title)
                old_title = title
            
        if len(texts) > 0:
            wikipedia_id = str(KILT_mapping[old_title])
                    
            doc = {}

            full_text = ' '.join(texts)
            doc["id"] = f"{wikipedia_id}#{i}"
            doc["wikipedia_title"] = old_title
            doc["wikipedia_id"] = wikipedia_id
            doc["contents"] = f"{old_title}\n{full_text}" if args.concat_title else full_text

            _ = outp.write(json.dumps(doc))
            _ = outp.write('\n')
            
    print(f"Not found: {not_found}")
    
    print(len(titles))
    
    print(len(list(set(titles))))

