import json
import tqdm

input_path = 'cache/100w_passage_kilt_knowledgesource_full.jsonl'

wikititle2id = {}
title2wikiid = {}
with open(input_path, "r", 2**16) as f:
    cnt = 0
 
    pieces_it = [json.loads(line) for line in f]

    pieces_it = tqdm.tqdm(pieces_it)

    for row in pieces_it:
        
        idx = row["wikipedia_id"]
        title = row["wikipedia_title"]
        text = row["contents"]
        
        idx = idx.strip()
        title = title.strip()
        
        if title not in title2wikiid:
            title2wikiid[title] = idx
  
        if title not in wikititle2id:
            wikititle2id[title] =  [cnt]
        else:
            wikititle2id[title].append(cnt)
            
        cnt += 1

print(cnt)
print(len(title2wikiid))

with open('cache/title2wikiid.json', 'w', encoding='utf-8') as f:
    json.dump(title2wikiid, f)
    
with open('cache/w1002full_wikititle2id.json', 'w', encoding='utf-8') as f:
    json.dump(wikititle2id, f)