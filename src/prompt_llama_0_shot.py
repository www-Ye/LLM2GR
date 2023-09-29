

TASKS = ['eli5', 'fever', 'hotpotqa', 'nq', 'triviaqa', 'wow']

# recite

fever_recite_prompt1 = '''Claim: {}

The Wikipedia paragraph to support or refute the above claim is:

Answer:'''

fever_recite_prompt2 = '''background: {}
claim: {}
Q: Is the claim true or false?
A:'''

triviaqa_recite_prompt1 = '''Question: {}

The Wikipedia paragraph to answer the above question is:

Answer:'''

triviaqa_recite_prompt2 = '''Refer to the passage below and answer the following question with just a few words.
Passage: {}
Q: {}
A: The answer is'''

nq_recite_prompt1 = '''Question: {}

The Wikipedia paragraph to answer the above question is:

Answer:'''

nq_recite_prompt2 = '''Refer to the passage below and answer the following question with just a few words.
Passage: {}
Q: {}
A: The answer is'''

hotpotqa_recite_prompt1 = '''Question: {}

The Wikipedia paragraph to answer the above question is:

Answer:'''

hotpotqa_recite_prompt2 = '''Refer to the passage below and answer the following question with just a few words.
Passage: {}
Q: {}
A: The answer is'''

eli5_recite_prompt1 = '''Question: {}

The Wikipedia paragraph to answer the above question is:

Answer:'''

eli5_recite_prompt2 = '''Refer to the passage below and answer the following question in detail.
Passage: {}
Q: {}
A:'''

wow_recite_prompt1 = '''Conversation: {}

The Wikipedia paragraph to answer the above conversation is:

Answer:'''

wow_recite_prompt2 = '''background: {}
{}
'''

recite_prompt = {'triviaqa': [triviaqa_recite_prompt1, triviaqa_recite_prompt2], \
	'hotpotqa': [hotpotqa_recite_prompt1, hotpotqa_recite_prompt2], \
	'nq': [nq_recite_prompt1, nq_recite_prompt2], \
	'fever': [fever_recite_prompt1, fever_recite_prompt2], \
	'eli5': [eli5_recite_prompt1, eli5_recite_prompt2], \
	'wow': [wow_recite_prompt1, wow_recite_prompt2]}