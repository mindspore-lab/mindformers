'''for transfer codealpaca dataset to codegeex2 finetune use'''

import json
from sklearn import model_selection

with open('code_alpaca_20k.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

total = len(data)
sample_train, sample_test = model_selection.train_test_split(data, test_size=0.2)
with open('./train.json', 'a', encoding="utf-8") as f:
    total_written = 0
    for item in sample_train:
        text = ''
        if item["instruction"] != "":
            text += item["instruction"]
            if item["input"] != "":
                text += "\n\nASK:"
                text += item["input"]
        d = {}
        d['PROMPT'] = text
        d['ANSWER'] = item["output"]
        f.write(json.dumps(d) + '\n')
        total_written += 1
    print('train:', total_written)

with open('./dev.json', 'a', encoding="utf-8") as f:
    total_written = 0
    for item in sample_test:
        text = ''
        if item["instruction"] != "":
            text += item["instruction"]
            if item["input"] != "":
                text += "\n\nASK:"
                text += item["input"]
        d = {}
        d['PROMPT'] = text
        d['ANSWER'] = item["output"]
        f.write(json.dumps(d) + '\n')
        total_written += 1
    print('dev:', total_written)
