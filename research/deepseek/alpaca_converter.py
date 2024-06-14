"""
fastchat stanford code_alpaca data convert tools.
"""

import argparse

import json

import pathlib

# Prompt from stanford code_alpaca's training script

PROMPT_DICT = {
    "prompt": (
        'You are an AI programming assistant, utilizing the DeepSeek Coder model, developed by DeepSeek Company, '
        'and you only answer questions related to computer science. '
        'For politically sensitive questions, security and privacy issues, and other non-computer science questions,'
        ' you will refuse to answer.\n'
        '### Instruction:\n'
        '{instruction}\n'
        '### Response:\n'
    )
}


def main(args_param):
    data_path = pathlib.Path(args_param.data_path)
    with data_path.open(encoding='utf-8') as f:
        data = json.load(f)
    prompt = PROMPT_DICT["prompt"]

    sources = [
        prompt.format_map(example)
        for example in data
    ]

    targets = [example["output"] for example in data]

    new_data = []

    cnt = 1
    l = 0
    for s, t in zip(sources, targets):
        l = max(l, len(s))
        new_data.append(
            {
                "id": str(cnt),
                "conversations": [
                    {
                        "from": "human",
                        "value": s,
                    },
                    {
                        "from": "gpt",
                        "value": t,
                    },
                ],
            }
        )

        cnt += 1
    print('1')
    json.dump(new_data, open(args_param.output_path, "w", encoding='utf-8'), ensure_ascii=False, indent=2)
    print('1', l)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="alpaca-data.json")
    parser.add_argument(
        "--output_path", type=str, default="alpaca-data-conversation.json"
    )
    args = parser.parse_args()
    main(args)
