# CodeGeex

CodeGeeX是一个具有130亿参数的多编程语言代码生成预训练模型。CodeGeeX采用华为MindSpore框架实现，在鹏城实验室“鹏城云脑II”中的192个节点（共1536个国产昇腾910 AI处理器）上训练而成。

## 快速使用

### CodeGeex-13B 预训练权重转换

通过[该链接](https://models.aminer.cn/codegeex/download/request)申请权重，您将收到一个包含临时下载链接文件```urls.txt```的邮件。推荐使用[aria2](https://aria2.github.io/)通过以下命令快速下载（请保证有足够的硬盘空间存放权重（～26GB））：

```bash
aria2c -x 16 -s 16 -j 4 --continue=true -i urls.txt
```

使用以下命令合并得到完整的权重：

```bash
cat codegeex_13b.tar.gz.* > codegeex_13b.tar.gz
tar xvf codegeex_13b.tar.gz
```

执行权重转换脚本

```shell
python research/codegeex/convert_weight.py --torch_path TORCH_CKPT_DIR --mindspore_path MS_CKPT_NAME
```

```text
# 参数说明
TORCH_CKPT_DIR: torch权重保存目录路径
mindspore_path: 权重保存文件名，保存为TORCH_CKPT_DIR/OUTPUT_NAME, 也可以指定为自定义保存路径
```

### 基于API接口推理

使用MindSpore API进行推理.

Atlas 800T A2需要配置环境变量

```shell
# node 1
export MS_ENABLE_GE=1
export MS_GE_TRAIN=1
export MS_ENABLE_REF_MODE=1
export MS_GE_ATOMIC_CLEAN_POLICY=1
```

```python
# >>> `chat.py`文件
import numpy as np
from typing import *
from mindspore.parallel import set_algo_parameters
from mindformers import PanguAlphaConfig, init_context
from code_tokenizer import CodeTokenizer
from codegeex import CodeGeexHeadModel


# set context
context_config = {"device_target": "Ascend", "mode": 0,  "max_device_memory": "31GB", "device_id": 2}
parallel_context_config = {"parallel_mode": 1, "gradients_mean": False, "full_batch": True}
rank_id, device_num = init_context(use_parallel=False, context_config=context_config, parallel_config=parallel_context_config)
set_algo_parameters(elementwise_op_strategy_follow=True, fully_use_devices=True)

config = PanguAlphaConfig(
    checkpoint_name_or_path=CKPT_PATH,
    batch_size = 1,
    seq_length = 2048,
    vocab_size = 52224,
    hidden_size = 5120,
    ffn_hidden_size = 20480,
    num_layers = 40,
    num_heads = 40,
    pad_token_id = 50256,
    eos_token_id = 50256,
    post_layernorm_residual = False,
    param_init_type = 'float16',
    compute_dtype = 'float16',
    softmax_compute_type = 'float32',
    dropout_rate = 0.1,
    hidden_act = 'fast_gelu',
    use_past = True,
    use_moe = False,
    expert_num = 1,
    per_token_num_experts_chosen = 1,
    repetition_penalty = 1,
    max_decode_length = 1024,
    top_k = 100,
    top_p = 0.95,
    temperature = 0.8,
    do_sample = True,
    eod_mask_loss = False,
    )

def chat():
    model = CodeGeexHeadModel(config)
    model.set_train(False)
    question_list = [
        "def add(a, b):\n    '''\n    Find the sum of a and b.\n    '''\n",
        "bool prime(int n) {\n    // Find whether n is a prime number\n",
        ]

    # Define tokenizer
    tokenizer = CodeTokenizer(config.vocab_size)
    i = 0
    for question in question_list:
        inputs = tokenizer.encode_code(question)
        inputs = np.array([inputs]).astype(np.int32) # add batch dim
        outputs = model.generate(inputs, max_length=1024, top_p=0.95, temperature=0.8, eos_token_id=50256)
        output_samples = tokenizer.decode_code(outputs)
        output_samples_str = "".join(output_samples)
        print(f"=================== prompt {i} ====================")
        print(question, flush=True)
        print(f"=================== generation {i} ====================")
        print(output_samples_str, flush=True)
        i = i + 1


if __name__ == "__main__":
    chat()


```

### 单机多卡运行训练

```shell
# node 1
export MS_ENABLE_GE=1
export MS_GE_TRAIN=1
export MS_ENABLE_REF_MODE=1
export MS_GE_ATOMIC_CLEAN_POLICY=1
cd mindformers/research
bash run_singlenode.sh "python codegeex/run_codegeex.py --config codegeex/run_codegeex_910b.yaml --run_mode=train --train_data path/to/mindrecord_dir" path/to/rank_table_file [0,8] 8
```

**参数说明**
  `config`: code_geex相关配置文件
  `run_mode`：运行模式，包括train，finetune，eval，predict
  `train_data`：train数据，训练时需要填入。

  更多输入可参考`run_codegeex.py
  `脚本内入参
