# ZiYa

“姜子牙”系列大模型是由IDEA研究院推出的开源通用大模型，具备翻译，编程，文本分类，信息抽取，摘要，文案生成，常识问答和数学计算等能力。目前姜子牙通用大模型v1(Ziya-LLaMA-13B-v1)已完成大规模预训练、多任务有监督微调和人类反馈学习三阶段的训练过程。

姜子牙通用大模型v1.1(Ziya-LLaMA-13B-v1.1)对Ziya-LLaMA-13B-v1模型进行继续优化，通过调整微调数据的比例和采用更优的强化学习策略，本版本在问答准确性、数学能力以及安全性等方面得到了提升。

## Ziya-LLaMA-13B

Ziya-LLaMA-13B拥有130亿参数，模型结构采用LLaMA-13B，重新构建了中文词表，进行千亿token量级的已知的最大规模继续预训练，使模型具备原生中文能力。再经过500万条多任务样本的有监督微调(SFT)和综合人类反馈训练(RM+PPO+HFFT+COHFT+RBRS)，进一步激发和加强各种AI任务能力。

我们可以复用llama的代码，通过转换脚本将huggingface格式的子牙权重文件转换为mindspore格式的ckpt，再基于mindformer提供的高阶接口进行训练推理。

### 快速使用

#### Ziya-LLaMA-13B 预训练权重转换

请参考[Ziya-LLaMA-13B-v1](https://huggingface.co/IDEA-CCNL/Ziya-LLaMA-13B-v1#-%E4%BD%BF%E7%94%A8-usage-)使用Usage，按照步骤得到子牙原始权重。

- 其中step1获取huggingface权重可以下载[llama-13b-hf](https://huggingface.co/decapoda-research/llama-13b-hf/tree/main)，然后根据step2和子牙权重合并，得到完整的子牙13B权重。

执行权重转换脚本

```shell
python mindformers/models/llama/convert_weight.py --torch_ckpt_dir TORCH_CKPT_DIR --mindspore_ckpt_path MS_CKPT_NAME
```

```text
# 参数说明
TORCH_CKPT_DIR: huggingface权重保存目录路径
mindspore_ckpt_path: 权重保存文件名，保存为TORCH_CKPT_DIR/OUTPUT_NAME, 也可以指定为自定义保存路径
```

#### 推理

- pipeline接口推理

```python
import mindspore as ms

from mindformers.pipeline import pipeline
from mindformers.tools.register import MindFormerConfig
from mindformers.models import LlamaConfig, LlamaForCausalLM, LlamaTokenizer

ms.set_context(device_target="Ascend", device_id=6, mode=0)
config = MindFormerConfig('research/ziya/run_ziya_13b.yaml')

model_path = 'Your model path'
tokenizer_path = 'Your tokenizer path'

config.model.model_config.checkpoint_name_or_path = model_path
model_config = LlamaConfig(**config.model.model_config)
ziya_model = LlamaForCausalLM(model_config)
tokenizer = LlamaTokenizer(tokenizer_path, add_bos_token=True, add_eos_token=False)
tokenizer.add_tokens(["<human>", "<bot>"], special_tokens=True)

pipeline_task = pipeline("text_generation", model=ziya_model, tokenizer=tokenizer)

query = "帮我写一份去西安的旅游计划"
pipeline_result = pipeline_task(inputs, do_sample=False, max_length=512, add_special_tokens=True)
print(pipeline_result[0]['text_generation_text'])
```

**推理结果示例**

```text
'帮我写一份去西安的旅游计划 1、行程安排 2、交通方式 3、住宿安排 4、景点推荐 5、美食推荐 6、注意事项 7、其他建议 1、行程安排 第一天：到达西安，入住酒店，游览大雁塔、明城墙、回民街 第二天：参观兵马俑、华清池、大唐芙蓉园 第三天：游览西安城墙、钟鼓楼、陕西历史博物馆 第四天：参观西安碑林、陕西国际博览中心、大唐芙蓉园 第五天：游览华山、参观华山景区内的景点 第六天：游览华山、参观华山景区内的景点 第七天：游览华山、参观华山景区内的景点 第八天：离开西安 2、交通方式 建议乘坐高铁或飞机前往，可以选择在西安市区内乘坐地铁或出租车。 3、住宿安排 可以选择在市中心或景区附近的酒店住宿，方便游览景点。 4、景点推荐 大雁塔、明城墙、回民街、兵马俑、华清池、大唐芙蓉园、西安城墙、钟鼓楼、陕西历史博物馆、碑林、陕西国际博览中心、华山、华山景区内的景点。 5、美食推荐 可以品尝到肉夹馍、凉皮、羊肉泡馍、羊肉串、糖葫芦等特色美食。 6、注意事项 注意防晒、防蚊虫叮咬，注意保暖，避免着凉。 7、其他建议 可以购买当地特色纪念品，可以参加当地的文化活动，可以品尝当地美食。 以上是我的建议，希望能够帮助到您。'
```

#### 训练与微调

基于ziya-13b，目前提供了模型的基础配置文件`research/ziya/run_ziya_13b.yaml`。可参考llama的[预训练](https://gitee.com/mindspore/mindformers/blob/dev/docs/model_cards/llama.md#%E9%A2%84%E8%AE%AD%E7%BB%83)与[微调](https://gitee.com/mindspore/mindformers/blob/dev/docs/model_cards/llama.md#%E5%BE%AE%E8%B0%83)章节。

`注：使用ziya-13b进行训练或者微调时，需要使用ziya-13b配套的tokenizer.model处理数据集，以及选用ziya-13b的yaml配置文件进行任务启动。`