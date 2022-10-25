# 如何转换Huggingface的t5权重

## T5 Model

### 从HuggingFace的官方中搜索t5-small

下载模型权重。`t5-small`的层数为6层，然后执行下述命令

> python convert_t5_weight.py --layers 6 --torch_path pytorch_model.bin --mindspore_path ./converted_mindspore_t5.ckpt

### 加载T5模型，开始执行训练

在`examples/pretrain/pretrain_t5.sh`中，增加`--load_checkpoint_path`参数。
一个完整的示例如下所示。其中`--device_target="Ascend"`表示下述的命令将会在`Ascend`上面执行训练。

```bash
DEVICE_ID=$1
EPOCH_SIZE=$2
DATA_DIR=$3

python -m transformer.trainer.trainer \
    --auto_model='t5 \
    --epoch_size=$EPOCH_SIZE \
    --device_id=$DEVICE_ID \
    --train_data_path=$DATA_DIR \
    --optimizer="adam" \
    --max_seq_length=512 \
    --max_decode_length=512 \
    --parallel_mode="stand_alone" \
    --max_position_embeddings=16 \
    --d_kv=64 \
    --global_batch_size=96 \
    --vocab_size=32128 \
    --hidden_size=512 \
    --intermediate_size=2048 \
    --num_hidden_layers=6 \
    --num_attention_heads=8 \
    --load_checkpoint_path='mindspore_t5_small.ckpt'
    --bucket_boundaries=16 \
    --has_relative_bias=True \
    --device_target="Ascend"
```

## OPT Model

### OPT权重下载和OPT词表下载

从HuggingFace的[官网](https://huggingface.co/facebook/opt-2.7b) 下载`facebook/opt-2.7b`模型权重,记名字为`pytorch_model.bin`。`opt-2.7b`的层数为32层，设置为`--layers 32`，然后执行下述命令
将HuggingFace的权重转换为MindSpore的权重。

```bash
python tools/convert_opt_weight.py --layers 32 --torch_path pytorch_model.bin --mindspore_path ./converted_mindspore_opt.ckpt
```

从HuggingFace的[官网](https://huggingface.co/facebook/opt-2.7b) 下载`facebook/opt-2.7b`对应的词表文件，记为`vocab.json`

### 加载OPT模型，开始执行训练

在`examples/pretrain/pretrain_opt_distributed.sh`中，增加`--load_checkpoint_path`参数，指定转换后的权重的文件路径。
一个完整的示例如下所示。下述的命令将会启动OPT在8卡GPU上面进行训练

```bash
bash examples/pretrain/pretrain_opt_distributed.sh EPOCH_SIZE hostfile DATA_DIR
```

### 使用OPT进行推理

使用转换的权重或者训练完成的权重，用户可以使用下述的命令执行执行单卡2.6B模型OPT模型的推理。

在此脚本中 `--device_target="Ascend"`指定运行设备为`Ascend`，用户可以该值修改为`GPU`。

>注意：在此脚本中，已经默认设置load_checkpoint_path=converted_mindspore_opt.ckpt，vocab_path=vocab.json

如果用户需要自定义文件路径，请在`examples/pretrain/eval_opt.sh`进行修改。

```bash
bash examples/pretrain/eval_opt.sh "who are you?"
```

## ViT Model

### ViT权重下载

从HuggingFace的[官网](https://huggingface.co/google/vit-base-patch16-224/tree/main) 下载`pytorch_model.bin`模型权重。
该权重对应模型结构为`vit_base`，设置`--backbone_name vit_base`，然后执行下述命令
将HuggingFace的权重转换为MindSpore的权重。

```bash
python convert_vit_weight.py --backbone_name vit_base --torch_path pytorch_model.bin --mindspore_path converted_mindspore_vit.ckpt
```

# 执行翻译任务

## 数据集下载

下载WMT16翻译数据集，点击[此处](https://cdn-datasets.huggingface.co/translation/wmt_en_ro.tar.gz)下载，并且解压。

## 词表文件下载

词表文件可以从此处[下载](https://huggingface.co/t5-small/tree/main)。对应的文件名字为`spiece.model`。

## 转换成MindRecord格式

执行下述命令，可以将WMT16中的`train`数据集转换为mindrecord格式。如果用户需要转换`val`或者`test`，可以修改参数为`--split=val`或者
`--split=test`。

```bash
python tools/wmt16_to_mindrecord.py  \
       --split=train \
       --sp_model_path=/absolute path of spiece.model \
       --raw_dataset=/absolute path of wmt_en_ro \
       --output_file_path='wmt16'
```

### 使用OpenWebText数据集

- 使用Brown University处理后的版本，请点击此处[下载](https://skylion007.github.io/OpenWebTextCorpus/) 。

该数据集在下载过程中过滤了非英语网页，使用局部敏感哈希(LSH)来识别接近重复的文档并删除了相似阈值大于0.5的文档。

- 解压数据集

将数据集下载至preprocess_gpt目录后，进行下列操作解压。

```shell
cd preprocess_gpt
tar xvJf openwebtext.tar.xz
cd openwebtext
xz -dk *
```

在openwebtext数据集中摘取了少量数据集案例存放到了OpenWebText目录中。

- OpenWebText数据集预处理

```shell
cd preprocess_gpt
python tools/wiki_to_mindrecord.py \
--input_glob=./openwebtext/*
--dataset_type=openwebtext
--output_file=./output/openwebtext.mindrecord
```

参数说明：

- input_glob：原始的数据集文件夹目录，默认为 './OpenWebText_tiny/*'。
- dataset_type=：处理的数据集类型，默认为 openwebtext。
- output_file：输出文件的路径及名称，默认为 './output/openwebtext.mindrecord'。
- file_batch_size：文件的batch_size，默认为 1024。
- file_partition：输出文件是否进行分区存储，默认为 1。
- 运行成功后在output文件夹中会生成mindrecord文件：openwebtext.mindrecord，openwebtext.mindrecord.db。

## 2、使用GPT训练OpenWebText数据集

```bash
python -m transformer.trainer.trainer \
--auto_model='gpt' \
--epoch_size=$EPOCH_SIZE \
--train_data_path=.\preprocess_gpt\output\ \
--optimizer="adam"  \
--seq_length=1024 \
--parallel_mode="stand_alone" \
--global_batch_size=4 \
--vocab_size=50257 \
--hidden_size=2048 \
--num_layers=24 \
--num_heads=16 \
--device_target="GPU" > standalone_train_gpu_log.txt 2>&1 &
```

# BERT数据预处理

## 1、下载zhwiki或者enwiki数据集

- enwiki网址：https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2
- zhwiki网址：https://dumps.wikimedia.org/zhwiki/latest/zhwiki-latest-pages-articles.xml.bz2

## 2、使用WikiExactor提取和整理数据集中的文本

使用步骤：

```commandline
pip install wikiexactor
python -m wikiextractor.WikiExtractor ****.bz2 -o -b
```

参数说明：

- o：输出的文件夹，默认为text
- b：输出生成的每个文件的大小，例如20M

最后会在输出的文件夹下生成一系列的输出文件

## 3、生成tfrecord文件

运行文件为[create_pretraining_data.py](https://github.com/google-research/bert#pre-training-with-bert)，其代码思路为获取所有的输入文件，将输入文件的所有文档经过分词之后全部存入到all_documents列表中，然后通过all_documents列表生成mlm和nsp的实例，最后将所有的实例存入列表中，保存到tfrecord文件中。使用步骤:

```commandline
python3 create_pretraining_data.py --input-file=/PATH/sample.txt --output-file=/PATH/example.tfrecord --vocab-file=/PATH/vocab.txt
```

参数说明：

- input-file：输入文件，其中`PATH`为用户指定的路径
- output-file：输出文件，最后生成的tfrecord文件，其中`PATH`为用户指定的路径
- vocab-file：字典文件，其中`PATH`为用户指定的路径

tfrecord中存放的数据说明:

- input_ids: 经过mlm预训练任务处理之后的tokens对应于字典的id列表
- input_mask：表示哪些数据是有用的哪些数据是没有用的，1为有用，0为没用
- segment_ids：段id，表示token属于第几句话
- masked_lm_positions: mask的位置index
- masked_lm_ids: 将mask对应的标签数据转为label id
- masked_lm_weights: 哪些mask是有用的，1为有用，0为没用
- next_sentence_labels：nsp任务的标签

## 4、开始BERT预训练

将生成的tfrecord保存到工作路径中，然后将其指定到[BERT预训练脚本](https://gitee.com/mindspore/transformer/blob/master/examples/pretrain/pretrain_bert_distributed.sh)的对应变量`DATASET`中。

```bash
python -m transformer.trainer.trainer \
    --auto_model="bert" \
    --dataset_format="tfrecord" \
    --device_num=$RANK_SIZE \
    --data_path=$DATASET \
    --max_seq_length=512 \
    --global_batch_size=64 \
    --vocab_size=30522 \
    --parallel_mode="data_parallel" \
    --hidden_size=768 \
    --num_hidden_layers=24 \
    --num_attention_heads=16 \
    --data_parallel=8 \
    --model_parallel=1 \
    --device_target="GPU" > distribute_train_gpu_log.txt 2>&1 &
```
