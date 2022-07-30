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
    -o：输出的文件夹，默认为text
    -b：输出生成的每个文件的大小，例如20M
最后会在输出的文件夹下生成一系列的输出文件

## 3、生成tfrecord文件
- 运行文件为create_pretraining_data.py，其代码思路为获取所有的输入文件，将输入文件的所有文档经过分词之后全部存入到all_documents列表中，然后通过all_documents列表生成mlm和nsp的实例，最后将所有的实例存入列表中，保存到tfrecord文件中。
- 使用步骤:
```commandline
python3 create_pretraining_data.py --input-file=/PATH/sample.txt --output-file=/PATH/example.tfrecord --vocab-file=/PATH/vocab.txt
```

参数说明：
- input-file：输入文件，其中`PATH`为用户指定的路径
- output-file：输出文件，最后生成的tfrecord文件，其中`PATH`为用户指定的路径
- vocab-file：字典文件，其中`PATH`为用户指定的路径

tfrecord中存放的数据说明：
- input_ids: 经过mlm预训练任务处理之后的tokens对应于字典的id列表
- input_mask：表示哪些数据是有用的哪些数据是没有用的，1为有用，0为没用
- segment_ids：段id，表示token属于第几句话
- masked_lm_positions: mask的位置index
- masked_lm_ids: 将mask对应的标签数据转为label id
- masked_lm_weights: 哪些mask是有用的，1为有用，0为没用
- next_sentence_labels：nsp任务的标签

## 4、开始BERT预训练

将生成的tfrecord保存到工作路径中，然后将其指定到[BERT预训练脚本](https://gitee.com/mindspore/transformer/blob/master/examples/pretrain/pretrain_bert_distributed.sh)的对应变量`DATASET`中。
```
python ./transformer/train.py \
    --config='./transformer/configs/bert/bert_base.yaml' \
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

## 社区贡献者
感谢社区贡献者 `@wangjincheng2` 在对本文档的贡献。