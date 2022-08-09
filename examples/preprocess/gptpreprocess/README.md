# GPT 数据预处理

## 1、预训练数据处理

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
python pre_process.py \
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
python -m transformer.train \
--config='./transformer/configs/gpt/gpt_base.yaml' \
--epoch_size=$EPOCH_SIZE \
--data_url=.\preprocess_gpt\output\ \
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

## 社区贡献者

感谢社区贡献者 `@miaokongmiao` 在对本文档的贡献。
