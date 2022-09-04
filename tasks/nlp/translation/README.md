# 执行翻译任务

## 数据集下载

下载WMT16翻译数据集，点击[此处](https://cdn-datasets.huggingface.co/translation/wmt_en_ro.tar.gz)下载，并且解压。

## 词表文件下载

词表文件可以从此处[下载](https://huggingface.co/t5-small/tree/main)。对应的文件名字为`spiece.model`。

## 转换成MindRecord格式

执行下述命令，可以将WMT16中的`train`数据集转换为mindrecord格式。如果用户需要转换`val`或者`test`，可以修改参数为`--split=val`或者
`--split=test`。

```bash
python tasks/nlp/translation/wmt16_process.py  \
       --split=train   \ 
       --sp_model_path=/absolut path of spiece.model \
       --raw_dataset=/absolut path of wmt_en_ro \
       --output_file_path='wmt16'
```



