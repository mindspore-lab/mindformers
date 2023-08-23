### 边训练边评估

大模型的训练效果需要评测任务来作为衡量标准，而当前大模型的训练耗时长，等到训练整体结束后再进行评测任务的时间与算力成本过高

本功能提供了一套在训练过程中进行评估的流程方法，以动态观察模型的评估结果，具有以下特性：

1. 能够在训练过程中执行评估流程，并在日志中打印相关的评估结果信息；
2. 具有功能开关，根据开关状态决定是否启用边训练边评估功能；
3. 具备调用间隔控制项，根据该控制项决定调用边训练边评估功能的时间间隔；
4. 支持多种模型，只要模型能够调用Model.eval()完成评估，都可以复用该项功能；无法通过Model.eval()完成评估的模型则需要额外编码适配，暂不支持；

#### [边训练边评估支持度表](../model_support_list.md#边训练边评估支持度表)

> 说明：边训练边评估功能需模型已支持评估，并且该评估指标可以通过Model.eval()完成

#### 使用用例

- run_mindformer启用边训练边评估

描述：通过run_mindformers脚本参数，启用边训练边评估功能
测试方式：`--do_eval`开关启用边训练边评估功能，`--eval_dataset_dir`指定评估数据集

```shell
python run_mindformer.py \
--config configs/gpt2/run_gpt2.yaml \
--run_mode train \
--train_dataset_dir /your_path/wikitext-2-train-mindrecord \
--eval_dataset_dir /your_path/wikitext-2-eval-mindrecord \
--do_eval True
```

- trainer启用边训练边评估

描述：通过Trainer高阶接口入参，启用边训练边评估功能
测试方式：执行以下python脚本，其中数据集路径替换为实际路径

```python
def test_trainer_do_eval():
    from mindformers.trainer import Trainer
    # 初始化预训练任务
    trainer = Trainer(task='text_generation', model='gpt2',
                      train_dataset="/your_path/wikitext-2-train-mindrecord",
                      eval_dataset="/your_path/wikitext-2-eval-mindrecord")
    # 开启训练，并打开do_eval开关
    trainer.train(do_eval=True)

if __name__ == "__main__":
    test_trainer_do_eval()
```

- 配置评估间隔时间

描述：更改评估间隔时间，以控制执行评估的频率
测试方式：更改配置项，将 `configs/gpt2/run_gpt2.yaml` 文件中的 `eval_epoch_interval` 项修改为其他数值

执行`run_mindformer.py`启用边训练边评估中的启动脚本

```yaml
do_eval: False
eval_step_interval: -1    # num of step intervals between each eval, -1 means no step end eval.
# 修改此项eval_epoch_interval数值：
eval_epoch_interval: 50   # num of epoch intervals between each eval, 1 means eval on every epoch end.
```
