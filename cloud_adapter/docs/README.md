# 微调引擎包配置文件使用
通过 fm config 命令进行配置文件注册，后续直接执行应用该配置执行模型任务
```
fm config --scenario modelarts --app_config obs://xxx/app_config.yaml
fm finetune
```
通过 --app_config 参数动态指定配置文件，执行模型任务
```
# modelarts 场景下微调功能对应的新版 model_config 文件请参考微调工具包（mxTuningKit）文档
fm finetune --scenario modelarts --app_config obs://xxx/app_config.yaml --model_config_path obs://xxx/model_config.yaml
```
