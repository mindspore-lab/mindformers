# 启动包 (mxLaunchKit)
微调组件与外部平台任务运行时(Runtime)对接部分。预置在镜像中由引擎包直接或间接调用，依托外部平台完成微调基础工具包需求的代码、模型、输入及输出配置。
## 通用工具
| 工具名 | 备注 |
| ---- | ---- |
| execution.py | 通过Python的subprocess及multiprocessing库拉起子进程的函数封装 |
| log_utils.py | 依赖Python的logging库提供获取自定义名称日志记录器的函数封装 |
## 支持平台
| 平台 | 模型/启动脚本 | 备注                                    |
| ---- | ---- |---------------------------------------|
| HCSO | ma/launcher.py | 使用ModelArts服务，仅支持新版训练作业 |
### HCSO
ma文件夹下为适用于HCSO场景的微调组件ModelArts启动包，依赖ModelArts提供的内源库MoXing，当前只能在基于ModelArts提供基础镜像制作的镜像中运行。
#### 镜像制作
```
# MA提供基础镜像（Python 3.7, CANN 5.1.RC1, MindSpore 1.7）
# 构建镜像名称：fmtk-ma:py_3.9-ms_1.8.1_cann_5.1.RC2-euler_2.8.3-aarch64-d910-xxx（Python 3.9, CANN 5.1.RC2, MindSpore 1.8.1）  
cd script/docker/modelarts
sh build.sh # 需要提前准备好依赖脚本和安装包并放至指定目录，详情参考 script/docker/modelarts/README.md

```
#### 使用示例
```
# 镜像的默认工作路径：/home/ma-user
python mxLaunchKit/ma_user/launcher.py --task_type=finetune \
    --model_config_path=obs://xxx/xxx \
    --pretrained_model_path=obs://xxx/xxx/ \
    --code_path=/home/ma-user/modelarts/user-job-dir/xxx/ \
    --boot_file_path=/home/ma-user/modelarts/user-job-dir/xxx/xxx.py \
    --data_path=/cache/fmtk_data/ \
    --output_path=/cache/fmtk_output/ \
    --use_sfs=False \
    --ckpt_path=obs://xxx/xxx/ \
    --log_path=obs://xxx/xxx/
```
