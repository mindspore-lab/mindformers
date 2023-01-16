
# 微调组件用户手册（AICC场景）

## 1. 软件介绍
本系统是面向大模型应用场景的基础能力平台，旨在提升用户开发和部署大模型的效率，当前教程面向使用人工智能计算中心（AICC）算力场景，提供大模型微调（finetune）、评估（evaluate）、推理（infer）、部署（deploy）等功能，包括 **命令行** 和  **SDK接口** 两种使用方式。

![输入图片说明](resources/%E5%A4%A7%E6%A8%A1%E5%9E%8B%E5%BE%AE%E8%B0%83%E5%A5%97%E4%BB%B6%E4%BA%A7%E5%93%81.png)


## 2. 资源准备

- **OBS云存储平台账号**：下载安装 [OBS Browser+](https://support.huaweicloud.com/browsertg-obs/obs_03_1003.html) 或 [obsutil](https://support.huaweicloud.com/utiltg-obs/obs_11_0003.html)；

- **软硬件依赖**：Linux系统、Python、Docker（>=20.10.17版本）；

- **ModelArts**：modelarts-latest-*.whl  [下载对应版本](https://support.huaweicloud.com/sdkreference-modelarts/modelarts_04_0004.html)，模型发布与部署功能要求 ModelArts SDK的版本 ≥ 1.4.13；

- **微调组件引擎包**：Ascend_mindxsdk_mxFoundationModel-*.whl （在当前代码仓编译获取：bash build/build.sh ）；

- **计算中心账号等信息**： 联系AICC完成账号注册，获取相应SWR endpoint、Modelarts endpoint、OBS endpoint、NAS IP、Project ID、AK/SK等信息；
  


## 3 软件安装

```shell
  # 下载微调组件代码
  git clone https://gitee.com/HUAWEI-ASCEND/mindxsdk-mxfoundationmodel.git
  
  # 编译微调组件whl包，生成在dist目录下
  cd mindxsdk-mxfoundationmodel
  bash build/build.sh

  # 微调组件提供用户安装卸载脚本(`fm_user.sh`，存放路径`script/fm_user.sh`)，分别将该脚本与微调组件、Modelarts的whl包放置在同级目录下，使用如下命令进行安装
  bash fm_user.sh install fm #安装 Ascend_mindxsdk_mxFoundationModel
  bash fm_user.sh install ma #安装 modelarts
  
```

## 4 快速上手

### 4.1 镜像准备

参考 `script/docker/modelarts/README.md` 准备并上传镜像。

备注：当使用其他模型的时候，请同步使用其配套的DockerFile。


### 4.2 模型准备


按照如下文件结构准备训练资源

1. [下载opt模型代码 adapt_tk分支](https://gitee.com/mindspore/omni-perception-pretrainer/tree/adapt_tk/)：

    ```shell
    git clone https://gitee.com/mindspore/omni-perception-pretrainer.git -b adapt_tk
    ```

2. 依据如下目录树下载数据集：   

   [下载COCO caption数据集](https://pan.baidu.com/s/1ECN5JXlRPQsBS8O763Y8pA)（提取码84me），在`{opt模型根目录}/dataset`目录下解压；

   [下载COCO图片数据训练集](http://images.cocodataset.org/zips/train2014.zip)，将所有图片解压至`{opt模型根目录}/dataset/data/train/img/mscoco/train2014/`路径；

   [下载COCO图片数据测试集](http://images.cocodataset.org/zips/val2014.zip)，将所有图片解压至`{opt模型根目录}/dataset/data/train/img/mscoco/val2014/`路径；

   准备推理数据集，将任意张以`.jpg`或`.png`为后缀的图片文件，放置在`{opt模型根目录}/dataset/data_infer/`目录，还需将`{opt模型根目录}/dataset/data/ids_to_tokens_zh.json`文件拷贝至该目录。

3. [下载预训练模型文件](https://opt-release.obs.cn-central-221.ovaijisuan.com:443/model/OPT_1-38_136.ckpt)（`OPT_1-38_136.ckpt`）存放至`{opt模型根目录}/pretrained_model`路径。

4. 将云端训练涉及的**应用配置文件**`{opt模型根目录}/code/model_configs/app_config_*.yaml`中路径替换为实际obs路径与镜像路径。

5. （可选）将任务类型对应的模型配置文件`{opt模型根目录}/omni-perception-pretrainer/code/model_configs/model_config_*.yaml`，中的参数替换为实际用户所需参数，也可直接使用示例文件。

6. 准备完成后将`omni-perception-pretrainer`文件夹及其包含文件上传至obs。

    


### 4.3 功能体验

使用微调组件功能前需注册微调组件，运行如下命令，交互输入认证信息：

```shell
fm registry  # 依次输入registry type 1，以及计算中心账号对应的ak，sk，endpoint, 加密启用/关闭选项（T/t、F/f）
```

#### 4.3.1 模型微调

```shell
fm finetune --scenario modelarts --app_config obs://HwAiUser/omni-perception-pretrainer/code/model_configs/app_config_finetune.yaml --model_config_path obs://HwAiUser/code/model_configs/model_config_finetune.yaml
```

#### 4.3.2 模型评估

```shell
fm evaluate --scenario modelarts --app_config obs://HwAiUser/omni-perception-pretrainer/code/model_configs/app_config_evaluate.yaml --model_config_path obs://HwAiUser/code/model_configs/model_config_evaluate.yaml
```

#### 4.3.3 模型推理

```shell
fm infer --scenario modelarts --app_config obs://HwAiUser/omni-perception-pretrainer/code/model_configs/app_config_infer.yaml --model_config_path obs://HwAiUser/code/model_configs/model_config_infer.yaml
```

#### 4.3.4 查看状态

- 查看任务运行状态

```shell
fm job-status --scenario modelarts --app_config obs://HwAiUser/omni-perception-pretrainer/code/model_configs/app_config_finetune.yaml --job_id ***  # ***为job_id，任务拉起成功后生成
```


任务结束后，可在任务对应的`app_config_*.yaml`中指定的`output_path`下查看任务输出结果；在指定的`log_path`下查看任务输出日志， **微调组件配置文件配置项请参考附录A，B，更多功能接口参数详解请参考附录C** 。



## 5. 模型适配

本教程快速上手案例使用的模型已完成微调组件适配，如用户需要支持新的模型，适配方法可参考[微调组件用户手册（本地场景）](https://www.hiascend.com/document/detail/zh/mind-sdk/30rc3/mxtuningkit/tuningkitug/mxtuningug_0001.html)的模型适配章节。



## 6. 版权声明

华为技术有限公司版权所有，引用请注明出处



## 附录

### A 配置文件  

#### A1 app_config*.yaml配置项


- modelarts 场景

| 配置项                   | 类型     | 典型值                                  | 描述                    | 支持值                                                               |
|:----------------------|--------|--------------------------------------|-----------------------|-------------------------------------------------------------------|
| iam_endpoint          | str    | https://iam-pub.cn-xxx-221.yyy.com   | 统一身份认证服务终端节点          | 具体取值需要与所属地AI计算中心确认                                                |
| obs_endpoint          | str    | https://obs.cn-xxx-221.yyy.com       | 对象存储服务终端节点            | 具体取值需要与所属地AI计算中心确认                                                |
| modelarts_endpoint    | str    | https://modelarts.cn-xxx-221.yyy.com | modelarts平台服务终端节点     | 具体取值需要与所属地AI计算中心确认                                                |
| region_name           | str    | cn-xxx-221                           | 区域名称                  | 具体取值需要与所属地AI计算中心确认                                                |
| project_id            | str    | 52xxxxxxxxxcc                        | 项目ID                  | 获取方式：登录HSCO->账户->项目ID                                             |
| data_path             | str    | obs://xxx/datasets/cxxxr10/          | 模型的输入数据，支持OBS路径与SFS路径 | 使用obs账号，自行设置obs桶路径                                                |
| output_path           | str    | obs://xxx/xxx/train_output/          | 模型输出（如模型文件等）路径        | 使用obs账号，自行设置obs桶路径                                                |
| ckpt_path             | str    | obs://xxx/xxx/model/                 | 模型文件根目录               | **评估、推理任务**需指定该参数。使用obs账号，自行设置obs桶路径                              |
| code_url              | str    | obs://xxx/model/rexxx50/             | 模型脚本的根目录              | 使用obs账号，自行设置obs桶路径                                                |
| boot_file_path        | str    | train.py                             | 模型启动脚本（需在code_url目录下） | 使用obs账号，自行设置obs桶路径                                                |
| pretrained_model_path | str    | obs://xxx/model/                     | 预训练模型文件所在根目录          | **微调任务**使用预训练模型时需指定该参数，使用obs账号，自行设置obs桶路径                         |
| log_path              | str    | obs://xxx/xxx/log_path/              | 日志输出路径                | 使用obs账号，自行设置obs桶路径                                                |
| user_image_url        | str    | huawei/mav2-training-image:latest    | 自定义镜像名称               | 必须按照华为云swr自定义镜像规范命名                                               |
| pool_id               | str    | poolexxx3d                           | 专属资源池id, 共享资源池为None   | 专属资源池id, 登录HCSO->专属资源池                                            |
| node_num              | int    | 1,2,3,4,5,6,7,8....                  | 计算节点数量                | 大于0且小于资源池中服务器数量                                                   |
| device_num            | int    | 1,2,4,8                              | AI加速卡数量               | modelarts场景使用专属资源池可设置为1/2/4/8，共享资源池可设置为1/2/4/8，node_num不为1时必须设置为8 |
| nas_share_addr        | str    | xx.xx.xx.xx:/                        | 弹性文件服务地址              | 格式：IP:/                                                           |
| nas_mount_path        | str    | /mnt/sfs_turbo                       | 弹性文件服务挂载路径            | 一般路径                                                              |
| deployment            | object | 无                                    | 模型发布与部署相关参数           |                                                                   |
| deployment.pool_id    | str    | 2c9080xxxff017f1f7                   | 模型服务部署专属资源池id         |                                                                   |
| deployment.node_num   | int    | 1                                    | 模型服务的节点数              |                                                                   |
| deployment.device_num | int    | 1                                    | 模型服务单个节点的设备数          |                                                                   |



文件格式示例：

```yaml
scenario:
  modelarts:
    parameters: abc
```



#### A2 model_config*.yaml配置

该文件为模型配置文件，且为**可选配置**，即用户可根据训练需要选择是否配置该文件。配置方式简洁快速，仅需定义`params`与`freeze`两部分参数，具体配置规则可参考微调工具包（mxTuningKit）文档 模型适配章节。该文档包含在`Ascend_mindxsdk_mxTuningKit-*.whl`压缩包中，路径为`Ascend_mindxsdk_mxTuningKit-*.whl/data/README.md`。



### B 详细接口参考

微调组件的核心功能是基于fm命令（函数），目前支持 Modelarts，提供模型微调、评估和推理功能，支持命令行和SDK两种使用场景，本节将详细介绍两种场景对应的功能选项以及参数说明。



#### B1 命令行场景

微调组件命令行调用的一般格式为:	 `fm <功能> <参数> 参数值`

在命令行场景下，fm主要包含以下功能选项：

**任务管理**

在使用微调组件功能前，用户需首先运行`fm registry`命令注册微调组件，交互输入registry type（认证类型）以及对应认证信息。目前只支持 [1] ak/sk认证方式，用户需要依次填写从AI计算中心账号系统中获取的ak，sk密钥，以及endpoint（对象存储服务obs终端节点地址，同`app_config.yaml`配置文件中的 `obs_endpoint` 地址），来完成认证信息注册。

| 功能选项           | 功能说明            | 必选参数                     | 可选参数                                                                                   |
|----------------|:----------------|:-------------------------|:---------------------------------------------------------------------------------------|
| registry       | 注册微调组件          | 交互式注册，无需参数               | 交互式注册，无需参数                                                                             |
| config         | 配置使用场景、应用程序配置信息 | --scenario, --app_config | --help                                                                                 |
| show           | 展示当前账号历史任务状态    | --scenario, --app_config | --display, --job_id, --model_id, --service_id, --instance_type, --instance_num, --help |
| delete         | 删除指定任务          | --scenario, --app_config | --job_id, --model_id, --service_id, --help                                             |
| stop           | 停止指定任务          | --scenario, --app_config | --job_id, --help                                                                       |
| job-status     | 任务状态            | --scenario, --app_config | --job_id，--help                                                                        |
| model-status   | 任务状态            | --scenario, --app_config | --model_id，--help                                                                      |
| service-status | 任务状态            | --scenario, --app_config | --service_id，--help                                                                    |

注意：用户可使用fm config命令设置默认场景与默认应用程序配置，在使用其他管理功能及以下训练功能时可不用再次指定`--scenario`与`--app_config`参数。



**训练任务**

| 功能选项     | 功能说明 | 支持场景      | 必选参数                     | 可选参数                                                                                                                                                                      |
|----------|:-----|:----------|:-------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| finetune | 微调   | modelarts | --scenario, --app_config | --pretrained_model_path, --job_name, --model_name, --model_config_path,  --resume, --backend, --device_type, --device_num, --node_num, --output_path, --data_path, --help |
| evaluate | 评估   | modelarts | --scenario, --app_config | --ckpt_path, --job_name, --model_config_path, --resume, --backend, --device_type, --device_num, --node_num, --output_path, --data_path,  --help                           |
| infer    | 推理   | modelarts | --scenario, --app_config | --ckpt_path, --job_name, --model_config_path, --resume, --backend, --device_type, --device_num, --node_num, --output_path, --data_path,  --help                           |


**模型发布与部署**

| 功能选项    | 功能说明 | 支持场景      | 必选参数                                                    | 可选参数                                                                       |
|---------|:-----|:----------|:--------------------------------------------------------|:---------------------------------------------------------------------------|
| publish | 发布   | modelarts | --scenario, --app_config, --model_version, --model_path | --model_name, --backend, --device_type, --help                             |
| deploy  | 部署   | modelarts | --scenario, --app_config, --model_id                    | --service_name, --backend, --device_type, --device_num, --node_num, --help |


**对应参数说明**

| 参数                      | 缩写   | 类型   | 默认值       | 说明                                             |
|:------------------------|:-----|:-----|:----------|:-----------------------------------------------|
| --scenario              | -sn  | str  | None      | 使用场景，目前仅支持'modelarts'，对应新版训练作业                 |
| --app_config            | -a   | str  | None      | 带文件名的OBS路径，指向存放在OBS上的应用程序配置文件，具体参考配置文件说明       |
| --model_config_path     | -c   | str  | None      | 带文件名的OBS路径，指向存放在OBS上的模型配置信息，具体参考配置文件说明         |
| --data_path             | -dp  | str  | None      | OBS路径，指向模型输入数据路径                               |
| --output_path           | -op  | str  | None      | OBS路径，指定模型训练结果输出位置                             |
| --node_num              | -nn  | int  | 1         | 计算集群节点数量                                       |
| --device_num            | -dn  | int  | 8         | 每个节点NPU设备数量                                    |
| --device_type           | -d   | str  | npu       | 设备类型，只支持Ascend NPU                             |
| --backend               | -b   | str  | mindspore | 训练框架，只支持mindspore                              |
| --resume                | -r   | bool | FALSE     | 是否断点续训（目前暂不支持）                                 |
| --job_name              | -jn  | str  | 空字符串      | 训练作业名称，不指定会随机生成（格式：job_fm_yyyy-MM-dd-hh-mm-ss） |
| --pretrained_model_path | -pm  | str  | 空字符串      | 带文件名的OBS路径，指定OBS上存放的预训练模型文件                    |
| --ckpt_path             | -cp  | str  | 空字符串      | 带文件名的OBS路径，指向OBS上存放评估、推理模型文件的文件夹               |
| --job_id                | -j   | int  | None      | 训练作业id号                                        |
| --instance_num          | -im  | int  | 20        | 显示任务数量                                         |
| --display               | -dis | bool | TRUE      | 是否显示在命令行                                       |
| --model_name            | -m   | str  | 空字符串      | 模型名称，不指定会随机生成                                  |
| --model_version         | -mv  | str  | None      | 模型版本                                           |
| --model_path            | -mp  | str  | None      | 模型路径（文件夹）                                      |
| --model_id              | -mid | str  | None      | 模型id号                                          |
| --service_name          | -sed | str  | 空字符串      | 模型服务名称，不指定会随机生成                                |
| --service_id            | -sid | str  | None      | 模型服务id号                                        |
| --help                  |      |      |           | 展示命令对应帮助信息                                     |




注：在命令行场景下使用`fm --help`命令可查询微调组件功能接口名称，使用`fm [接口名] --help`命令可查询特定接口参数规格。



**示例：**

```shell
fm registry # 配置认证信息，交互输入registry type以及对应认证信息、endpoint地址、加密启用开关等
fm config --scenario modelarts --app_config obs://xxx/app_config.yaml # 配置默认场景、应用程序信息
fm finetune --model_config_path obs://xxx/model_config.yaml --job_name test_job # 指定一个名为test_job的微调任务
fm show # 展示当前账号历史任务状态
```



#### B2 SDK场景

##### B2.1 实例管理（实例类型：job、model以及service）

###### B2.1.1 注册组件

```Python
fm.registry(registry_info = None)
```

**功能说明**

​	配置认证信息。

**命令示例**

```python
import fm.fm_sdk as fm
# 默认启用加密组件加密认证信息
fm.registry(registry_info='1 ak sk endpoint')
# 或采用如下接口参数方式，选择性开启/关闭加密
fm.registry(registry_info='1 ak sk endpoint encryption_option')
```

**参数说明**

- 入参

| 参数名称          | 是否必选 | 参数说明                                                                                                                                                   |
|---------------|------|--------------------------------------------------------------------------------------------------------------------------------------------------------|
| registry_info | 是    | 认证信息，形式为'type xx yy zz T(t)/F(f)', 目前支持使用ak/sk认证方式，填写'1 ak sk endpoint encrypt_option'。 其中ak, sk从AI计算中心账号系统中获取;endpoint为对象存储服务（obs）终端节点，同app_config.yaml 配置文件中的 obs_endpoint 地址; encrypt_option是可选项，其意为是否启用加密开关，如未配置则默认开启加密，可选值为单字母，可配置'T/t'、 'F/f'。|
- 返回值：True/False



###### B2.1.2 进行默认项配置（可选）

```Python
fm.config(scenario=None, app_config=None)
```

**功能说明**

​	配置默认场景类型和模型基础信息

**前提约束**

- 第一次配置需要同时配置`scenario`与`app_config`参数，之后可以仅修改单个参数配置；
- 已将应用程序配置文件上传至obs。

**命令示例**

```python
import fm.fm_sdk as fm
fm.config(scenario='modelarts', app_config='obs://HwAiUser/app_config.yaml')
```

**参数说明**

- 入参

|    参数名称    |      是否必选      | 参数说明                                         |
|:----------:|:--------------:|----------------------------------------------|
|  scenario  | 第一次配置为必选，之后为可选 | 使用场景，目前仅支持modelarts                          |
| app_config | 第一次配置为必选，之后为可选 | 带文件名的OBS路径，指向存放在OBS上的应用程序配置文件，具体参考 附录A配置文件说明 |

- 返回值：True/False



###### B2.1.3 展示实例信息（默认实例类型job）

```Python
fm.show(scenario=None, app_config=None, job_id=None, model_id=None, servcei_id=None, instance_type='job', instance_num=20, display=True)
```

**功能说明**

​	查看当前账号历史实例状态，默认实例类型为job，可通过instance_type参数指定实例类型，但需注意id参数的实例类型优先级更高。

**命令示例**

```python
import fm.fm_sdk as fm
fm.show(scenario='modelarts', app_config='obs://HwAiUser/app_config.yaml', instance_num=10)
```

**参数说明**

- 入参

|     参数名称      |          是否必选           | 参数说明                                         |
|:-------------:|:-----------------------:|----------------------------------------------|
|   scenario    | 未配置组件（config）为必选，配置后为可选 | 使用场景，目前仅支持modelarts                          |
|  app_config   | 未配置组件（config）为必选，配置后为可选 | 带文件名的OBS路径，指向存放在OBS上的应用程序配置文件，具体参考 附录A配置文件说明 |
|    job_id     |            否            | 训练作业id号                                      |
|   model_id    |            否            | 模型id号                                        |
|  service_id   |            否            | 模型服务id号                                      |
| instance_type |            否            | 显示实例类型                                       |
| instance_num  |            否            | 显示实例数量                                       |
|    display    |            否            | 是否在命令行显示任务信息                                 |


- 返回值：具体任务信息/空字符串



###### B2.1.4 删除任务

```Python
fm.delete(scenario=None, app_config=None, instance_id=None)
```

**功能说明**

​	删除指定任务，通过id参数判断删除哪个类型的实例。

**命令示例**

```python
import fm.fm_sdk as fm
fm.delete(scenario='modelarts', app_config='obs://HwAiUser/app_config.yaml', job_id='abc')
```

**参数说明**

- 入参

|    参数名称    |          是否必选           | 参数说明                                         |
|:----------:|:-----------------------:|----------------------------------------------|
|  scenario  | 未配置组件（config）为必选，配置后为可选 | 使用场景，目前仅支持'modelarts'，对应新版训练作业               |
| app_config | 未配置组件（config）为必选，配置后为可选 | 带文件名的OBS路径，指向存放在OBS上的应用程序配置文件，具体参考 附录A配置文件说明 |
|   job_id   |            否            | 训练作业id号                                      |
|  model_id  |            否            | 模型id号                                        |
| service_id |            否            | 模型服务id号                                      |


- 返回值：True/False



###### B2.1.5 停止任务

```Python
fm.stop(scenario=None, app_config=None, job_id=None)
```

**功能说明**

​	停止指定实例，只支持训练作业（对应实例类型job）。

**命令示例**

```python
import fm.fm_sdk as fm
fm.stop(scenario='modelarts', app_config='obs://HwAiUser/app_config.yaml')
```

**参数说明**

- 入参

|    参数名称    |          是否必选           | 参数说明                                         |
|:----------:|:-----------------------:|----------------------------------------------|
|  scenario  | 未配置组件（config）为必选，配置后为可选 | 使用场景，目前仅支持modelarts                          |
| app_config | 未配置组件（config）为必选，配置后为可选 | 带文件名的OBS路径，指向存放在OBS上的应用程序配置文件，具体参考 附录A配置文件说明 |
|   job_id   |            否            | 训练作业id号                                      |


- 返回值：True/False



###### B2.1.6 查看训练作业状态

```Python
fm.job_status(scenario=None, app_config=None, job_id=None)
fm.model_status(scenario=None, app_config=None, model_id=None)
fm.service_status(scenario=None, app_config=None, service_id=None)
```

**功能说明**

​	查看指定实例状态。

**命令示例**

```python
import fm.fm_sdk as fm
fm.job_status(scenario='modelarts', app_config='obs://HwAiUser/app_config.yaml', job_id='abc')
fm.model_status(scenario='modelarts', app_config='obs://HwAiUser/app_config.yaml', model_id='abc')
fm.service_status(scenario='modelarts', app_config='obs://HwAiUser/app_config.yaml', service_id='abc')
```

**参数说明**

- 入参

|    参数名称    |          是否必选           | 参数说明                                         |
|:----------:|:-----------------------:|----------------------------------------------|
|  scenario  | 未配置组件（config）为必选，配置后为可选 | 使用场景，目前仅支持'modelarts'，对应新版训练作业               |
| app_config | 未配置组件（config）为必选，配置后为可选 | 带文件名的OBS路径，指向存放在OBS上的应用程序配置文件，具体参考 附录A配置文件说明 |
|   job_id   |            否            | 训练作业id号                                      |
|  model_id  |            否            | 模型id号                                        |
| service_id |            否            | 模型服务id号                                      |


- 返回值（训练作业）

      'Creating'：初始化
      'Pending'：任务等待中
      'Running'：任务运行中
      'Terminating'：任务停止中
      'Terminated'：任务已停止
      'Completed'：任务已完成
      'Failed'：任务失败
      'Abnormal'：任务异常

- 返回值（模型）

      'publishing'：发布中
      'published'：已发布
      'failed'：发布失败

- 返回值（模型服务）

      'running'：运行中
      'deploying'：部署中
      'concerning'：部分就绪
      'failed'：异常
      'stopped'：已终止


##### B2.2 拉起实例

###### B2.2.1 模型微调

```python
fm.finetune(scenario=None, app_config=None, pretrained_model_path='', job_name='', model_config_path=None, resume=FALSE, backend='mindspore', device_type='npu', device_num=8, node_num=1, output_path=None, data_path=None)
```

**功能说明**

​	创建并拉起微调任务。

**命令示例**

```python
import fm.fm_sdk as fm
fm.finetune(scenario='modelarts', app_config='obs://HwAiUser/app_config.yaml', job_name='fm666')
```

**参数说明**

- 入参

|         参数名称          |                 是否必选                 | 参数说明                                                     |
|:---------------------:| :--------------------------------------: | ------------------------------------------------------------ |
|       scenario        | 未配置组件（config）为必选，配置后为可选 | 使用场景，目前仅支持'modelarts'，对应新版训练作业            |
|      app_config       | 未配置组件（config）为必选，配置后为可选 | 带文件名的OBS路径，指向存放在OBS上的应用程序配置文件，具体参考 附录A配置文件说明 |
|       data_path       |                    是                    | OBS文件夹路径，指向模型输入数据路径                          |
|      output_path      |                    是                    | OBS文件夹路径，指定模型训练结果输出位置                      |
| pretrained_model_path |                    否                    | OBS文件夹路径，指向OBS上存放预训练模型文件的文件夹。         |
|   model_config_path   |                    否                    | 带文件名的OBS路径，指向存放在OBS上的模型配置信息，具体参考配置文件说明 |
|       job_name        |                    否                    | 任务名称，用户指定名称需保证该账号下无相同名称任务；如用户未指定该参数，任务名称会按照内置规则自动生成。 |
|        resume         |                    否                    | 是否断点续训                                                 |
|        backend        |                    否                    | 训练框架，只支持mindspore                                    |
|      device_type      |                    否                    | 设备类型，只支持Ascend NPU                                   |
|      device_num       |                    否                    | 每个节点NPU设备数量                                          |
|       node_num        |                    否                    | 计算集群节点数量                                             |



- 返回值：

|  返回值  |   说明   |
|:-----:|:------:|
| True  | 任务下发成功 |
| False | 任务下发失败 |



###### B2.2.2 模型评估

```python
fm.evaluate(scenario=None, app_config=None, pretrained_model_path='', job_name='', model_config_path=None, resume=FALSE, backend='mindspore', device_type='npu', device_num=8, node_num=1, output_path=None, data_path=None, ckpt_path=None)
```

**功能说明**

​	创建并拉起评估任务。

**命令示例**

```python
import fm.fm_sdk as fm
fm.evaluate(scenario='modelarts', app_config='obs://HwAiUser/app_config.yaml', job_name='fm666')
```

**参数说明**

- 入参

|       参数名称        |                 是否必选                 | 参数说明                                                     |
|:-----------------:| :--------------------------------------: | ------------------------------------------------------------ |
|     scenario      | 未配置组件（config）为必选，配置后为可选 | 使用场景，目前仅支持'modelarts'，对应新版训练作业            |
|    app_config     | 未配置组件（config）为必选，配置后为可选 | 带文件名的OBS路径，指向存放在OBS上的应用程序配置文件，具体参考 附录A配置文件说明 |
|     data_path     |                    是                    | OBS文件夹路径，指向模型输入数据路径                          |
|     ckpt_path     |                    是                    | OBS文件夹路径，指向OBS上存放评估模型文件的文件夹。           |
|    output_path    |                    是                    | OBS文件夹路径，指定模型训练结果输出位置                      |
| model_config_path |                    否                    | 带文件名的OBS路径，指向存放在OBS上的模型配置信息，具体参考配置文件说明 |
|     job_name      |                    否                    | 任务名称，用户指定名称需保证该账号下无相同名称任务；如用户未指定该参数，任务名称会按照内置规则自动生成。 |
|      resume       |                    否                    | 是否断点续训                                                 |
|      backend      |                    否                    | 训练框架，只支持mindspore                                    |
|    device_type    |                    否                    | 设备类型，只支持Ascend NPU                                   |
|    device_num     |                    否                    | 每个节点NPU设备数量                                          |
|     node_num      |                    否                    | 计算集群节点数量                                             |

- 返回值：

|  返回值  |   说明   |
|:-----:|:------:|
| True  | 任务下发成功 |
| False | 任务下发失败 |



###### B2.2.3 模型推理

```python
fm.infer(scenario=None, app_config=None, pretrained_model_path='', job_name='', model_config_path=None, resume=FALSE, backend='mindspore', device_type='npu', device_num=8, node_num=1, output_path=None, data_path=None, ckpt_path=None)
```

**功能说明**

​	创建并拉推理任务。

**命令示例**

```python
import fm.fm_sdk as fm
fm.infer(scenario='modelarts', app_config='obs://HwAiUser/app_config.yaml', job_name='fm666')
```

**参数说明**

- 入参

|       参数名称        |                 是否必选                 | 参数说明                                                 |
|:-----------------:| :--------------------------------------: |------------------------------------------------------|
|     scenario      | 未配置组件（config）为必选，配置后为可选 | 使用场景，目前仅支持'modelarts'，对应新版训练作业                       |
|    app_config     | 未配置组件（config）为必选，配置后为可选 | 带文件名的OBS路径，指向存放在OBS上的应用程序配置文件，具体参考 附录A配置文件说明         |
|     data_path     |                    是                    | OBS文件夹路径，指向模型输入数据路径                                  |
|     ckpt_path     |                    是                    | OBS文件夹路径，指向OBS上存放推理模型文件的文件夹。                         |
|    output_path    |                    是                    | OBS文件夹路径，指定模型训练结果输出位置                                |
| model_config_path |                    否                    | 带文件名的OBS路径，指向存放在OBS上的模型配置信息，具体参考配置文件说明               |
|     job_name      |                    否                    | 任务名称，用户指定名称需保证该账号下无相同名称任务；如用户未指定该参数，任务名称会按照内置规则自动生成。 |
|      resume       |                    否                    | 是否断点续训                                               |
|      backend      |                    否                    | 训练框架，只支持mindspore                                    |
|    device_type    |                    否                    | 设备类型，只支持Ascend NPU                                   |
|    device_num     |                    否                    | 每个节点NPU设备数量                                          |
|     node_num      |                    否                    | 计算集群节点数量                                             |

- 返回值：

|  返回值  |   说明   |
|:-----:|:------:|
| True  | 任务下发成功 |
| False | 任务下发失败 |


###### B2.2.4 模型发布

```python
fm.publish(scenario=None, app_config=None, model_version=None, model_path=None, model_name=None, backend='mindspore', device_type='npu')
```

**功能说明**

​	发布模型。

**命令示例**

```python
import fm.fm_sdk as fm
fm.publish(scenario='modelarts', app_config='obs://HwAiUser/app_config.yaml', model_version='1.1.1', model_path='obs://HwAiUser/test_model/')
```

**参数说明**

- 入参

|     参数名称      |          是否必选           | 参数说明                                         |
|:-------------:|:-----------------------:|----------------------------------------------|
|   scenario    | 未配置组件（config）为必选，配置后为可选 | 使用场景，目前仅支持'modelarts'，对应新版训练作业               |
|  app_config   | 未配置组件（config）为必选，配置后为可选 | 带文件名的OBS路径，指向存放在OBS上的应用程序配置文件，具体参考 附录A配置文件说明 |
|  model_name   |            否            | 模型名称，可重名；如用户未指定该参数，名称会按照内置规则自动生成。            |
| model_version |            是            | 模型版本，用户指定版本时需保证该账号下同名模型无相同版本号。               |
|  model_path   |            是            | 模型存放路径（文件夹）                                  |
|    backend    |            否            | 训练框架，只支持mindspore                            |
|  device_type  |            否            | 设备类型，只支持Ascend NPU                           |

- 返回值：

|   返回值    |      说明      |
|:--------:|:------------:|
| model_id | 模型发布成功返回模型ID |


###### B2.2.5 模型部署

```python
fm.deploy(scenario=None, app_config=None, model_id=None, service_name=None, backend='mindspore', device_type='npu', device_num=1, node_num=1)
```

**功能说明**

​	部署模型为服务。

**命令示例**

```python
import fm.fm_sdk as fm
fm.deploy(scenario='modelarts', app_config='obs://HwAiUser/app_config.yaml', model_id='abc', device_num=1, node_num=1)
```

**参数说明**

- 入参

|     参数名称     |          是否必选           | 参数说明                                         |
|:------------:|:-----------------------:|----------------------------------------------|
|   scenario   | 未配置组件（config）为必选，配置后为可选 | 使用场景，目前仅支持'modelarts'，对应新版训练作业               |
|  app_config  | 未配置组件（config）为必选，配置后为可选 | 带文件名的OBS路径，指向存放在OBS上的应用程序配置文件，具体参考 附录A配置文件说明 |
| service_name |            否            | 服务名称，不可重名；如用户未指定该参数，名称会按照内置规则自动生成。           |
|   model_id   |            是            | 模型发布后的ID                                     |
|   backend    |            否            | 训练框架，只支持mindspore                            |
| device_type  |            否            | 设备类型，只支持Ascend NPU                           |
|   node_num   |            否            | 模型服务使用节点数量                                   |
|  device_num  |            否            | 每个节点NPU设备数量                                  |

- 返回值：

|    返回值     |       说明       |
|:----------:|:--------------:|
| service_id | 模型服务部署成功返回服务ID |



#### B3 参数优先级说明

可以发现微调组件在应用程序配置文件（app config）中配置的参数项与命令行、SDK接口中配置的参数存在重复项，这种设计目的是方便用户将可复用参数一次性配置在配置文件中，其余参数可以通过命令行来修改。

调用微调组件服务时，需要给定场景（scenario）以及应用程序配置信息（app_config）。为方便使用，用户可以先用`fm config`命令配置好默认的场景与应用程序信息，配置成功后本地缓存。之后用户使用其他功能时，如果上述信息没有变化，无需再次指定这些参数信息。

除fm config之外，fm的其他功能中，所有参数优先取命令行的配置值。如命令行未指定上述参数，会从fm config指令在本地生成的缓存信息中获取对应值。命令行设置的参数值，app_config会覆盖fm config在本地缓存的信息，但是其他参数如model_config_path，data_path等不会影响本地缓存。



在这种优先级设定下，微调组件命令调用支持两种模式：

**简易模式**：使用config配置好--scenario，--app_config信息，之后使用其他功能（show/delete/stop/finetune/job-status）无需再次指定

**全参模式**：不使用config配置信息，使用其他功能时必须指定scenario，app_config

示例：

```shell
# 简易模式
fm registry # 配置认证信息，在交互命令中输入
fm config --scenario modelarts --app_config obs://xxx/app_config.yaml 
fm finetune --model_config_path obs://xxx/model_config.yaml --job_name test1 
fm show
```



```shell
# 全参模式
fm registry # 配置认证信息，在交互命令中输入
fm finetune --scenario modelarts --app_config obs://xxx/app_config.yaml  --model_config_path obs://xxx/model_config.yaml --job_name test1
fm show --scenario modelarts --app_config obs://xxx/app_config.yaml
```

两种模式代码执行返回结果一致。

### C FAQ
| id  | 问题描述                                                                           | 出现原因                                                     | 解决方式                                                     |
|:----|--------------------------------------------------------------------------------|----------------------------------------------------------|----------------------------------------------------------|
| 1   | 使用registry命令注册时长时间无结果响应                                                        | 可能是用户网络问题，或者终端节点（endpoint）地址填写错误                         | 检查网络设置，检查注册信息中endpoint是否填写正确                             |
| 2   | 命令执行后一直阻塞（entropy may not enough）                                              | 当前机器熵池资源不足                                               | 安装并部署启动haveged服务或使用熵池资源充足的机器                             |
| 3   | service - ERROR - detect concurrency attack risk, current request is rejected! | 同一台服务器并发执行fm命令                                           | 微调组件侧现不支持并发请求，如需并发调用fm命令，请调用时自行实现任务队列以保证同一时刻只有一个fm命令在运行。 |
| 4   | 提示找不到镜像，或镜像地址非法                                                                | 1.镜像未上传至SWR；2.配置文件中指定镜像与SWR不一致；3.未给ModelArts配置包含SWR权限的委托 | 1.在SWR服务确认镜像已上传；2.确认app_config中镜像地址配置正确（不需要region前缀）；3.联系管理员配置正确的委托   |
