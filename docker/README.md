# dockerfile使用说明

在此给出官方标准镜像对应的dockerfile，如需对官方镜像做自定义修改，可以参考标准dockerfile进行改动

若无自定义镜像需求，建议直接使用[mindformers安装](../README.md#二mindformers安装)中提供的docker镜像链接

## 镜像内容

提供的标准镜像主要包含以下内容：

- Ubuntu: 20.04
- CANN: CANN 7.0.RC1.3.beta1
- MindSpore: 2.2.1
- MindSpore-Lite: 2.2.1
- MindFormers: 0.8.0

**注意**：镜像不包含Ascend硬件运行所需的固件和驱动，使用时需挂载宿主机的驱动；使用前请确认宿主机的固件驱动版本能够适配当前CANN版本，如不适配需要自行更新；昇腾固件驱动下载指引可参考[配套MindSpore 昇腾软件安装指引（23.0.RC3）](https://support.huawei.com/enterprise/zh/doc/EDOC1100336282)，社区版固件驱动可从这里获取：[固件与驱动](https://www.hiascend.com/hardware/firmware-drivers/community?product=4&model=26&cann=7.0.RC1.beta1&driver=1.0.RC3.alpha)

## 物理机dockerfile

提供了可用于Ascend服务器上的dockerfile，包括aarch架构和x86架构两个dockerfile

准备工作：

1. 准备可用的昇腾服务器环境，**确保可使用docker服务并且网络连接通畅**；
   > 注意：dockerfile构建镜像的过程中需下载较多网络资源，请确保网络连接通畅，否则镜像构建可能失败
2. 服务器上安装适配的固件驱动，如已有适配固件驱动，可跳过
3. 拉取ubuntu官方基础镜像

    ```shell
    docker pull ubuntu:20.04
    ```

镜像构建命令：

```bash
cd docker/ascend_xxx
docker build . -t mindformers:0.8.0
```

构建成功后可通过 `docker images` 命令查询构建好的镜像信息

容器启动命令：

```shell
# --device用于控制指定容器的运行NPU卡号和范围
# -v 用于映射容器外的目录
# --name 用于自定义容器名称

docker run -it -u root \
--ipc=host \
--network host \
--device=/dev/davinci0 \
--device=/dev/davinci1 \
--device=/dev/davinci2 \
--device=/dev/davinci3 \
--device=/dev/davinci4 \
--device=/dev/davinci5 \
--device=/dev/davinci6 \
--device=/dev/davinci7 \
--device=/dev/davinci_manager \
--device=/dev/devmm_svm \
--device=/dev/hisi_hdc \
-v /etc/localtime:/etc/localtime \
-v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
-v /var/log/npu/:/usr/slog \
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
--name {请手动输入容器名称} \
mindformers:0.8.0 \
/bin/bash
```

验证镜像可用性：

```bash
npu-smi info            # 检查npu卡挂载
pip list | grep mind    # 检查相关python依赖安装情况
python -c "import mindspore;mindspore.run_check()"  # 检查MindSpore安装正确性
python -c "import mindformers"      # 检查无导入依赖问题
```

## ModelArts dockerfile

modelarts上所使用的镜像与物理机有所不同，我们基于aicc给出的MindSpore镜像，安装MindFormers相关的环境依赖，以构建适用于modelarts的标准镜像

准备工作：

1. 准备可用的昇腾服务器环境，**确保可使用docker服务并且网络连接通畅**；
2. 拉取武汉aicc提供的MindSpore基础镜像

    ```shell
    swr.cn-central-221.ovaijisuan.com/wuh-aicc_dxy/mindspore2_2_0:MindSpore2.2.0-cann7.0rc1_py_3.9-euler_2.8.3-D910B
    ```

镜像构建命令：

```bash
cd docker/aicc
docker build . -t mindformers_aicc:0.8.0
```

镜像推送：

构建的modelarts镜像，需推送至aicc上进行使用；相关内容请参照aicc提供的[容器镜像服务 SWR](https://support.huaweicloud.com/swr/index.html)的使用说明以及[AICC上使用MindFormers教程](../docs/readthedocs/source_zh_cn/docs/practice/AICC.md)
