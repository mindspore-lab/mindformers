# ModelArts 适用镜像
包含微调启动包（mxLaunchKit）及微调工具包（mxTuningKit）的镜像。以构建包含以下工具版本的镜像为例
* Python 3.9
* CANN 5.1.RC2
* MindSpore 1.8.1

构建后镜像名称为 fmtk-ma:py_3.9-ms_1.8.1_cann_5.1.RC2-euler_2.8.3-aarch64-d910-{timestamp}
## 镜像制作与推送
准备一台可访问外网及AICC的构建机器，并获取基础镜像：swr.cn-central-221.ovaijisuan.com/mindxsdk/fmtk-ma_base:2022110901

驱动、软件要求：`Python 3.9`，`CANN 5.1.RC2`，`MindSpore 1.8.1`。

* Step 0 配置镜像构建路径

```shell
# 注意将以下命令中的镜像构建目录替换为实际镜像构建目录如：omni-perception-pretrainer/code/docker
export DOCKER_BUILD_PATH="mindxsdk-mxfoundationmodel/script/docker/modelarts"
```

* Step 1 [下载昇腾AI处理器 ARM 平台开发套件软件包](https://www.hiascend.com/software/cann/community-history)，拷贝至镜像构建目录下pkg文件夹：

```shell
cp Ascend-cann-toolkit_5.1.RC2_linux-aarch64.run ${DOCKER_BUILD_PATH}/pkg/
```

* Step 2 将获取的微调工具包（mxTuningKit）拷贝至镜像构建目录下pkg文件夹：

```shell
# 注意将以下命令中的工具包名称替换为实际获取到的名称
cp Ascend_mindxsdk_mxTuningKit-xxx.whl ${DOCKER_BUILD_PATH}/pkg/
```

* Step 3 执行镜像构建：

```shell
#     
bash ${DOCKER_BUILD_PATH}/build.sh
```

* Step 4 参考 **附录A** 部分，按照生成的登录指令登陆至 SWR ，将新构建镜像上传：

```shell
# 提前设置需要推送的SWR组织名称与执行build.sh得到镜像时间戳后缀（假设组织名称为mindxsdk，timestamp为20221028181325）
export org_name=mindxsdk
export timestamp=20221028181325
# 执行镜像名称前缀修改并推送至指定的SWR组织
docker tag fmtk-ma:py_3.9-ms_1.8.1_cann_5.1.RC2-euler_2.8.3-aarch64-d910-${timestamp} swr.cn-central-221.ovaijisuan.com/${org_name}/fmtk-ma:py_3.9-ms_1.8.1_cann_5.1.RC2-euler_2.8.3-aarch64-d910-${timestamp}
docker push swr.cn-central-221.ovaijisuan.com/${org_name}/fmtk-ma:py_3.9-ms_1.8.1_cann_5.1.RC2-euler_2.8.3-aarch64-d910-${timestamp}
```

## 镜像使用
* 在微调引擎包要求的应用配置文件（app_config）中指定 user_image_url 字段的值为镜像名称及标签即可
```
# 只适用场景：modelarts
user_image_url: {镜像组织名称}/fmtk-ma:py_3.9-ms_1.8.1_cann_5.1.RC2-euler_2.8.3-aarch64-d910-xxx
```
* 本地基于新制作镜像拉起容器示例，具体使用涉及hccl，参考官方提供 [READEME](https://gitee.com/mindspore/models/blob/master/utils/hccl_tools/README.md)
```
# 以下示例挂载1号卡
docker run -it \
    --device=/dev/davinci1 \
    --device=/dev/davinci_manager \
    --device=/dev/devmm_svm \
    --device=/dev/hisi_hdc \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /usr/local/Ascend/add-ons/:/usr/local/Ascend/add-ons/ \
    -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi \
    {镜像组织名称}/fmtk-ma:py_3.9-ms_1.8.1_cann_5.1.RC2-euler_2.8.3-aarch64-d910-xxx bash
```
* 不建议在模型代码中使用相对路径，若需要获取代码所在路径可以通过启动包设置环境变量 **MODEL_TASK_CWD** 获取
* 当前镜像集成了 ModelArts 提供的五个 Ascend910 配套脚本用于云端拉起训练作业 [参考链接](https://support.huaweicloud.com/docker-modelarts/develop-modelarts-0106.html)
  * 微调组件对其中部分逻辑进行修改，详见 **附录B** 中说明。脚本在镜像内的路径为 /home/ma-user/ascend910/

## 附录A：AICC云端环境上传镜像说明
将镜像打包上传到容器镜像服务（SWR）
  1.  修改上传服务器的配置
      +   按照下述内容修改 `/etc/docker/daemon.json` 文件（如没有此文件需手动创建）：
          
            ```json
            {
                "insecure-registries": [
                    "swr.cn-central-221.ovaijisuan.com"
                ]
            }
            ```
          
      +   重启Docker让配置生效
            ```shell
            systemctl daemon-reload # 重启守护进程
            systemctl restart docker
            ```
          
      +   为了能访问到计算中心的云端资源，需在 `/etc/hosts` 文件中增加下述内容，注意这里`*.*.*.*`需要换成对应的IP地址

            ```
            *.*.*.* swr.cn-central-221.ovaijisuan.com
            *.*.*.* modelarts.cn-central-221.ovaijisuan.com
            ```
  2.  登录镜像仓库
      
      在AI计算中心界面按照下述路径获取登录指令并在宿主机上执行：
      
      - 云资源 -> ModelArts -> 镜像服务控制台，等待跳转后进入 HCSO 容器镜像服务界面。
      
      + 组织管理 -> 创建组织 -> 输入组织名称（可自行指定）
      
      + 我的镜像 -> 客户端上传 -> 生成临时登录指令 -> 复制到宿主机（服务器）上执行即可登录至SWR

  3.  上传镜像
        ```shell
        sudo docker tag  test_local:0.0.1 swr.cn-central-221.ovaijisuan.com/组织名称/test_local:0.0.1
        sudo docker push swr.cn-central-221.ovaijisuan.com/组织名称/test_local:0.0.1
        ```
- 上传成功后可以在 我的镜像 - 自有镜像 下可以查看已上传的镜像。

- 其中`swr.cn-central-221.ovaijisuan.com/组织名称/test_local:0.0.1`为对应镜像地址，用于微调组件应用程序配置。

## 附录B：ModelArts 提供 Ascend910 配套脚本调整说明
  1. 打开 CANN 层日志落盘
```python
# 拷贝以下代码至fmk.py中的FMK类内部的gen_env_for_fmk函数中（return语句前），以配置 CANN 日志输出路径
log_dir = FMK.get_log_dir()
process_log_path = os.path.join(log_dir, self.job_id, 'ascend', 'process_log', 'rank_' + self.rank_id)
FMK.set_env_if_not_exist(current_envs, 'ASCEND_PROCESS_LOG_PATH', process_log_path)
os.makedirs(current_envs['ASCEND_PROCESS_LOG_PATH'], exist_ok=True)
```
  2. 调整每张卡上进程的工作路径为用户代码根目录
```
# Ascend910 配套脚本对每个卡上进程的工作目录进行了单独限制。
def get_working_dir(self):
        fmk_workspace_prefix = ModelArts.get_parent_working_dir()
        return os.path.join(os.path.normpath(fmk_workspace_prefix), 'device%s' % self.device_id)
# 由于模型脚本中可能存在的基于代码根目录的相对路径形式I/O，修改fmk.py文件中第80行代码涉及的工作目录，将每张卡上进程工作目录切换成代码根目录。
with self.switch_directory(os.getenv("MODEL_TASK_CWD", os.getcwd())):
```
用户可以对镜像进行增量构建来修改 Ascend910 配套脚本（/home/ma-user/ascend910/*.py）
