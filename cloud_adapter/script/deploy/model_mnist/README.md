# LeNet-Mnist 使用微调组件在云端快速发布并部署为在线服务

## 前期准备
* 确认当前环境安装了最新版本的 modelarts sdk（>=1.4.13）与微调组件引擎包
* 确认已参照`script/docker/modelarts/README.md`完成自定义镜像构建
* 调整`app_config_deploy.yaml`，将其中参数根据实际情况修改为合法值（注意`deployment`与`swr_endpoint`字段），上传至OBS
* 从 [ModelZoo-Lenet](https://www.hiascend.com/zh/software/modelzoo/models/detail/C/ed0f341d37a9ce8cf46d3b0ce734f8d0) 获取模型并重命名为`lenet.ckpt`
* 将得到的模型文件拷贝至`model_mnist`路径下，并将包含配置文件和模型的文件夹`model_mnist`上传至OBS

最终在OBS上model_mnist文件夹的文件树如下：
```shell
    model_caption \
       ----service.py
       ----lenet.py
       ----lenet.ckpt
       ----start.sh
       ----requirements.txt
```

## 发布与部署
假设文件夹与配置文件上传至OBS后路径为`obs://HwAiUser/model_mnist/`与`obs://HwAiUser/app_config_deploy.yaml`，执行以下命令（注意提前执行fm registry完成微调组件注册）
```shell
# modelarts sdk提供的模型服务相关接口为同步接口，发布与部署的命令会较为耗时
export obs_prefix="obs://HwAiUser"
fm config --scenario modelarts --app_config ${obs_prefix}/app_config_deploy.yaml
fm publish --model_version 0.0.1 --model_path ${obs_prefix}/model_mnist/
# 假设 publish 命令执行后得到的 model_id 为：07903661-9b1d-4b82-8074-d014e913690e
export published_model_id="07903661-9b1d-4b82-8074-d014e913690e"
fm deploy --model_id ${published_model_id}
```
## 注意事项
* `--model_path`指定的OBS模型文件夹被加载到容器中的路径为：/home/mind/model/
* 镜像的入口为：/home/mind/model/start.sh
* 部署完成后会获得API调用地址，拼接上在`service.py`脚本中预定义的URL`/infer/image`，即可进行调用
* 在modelarts的前端页面可参照以下配置完成预测

![infer_image](../doc/predict_in_ma_console.png)

