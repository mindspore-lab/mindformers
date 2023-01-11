## 微调引擎包服务化

# Dockerfile 安装的第三方包包括CANN、Modelarts、OBS、Flask等

- base image fetch

  ```shell
  https://support.huaweicloud.com/engineers-modelarts/modelarts_23_0217.html#modelarts_23_0217__section15496241203912
  
  docker pull swr.cn-north-4.myhuaweicloud.com/modelarts-job-dev-image/mindspore-ascend910-cp37-euleros2.8-aarch64-training:1.3.0-3.3.0-roma
  ```

  

- build images command

  ```shell
  docker build -t swr.cn-central-221.ovaijisuan.com/modelarts-image_zkyzdhs/cann5.euleros-flask-service:1.16 .
  ```

  

- docker push

  ```shell
  docker push swr.cn-central-221.ovaijisuan.com/modelarts-image_zkyzdhs/cann5.euleros-flask-service:1.16
  ```
