## wheel包测试使用脚本
- test_package_cmd.sh的命令
- python -m unittest test_engine_python.EngineTestCase
   （注意test_engine_python.py路径问题）

## 开发态测试脚本
- test_cmd.sh
- test_cmd_ez.sh

## 日志模块使用方式
- 微调组件的日志实例分为两个，分别是service_logger和logger，使用时只需要


  ```shell
from fm.src.aicc_tools.ailog.log import service_logger
from fm.src.aicc_tools.ailog.log import service_logger_without_std
from fm.src.aicc_tools.ailog.log import aicc_logger
  ```
在需要使用日志记录的地方使用对应的
  ```shell
logger.info()
logger.error()
logger.warn()
  ```
即可。

按照需要的业务场景使用service_logger或者logger实例。

service_logger用于记录微调组件 SDK或者CLI或者API 运行的日志信息；
service_logger_without_std记录的信息与service_logger相同，区别是service_logger_without_std的内容不会打印在屏幕上；
aicc_logger用于记录模型层面的日志信息。


## 开发态脚本调用
- 工作目录定位到项目层级 (mindxsdk-mxfoundationmodel)
- 命令行通过python命令执行功能参考 test_cmd.sh / test_cmd_ez.sh，示例
- 命令行通过fm命令执行功能参考 test_package_cmd.sh
- 命令行执行test目录下.py文件示例
```
    /xxxxx/mindxsdk-mxfoundationmodel# python -m fm.main 功能 --arg1 arg1_value --arg2 arg2_value
```