# 环境变量使用说明

Mindformers提供了以下环境变量的配置说明，请根据使用场景自行配置使用。常用的环境变量设置写入到`~/.bash_profile` 或 `~/.bashrc`中，可让环境变量设置在新启动的终端窗口中立即生效。

## 调试调优

以下配置适用于网络模型调试调优过程中的内存分析、DUMP功能、日志打印、通信等待等方面。

|            环境变量             | 功能                  |   类型    | 取值                                                                                                                   | 说明                                                      |
|:---------------------------:|:--------------------|:-------:|:---------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------|
|     LOG_MF_PATH     | Mindformers日志保存位置         | String | 路径，支持相对路径与绝对路径                                                                                            | 设置后，会将Mindformers的日志文件保存到该路径，建议使用绝对路径。                                |
|     MS_MEMORY_STATISTIC     | 内存析构                | Integer | 1：开启<br>   0：关闭<br> 默认值：0                                                                                            | 若开启内存析构，会在OOM时打印内存池占用情况。                                |
|    MINDSPORE_DUMP_CONFIG    | 指定Dump功能所依赖的配置文件的路径 | String  | 文件路径，支持相对路径与绝对路径                                                                                                     |                                                         |
|           GLOG_v            | 控制Mindspore日志的级别    | Integer | 0-DEBUG<br> 1-INFO<br> 2-WARNING<br> 3-ERROR，表示程序执行出现报错，输出错误日志，程序可能不会终止<br> 4-CRITICAL，表示程序执行出现异常，将会终止执行程序<br> 默认值：2 | 指定日志级别后，将会输出大于或等于该级别的日志信息。                              |
|   ASCEND_GLOBAL_LOG_LEVEL   | 控制CANN的日志级别         |    Integer     | 0-DEBUG<br> 1-INFO<br> 2-WARNING<br> 3-ERROR  <br> 默认值：3                                                             |                                                         |
| ASCEND_SLOG_PRINT_TO_STDOUT | 设置plog日志是否打屏        |    Integer     | 1：开启<br>   0：关闭<br> 默认值：0                                                                                            |                                                         |
| ASCEND_GLOBAL_EVENT_ENABLE  | 设置事件级别              |    Integer     | 1：开启Event日志<br>   0：关闭Event日志<br> 默认值：0                                                                              |                                                         |
|      HCCL_EXEC_TIMEOUT      | HCCL进程执行同步等待时间      |    Integer     | 执行同步等待时间（s）<br> 默认值：1800s                                                                                            | 不同设备进程在分布式训练过程中存在卡间执行任务不一致的场景，通过该环境变量可控制设备间执行时的同步等待的时间。 |
|    HCCL_CONNECT_TIMEOUT     | HCCL建链超时等待时间        |    Integer     | 建链等待时间（s）<br> 默认值：120s                                                                                               | 用于限制不同设备之间socket建链过程的超时等待时间。                            |

## 910相关配置

以下配置仅在910服务器上适用。

|            环境变量             | 功能                    |   类型    | 取值                                                                                | 说明                                              |
|:---------------------------:|:----------------------|:-------:|:----------------------------------------------------------------------------------|:------------------------------------------------|
|          MS_GE_TRAIN          | 训练/推理场景选择             | Integer | 1：训练场景   <br>   0：推理场景，host侧内存使用会大于训练场景。   <br> 默认值：1                             | MS_GE_TRAIN=1和=0分别用于训练和推理场景，GE编译流程不同。           |
|         MS_ENABLE_GE          | 使能GE后端                | Integer  | 1：开启<br>   0：关闭<br> 默认值：1                                                         |                                                 |
|      MS_ENABLE_REF_MODE       | REF_MODE编译优化          | Integer | 1：开启<br>   0：关闭<br> 默认值：1                                                         | CANN-7.0以上版本支持此模式，优化内存管理方式，建议开启。                |
|     MS_ENABLE_FORMAT_MODE     | 整网ND格式                |    Integer     | 1：开启<br>   0：关闭<br> 默认值：0                                                         | 将整网算子转换为ND格式计算，建议开启。                            |
|   MS_GE_ATOMIC_CLEAN_POLICY   | 清理网络中atomic算子占用的内存的策略 |    Integer     | 0：集中清理网络中所有atomic算子占用的内存。<br>      1：不集中清理内存，对网络中每一个atomic算子进行单独清零。 <br> 默认值：1    |                                                 |
|       ENABLE_CELL_REUSE       | 开启lazy inline         |    Integer     | 1：开启<br>   0：关闭<br> 默认值：0                                                         | 此特性在mindspore≥2.2.0下适用。通常在pipeline并行时使用以提高编译性能。 |
| MS_ASCEND_CHECK_OVERFLOW_MODE | 溢出检测模式                |    String     | 默认：饱和模式，不设置此参数，当中间过程溢出时会上报，停止loss更新<br>    INFNAN_MODE：NAN模式，忽略过程中的溢出，结果非溢出就会继续训练 | 遇到持续溢出问题时可尝试设置此变量为INFNAN_MODE。                  |

## Mindspore

mindspore相关环境变量请参考以下链接：

[环境变量](https://www.mindspore.cn/docs/zh-CN/r2.2/note/env_var_list.html)
