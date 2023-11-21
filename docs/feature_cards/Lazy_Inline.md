# Lazy Inline

## 基本介绍

神经网络模型的编译过程往往采用默认inline的方式，把层级的代码表达最终展开成一张扁平的计算图，一方面寻求最大的编译优化机会，另一方面也可以简化自动微分以及执行的逻辑。inline后形成的计算图包含了所有的计算节点，可以在更大的范围内进行优化，比如常量折叠、节点融合、并行分析等，也可以更好地实现内存分配，减少内存申请和性能开销。虽然inline优化对于运行期性能提升帮助非常大，但过度inline也带来了编译期的负担。例如随着计算图节点数量膨胀，执行pass的耗时也在急剧增长。

为了减轻inline对编译性能带来的损耗，对于重复调用相同计算单元的场景（典型的场景是在for循环中调用同一个Cell类的不同实例），我们支持通过环境变量的方式调用Mindspore的`lazy_inline`方法来减少编译时间。

mindspore实现参考：

[mindspore.lazy_inline](https://www.mindspore.cn/docs/zh-CN/r2.2/api_python/mindspore/mindspore.lazy_inline.html)

具体原理参考：

[Lazy inline-百亿/千亿大语言模型静态图编译性能提升N倍的的思路和实践](https://www.mindspore.cn/news/newschildren?id=2657)

当启用`pipeline`并行时，模型规模和节点数加大，如果原来图的规模是`O`，那开启`pipeline`并行，单节点图的规模变为`(O/X)*Y`，其中`X`为`pipeline`的`stage`数量，`Y`为`microbatch`的数量，在实际的配置过程中，`Y`比`X`大很多，比如`X`为`16`，而`Y`一般设置到`64-192`，这样开启流水线并行后，图编译的规模会进一步增大到原来的`4-12`倍。

开启流水线并行，各个`micro batch`的`Layer`层是完全一样的。按照`micro batch`为边界，保留`micro batch`的子图结构，那么理论上编译时间可以变为原来的`Y`分之一。具体做法为在相关的`layer`类上打标记，给编译器提示，打上标记的`layer`不论是在循环体内被调用，还是其他方式被调用，在编译期间都不内联，直到执行前，才进行内联展开，从而大幅提升了编译性能。

## 使用说明

**注：此特性在mindspore≥2.2.0下适用。通常在`pipeline`并行时使用以提高编译性能。**

对于模型，可以通过在`__init__`函数上注册装饰器`cell_reuse`，指定一个cell是可复用的。此装饰器会按照`attrs`的值去添加`__init__`函数对应的入参作为cell的属性。示例如下：

```python
from mindformers.models.utils import cell_reuse
from mindformers.modules.transformer.op_parallel_config import _check_config
from mindformers.models.base_model import BaseModel
from mindformers.models.llama.llama_config import LlamaConfig

class Baichuan7BV2ForCausalLM(BaseModel):
    #注册装饰器
    @cell_reuse
    def __init__(self, config: LlamaConfig = None):
        super(Baichuan7BV2ForCausalLM, self).__init__(config, auto_prefix=True)
        _check_config(config.parallel_config)
        self.ignore_token_id = config.ignore_token_id
```

在模型启动前，通过设置环境变量`ENABLE_CELL_REUSE=1`，开启lazy inline。
