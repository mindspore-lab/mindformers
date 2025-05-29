"""memory cost model"""
import mindspore.common.dtype as mstype

class Activation:
    r"""activation memory cost of one transformer layer"""
    def __init__(self, train_config, model_config):
        self.train_config = train_config
        self.model_config = model_config
        self.num_attention_heads = self.model_config.num_attention_heads
        self.micro_batch_size = self.train_config.dataset_config.batch_size
        self.hidden_size = self.model_config.hidden_size
        self.seq_len = self.model_config.seq_length

    def get_activation(self, num_layers):
        activation = num_layers * (2 * self.micro_batch_size * self.seq_len * self.hidden_size)
        compute_dtype = self.model_config.compute_dtype
        if compute_dtype == mstype.float32:
            compute_sizeof = 4
        else:
            compute_sizeof = 2
        return activation * compute_sizeof


class MemoryCostModel:
    r"""memory cost model include static memory and activation memory"""
    def __init__(self, train_config, model_config, optimizer_config):
        self.train_config = train_config
        self.model_config = model_config
        self.optimizer_config = optimizer_config
        self.unit_gb = 1024 ** 2

    def compute_params(self, num_layers):
        """compute the params of the model according to 'num_layers' """
        hidden_size = self.model_config.hidden_size
        vocab_size = self.model_config.vocab_size
        seq_length = self.model_config.seq_length

        embedding_params = hidden_size * vocab_size
        transformer_layer_params = num_layers * (2 * hidden_size * seq_length)

        return transformer_layer_params + embedding_params

    def compute_static_memory(self, num_layers):
        """compute static memory"""
        compute_dtype = self.model_config.compute_dtype
        optimizer = self.optimizer_config.optimizer_type
        if compute_dtype == mstype.float32:
            compute_sizeof = 4
        else:
            compute_sizeof = 2

        if optimizer == "mint.AdamW":
            k = 4
        else:
            k = 2

        model_paramters = self.compute_params(num_layers)
        grads = model_paramters
        optimizer_paramters = k * model_paramters
        total_static_memory = (model_paramters + grads + optimizer_paramters) \
                                * compute_sizeof

        return total_static_memory

    def get_peak_memory(self, num_layer):
        """get memory cost of each npu according to tp and num_layers in this npu"""
        activation = Activation(self.train_config, self.model_config)
        static_memory = self.compute_static_memory(num_layer)
        activation_memory = activation.get_activation(num_layer)
        return (static_memory + activation_memory) / self.unit_gb
