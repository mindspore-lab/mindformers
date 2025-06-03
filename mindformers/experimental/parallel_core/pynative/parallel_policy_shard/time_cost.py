"""time cost model for heterogeneous pipeline"""
import json
import os

class TimeCostModel:
    """time cost model"""
    def __init__(self, train_config, model_config):
        self.train_config = train_config
        self.model_config = model_config
        profile_data_path = self.train_config.search_parallel_data_path
        self.profiling_data = {}
        if os.path.exists(profile_data_path):
            f = open(profile_data_path, 'r')
            self.profiling_data = json.load(f)
        self.p2p_width = 24 * (1024 ** 3)

    def get_iteration_time(self, num_layer_list=None):
        """get time of one training step"""
        pipeline_model_parallel_size = len(num_layer_list)
        micro_batch_size = self.train_config.dataset_config.batch_size
        pp_stage_forward_time = []

        for i in range(len(num_layer_list)):
            stage_forward_time = num_layer_list[i] * self.profiling_data['layer']
            if i == 0:
                stage_forward_time += self.profiling_data['embedding']
            elif i == len(num_layer_list) - 1:
                stage_forward_time += self.profiling_data['loss']
            pp_stage_forward_time.append(stage_forward_time)

        communication_num = 2 * (pipeline_model_parallel_size + micro_batch_size - 1) - 1
        communication_time = communication_num * self.get_p2p_communication_time()
        backward_time = self.profiling_data['backward_time'] * (pipeline_model_parallel_size + micro_batch_size - 1)
        forward_time = max(pp_stage_forward_time) * (pipeline_model_parallel_size + micro_batch_size - 1)
        check_overflow_time = self.profiling_data['check_overflow']
        return communication_time + backward_time + forward_time + check_overflow_time

    def get_p2p_communication_time(self):
        micro_batch_size = self.train_config.dataset_config.batch_size
        hidden_size = self.model_config.hidden_size
        seq_length = self.model_config.seq_length
        communication_data_size = micro_batch_size * hidden_size * seq_length

        return communication_data_size / self.p2p_width
