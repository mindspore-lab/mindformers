# Copyright 2024 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Test LoRA"""

import os
import pytest

import mindspore as ms


class TestLora:
    """A test class for testing lora."""
    os.environ['ASCEND_RT_VISIBLE_DEVICES'] = "0,1,2,3"

    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_single
    @pytest.mark.run(order=1)
    def test_lora_pretrain(self):
        """
        Feature: test transformer block pretrain
        Description: run pynative mode to generate transformer block pretrain model.
        Expectation: test success
        """
        os.environ['HCCL_BUFFSIZE'] = "100"
        scripts_name = "run_lora.py"
        device_num = 1

        sh_path = os.path.split(os.path.realpath(__file__))[0]
        scripts_path = os.path.join(sh_path, scripts_name)

        scripts_cmd = f"{scripts_path} --pretrain"
        cmd = f"msrun --worker_num={device_num} " + \
              f"--local_worker_num={device_num} " + \
              "--master_port=8118 " + \
              "--log_dir=msrun_log_lora_pretrain " + \
              "--join=True " + \
              "--cluster_time_out=300 " + \
              f"{scripts_cmd} " + \
              "--pretrain"
        ret = os.system(cmd)
        os.system(f"grep -E 'ERROR|error' {sh_path}/msrun_log_lora_pretrain/worker_0.log -C 3")
        assert ret == 0, "msrun failed, please check msrun_log_lora_pretrain/worker_*.log"

    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_single
    @pytest.mark.run(order=2)
    def test_lora_col_row(self):
        """
        Feature: test transformer block finetune
        Description: run pynative mode to apply lora to pretrain model.
        Expectation: test success
        """
        os.environ['HCCL_BUFFSIZE'] = "100"
        scripts_name = "run_lora.py"
        device_num = 1

        sh_path = os.path.split(os.path.realpath(__file__))[0]
        scripts_path = os.path.join(sh_path, scripts_name)

        scripts_cmd = f"{scripts_path} --standalone"
        cmd = f"msrun --worker_num={device_num} " + \
              f"--local_worker_num={device_num} " + \
              "--master_port=8108 " + \
              "--log_dir=msrun_log_lora_col_row " + \
              "--join=True " + \
              "--cluster_time_out=300 " + \
              f"{scripts_cmd}"
        ret = os.system(cmd)
        os.system(f"grep -E 'ERROR|error' {sh_path}/msrun_log_lora_col_row/worker_0.log -C 3")
        assert ret == 0, "msrun failed, please check msrun_log_lora_col_row/worker_*.log"

    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_single
    @pytest.mark.run(order=3)
    def test_lora_dp_tp_sp_col_row(self):
        """
        Feature: test transformer block finetune
        Description: run pynative mode to apply lora to pretrain model.
        Expectation: test success
        """
        os.environ['HCCL_BUFFSIZE'] = "100"
        scripts_name = "run_lora.py"
        device_num = 4

        sh_path = os.path.split(os.path.realpath(__file__))[0]
        scripts_path = os.path.join(sh_path, scripts_name)

        scripts_cmd = f"{scripts_path} --use_sequence_parallel"
        cmd = f"msrun --worker_num={device_num} " + \
              f"--local_worker_num={device_num} " + \
              "--master_port=8158 " + \
              "--log_dir=msrun_log_lora_dp_tp_sp_col_row " + \
              "--join=True " + \
              "--cluster_time_out=300 " + \
              f"{scripts_cmd}"
        ret = os.system(cmd)
        os.system(f"grep -E 'ERROR|error' {sh_path}/msrun_log_lora_dp_tp_sp_col_row/worker_0.log -C 3")
        assert ret == 0, "msrun failed, please check msrun_log_lora_dp_tp_sp_col_row/worker_*.log"

    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_single
    @pytest.mark.run(order=4)
    def test_lora_dp_tp_sp_col_row_ckpt(self):
        """
        Feature: check lora checkpoint
        Description: check the trainable parameters are valid.
        Expectation: test success
        """
        lora_ckpt0_init = "msrun_log_lora_dp_tp_sp_col_row/lora_rank0_init.ckpt"
        lora_ckpt1_init = "msrun_log_lora_dp_tp_sp_col_row/lora_rank1_init.ckpt"
        lora_ckpt2_init = "msrun_log_lora_dp_tp_sp_col_row/lora_rank2_init.ckpt"
        lora_ckpt3_init = "msrun_log_lora_dp_tp_sp_col_row/lora_rank3_init.ckpt"
        lora_ckpt0 = "msrun_log_lora_dp_tp_sp_col_row/lora_rank0.ckpt"
        lora_ckpt1 = "msrun_log_lora_dp_tp_sp_col_row/lora_rank1.ckpt"
        lora_ckpt2 = "msrun_log_lora_dp_tp_sp_col_row/lora_rank2.ckpt"
        lora_ckpt3 = "msrun_log_lora_dp_tp_sp_col_row/lora_rank3.ckpt"
        lora_params0_init = ms.load_checkpoint(lora_ckpt0_init)
        lora_params1_init = ms.load_checkpoint(lora_ckpt1_init)
        lora_params2_init = ms.load_checkpoint(lora_ckpt2_init)
        lora_params3_init = ms.load_checkpoint(lora_ckpt3_init)
        lora_params0 = ms.load_checkpoint(lora_ckpt0)
        lora_params1 = ms.load_checkpoint(lora_ckpt1)
        lora_params2 = ms.load_checkpoint(lora_ckpt2)
        lora_params3 = ms.load_checkpoint(lora_ckpt3)
        lora_params_lst = [
            'transformer.layers.0.attention.qkv_proj',
            'transformer.layers.0.attention.out_proj',
            'transformer.layers.0.mlp.mapping',
            'transformer.layers.0.mlp.projection',
            'transformer.layers.1.attention.qkv_proj',
            'transformer.layers.1.attention.out_proj',
            'transformer.layers.1.mlp.mapping',
            'transformer.layers.1.mlp.projection',
        ]

        # valid lora params:
        lora_params_lst_ckpt = []
        for (name, value) in lora_params0.items():
            if 'lora' in name:
                module_name = name.split('.')
                split = '.'
                lora_params_lst_ckpt.append(split.join(module_name[:-1]))
        lora_params_lst_ckpt = list(set(lora_params_lst_ckpt))
        assert len(lora_params_lst) == len(lora_params_lst_ckpt)
        for param in lora_params_lst_ckpt:
            assert param in lora_params_lst

        # valid duplicate_params in each rank are same
        duplicate_params = []
        not_duplicate_params = []
        for param in lora_params_lst:
            if 'qkv_proj' in param or 'mapping' in param:
                duplicate_params.append(f'{param}.lora_a')
                not_duplicate_params.append(f'{param}.lora_b')
            elif 'projection' in param or 'out_proj' in param:
                duplicate_params.append(f'{param}.lora_b')
                not_duplicate_params.append(f'{param}.lora_a')
            else:
                raise ValueError('lora_params_lst value error.')
        print('duplicate_params:', duplicate_params)
        for (name, value) in lora_params0.items():
            if name in duplicate_params:
                assert (value.asnumpy() == lora_params1[name].asnumpy()).all()
            if name in not_duplicate_params:
                assert not (value.asnumpy() == lora_params1[name].asnumpy()).all()

                # valid lora params are trained
        for (name, value) in lora_params0.items():
            if 'lora' in name:
                assert not (value.asnumpy() == lora_params0_init[name].asnumpy()).any()
                assert not (lora_params1[name].asnumpy() == lora_params1_init[name].asnumpy()).any()
                assert not (lora_params2[name].asnumpy() == lora_params2_init[name].asnumpy()).any()
                assert not (lora_params3[name].asnumpy() == lora_params3_init[name].asnumpy()).any()
            else:
                assert (value.asnumpy() == lora_params0_init[name].asnumpy()).all()
                assert (lora_params1[name].asnumpy() == lora_params1_init[name].asnumpy()).all()
                assert (lora_params2[name].asnumpy() == lora_params2_init[name].asnumpy()).all()
                assert (lora_params3[name].asnumpy() == lora_params3_init[name].asnumpy()).all()

    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_single
    @pytest.mark.run(order=5)
    def test_lora_dp_tp_col_row(self):
        """
        Feature: test transformer block finetune
        Description: run pynative mode to apply lora to pretrain model.
        Expectation: test success
        """
        os.environ['HCCL_BUFFSIZE'] = "100"
        scripts_name = "run_lora.py"
        device_num = 4

        sh_path = os.path.split(os.path.realpath(__file__))[0]
        scripts_path = os.path.join(sh_path, scripts_name)

        scripts_cmd = f"{scripts_path}"
        cmd = f"msrun --worker_num={device_num} " + \
              f"--local_worker_num={device_num} " + \
              "--master_port=8148 " + \
              "--log_dir=msrun_log_lora_dp_tp_col_row " + \
              "--join=True " + \
              "--cluster_time_out=300 " + \
              f"{scripts_cmd}"
        ret = os.system(cmd)
        os.system(f"grep -E 'ERROR|error' {sh_path}/msrun_log_lora_dp_tp_col_row/worker_0.log -C 3")
        assert ret == 0, "msrun failed, please check msrun_log_lora_dp_tp_col_row/worker_*.log"

    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_single
    @pytest.mark.run(order=6)
    def test_lora_dp_tp_col_row_ckpt(self):
        """
        Feature: check lora checkpoint
        Description: check the trainable parameters are valid.
        Expectation: test success
        """
        lora_ckpt0_init = "msrun_log_lora_dp_tp_col_row/lora_rank0_init.ckpt"
        lora_ckpt1_init = "msrun_log_lora_dp_tp_col_row/lora_rank1_init.ckpt"
        lora_ckpt2_init = "msrun_log_lora_dp_tp_col_row/lora_rank2_init.ckpt"
        lora_ckpt3_init = "msrun_log_lora_dp_tp_col_row/lora_rank3_init.ckpt"
        lora_ckpt0 = "msrun_log_lora_dp_tp_col_row/lora_rank0.ckpt"
        lora_ckpt1 = "msrun_log_lora_dp_tp_col_row/lora_rank1.ckpt"
        lora_ckpt2 = "msrun_log_lora_dp_tp_col_row/lora_rank2.ckpt"
        lora_ckpt3 = "msrun_log_lora_dp_tp_col_row/lora_rank3.ckpt"
        lora_params0_init = ms.load_checkpoint(lora_ckpt0_init)
        lora_params1_init = ms.load_checkpoint(lora_ckpt1_init)
        lora_params2_init = ms.load_checkpoint(lora_ckpt2_init)
        lora_params3_init = ms.load_checkpoint(lora_ckpt3_init)
        lora_params0 = ms.load_checkpoint(lora_ckpt0)
        lora_params1 = ms.load_checkpoint(lora_ckpt1)
        lora_params2 = ms.load_checkpoint(lora_ckpt2)
        lora_params3 = ms.load_checkpoint(lora_ckpt3)
        lora_params_lst = [
            'transformer.layers.0.attention.qkv_proj',
            'transformer.layers.0.attention.out_proj',
            'transformer.layers.0.mlp.mapping',
            'transformer.layers.0.mlp.projection',
            'transformer.layers.1.attention.qkv_proj',
            'transformer.layers.1.attention.out_proj',
            'transformer.layers.1.mlp.mapping',
            'transformer.layers.1.mlp.projection',
        ]

        # valid lora params:
        lora_params_lst_ckpt = []
        for (name, value) in lora_params0.items():
            if 'lora' in name:
                module_name = name.split('.')
                split = '.'
                lora_params_lst_ckpt.append(split.join(module_name[:-1]))
        lora_params_lst_ckpt = list(set(lora_params_lst_ckpt))
        assert len(lora_params_lst) == len(lora_params_lst_ckpt)
        for param in lora_params_lst_ckpt:
            assert param in lora_params_lst

        # valid duplicate_params in each rank are same
        duplicate_params = []
        not_duplicate_params = []
        for param in lora_params_lst:
            if 'qkv_proj' in param or 'mapping' in param:
                duplicate_params.append(f'{param}.lora_a')
                not_duplicate_params.append(f'{param}.lora_b')
            elif 'projection' in param or 'out_proj' in param:
                duplicate_params.append(f'{param}.lora_b')
                not_duplicate_params.append(f'{param}.lora_a')
            else:
                raise ValueError('lora_params_lst value error.')
        print('duplicate_params:', duplicate_params)
        for (name, value) in lora_params0.items():
            if name in duplicate_params:
                assert (value.asnumpy() == lora_params1[name].asnumpy()).all()
            if name in not_duplicate_params:
                assert not (value.asnumpy() == lora_params1[name].asnumpy()).all()

                # valid lora params are trained
        for (name, value) in lora_params0.items():
            if 'lora' in name:
                assert not (value.asnumpy() == lora_params0_init[name].asnumpy()).any()
                assert not (lora_params1[name].asnumpy() == lora_params1_init[name].asnumpy()).any()
                assert not (lora_params2[name].asnumpy() == lora_params2_init[name].asnumpy()).any()
                assert not (lora_params3[name].asnumpy() == lora_params3_init[name].asnumpy()).any()
            else:
                assert (value.asnumpy() == lora_params0_init[name].asnumpy()).all()
                assert (lora_params1[name].asnumpy() == lora_params1_init[name].asnumpy()).all()
                assert (lora_params2[name].asnumpy() == lora_params2_init[name].asnumpy()).all()
                assert (lora_params3[name].asnumpy() == lora_params3_init[name].asnumpy()).all()

    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_single
    @pytest.mark.run(order=7)
    def test_lora_dp_col_row(self):
        """
        Feature: test transformer block finetune
        Description: run pynative mode to apply lora to pretrain model.
        Expectation: test success
        """
        os.environ['HCCL_BUFFSIZE'] = "100"
        scripts_name = "run_lora.py"
        device_num = 2

        sh_path = os.path.split(os.path.realpath(__file__))[0]
        scripts_path = os.path.join(sh_path, scripts_name)

        scripts_cmd = f"{scripts_path} --parallel_strategy dp"
        cmd = f"msrun --worker_num={device_num} " + \
              f"--local_worker_num={device_num} " + \
              "--master_port=8138 " + \
              "--log_dir=msrun_log_lora_dp_col_row " + \
              "--join=True " + \
              "--cluster_time_out=300 " + \
              f"{scripts_cmd}"
        ret = os.system(cmd)
        os.system(f"grep -E 'ERROR|error' {sh_path}/msrun_log_lora_dp_col_row/worker_0.log -C 3")
        assert ret == 0, "msrun failed, please check msrun_log_lora_dp_col_row/worker_*.log"

    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_single
    @pytest.mark.run(order=8)
    def test_lora_dp_col_row_ckpt(self):
        """
        Feature: check lora checkpoint
        Description: check the trainable parameters are valid.
        Expectation: test success
        """
        lora_ckpt0_init = "msrun_log_lora_dp_col_row/lora_rank0_init.ckpt"
        lora_ckpt1_init = "msrun_log_lora_dp_col_row/lora_rank1_init.ckpt"
        lora_ckpt0 = "msrun_log_lora_dp_col_row/lora_rank0.ckpt"
        lora_ckpt1 = "msrun_log_lora_dp_col_row/lora_rank1.ckpt"
        lora_params0_init = ms.load_checkpoint(lora_ckpt0_init)
        lora_params1_init = ms.load_checkpoint(lora_ckpt1_init)
        lora_params0 = ms.load_checkpoint(lora_ckpt0)
        lora_params1 = ms.load_checkpoint(lora_ckpt1)
        lora_params_lst = [
            'transformer.layers.0.attention.qkv_proj',
            'transformer.layers.0.attention.out_proj',
            'transformer.layers.0.mlp.mapping',
            'transformer.layers.0.mlp.projection',
            'transformer.layers.1.attention.qkv_proj',
            'transformer.layers.1.attention.out_proj',
            'transformer.layers.1.mlp.mapping',
            'transformer.layers.1.mlp.projection',
        ]

        # valid lora params:
        lora_params_lst_ckpt = []
        for (name, _) in lora_params0.items():
            if 'lora' in name:
                module_name = name.split('.')
                split = '.'
                lora_params_lst_ckpt.append(split.join(module_name[:-1]))
        lora_params_lst_ckpt = list(set(lora_params_lst_ckpt))
        assert len(lora_params_lst) == len(lora_params_lst_ckpt)
        for param in lora_params_lst_ckpt:
            assert param in lora_params_lst

        # valid lora params are trained
        for (name, _) in lora_params0.items():
            if 'lora' in name:
                assert not (lora_params0[name].asnumpy() == lora_params0_init[name].asnumpy()).any()
                assert not (lora_params1[name].asnumpy() == lora_params1_init[name].asnumpy()).any()
            else:
                assert (lora_params0[name].asnumpy() == lora_params0_init[name].asnumpy()).all()
                assert (lora_params1[name].asnumpy() == lora_params1_init[name].asnumpy()).all()

    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_single
    @pytest.mark.run(order=9)
    def test_lora_tp_col_row(self):
        """
        Feature: test transformer block finetune
        Description: run pynative mode to apply lora to pretrain model.
        Expectation: test success
        """
        os.environ['HCCL_BUFFSIZE'] = "100"
        scripts_name = "run_lora.py"
        device_num = 2

        sh_path = os.path.split(os.path.realpath(__file__))[0]
        scripts_path = os.path.join(sh_path, scripts_name)

        scripts_cmd = f"{scripts_path} --parallel_strategy tp"
        cmd = f"msrun --worker_num={device_num} " + \
              f"--local_worker_num={device_num} " + \
              "--master_port=8128 " + \
              "--log_dir=msrun_log_lora_tp_col_row " + \
              "--join=True " + \
              "--cluster_time_out=300 " + \
              f"{scripts_cmd}"
        ret = os.system(cmd)
        os.system(f"grep -E 'ERROR|error' {sh_path}/msrun_log_lora_tp_col_row/worker_0.log -C 3")
        assert ret == 0, "msrun failed, please check msrun_log_lora_tp_col_row/worker_*.log"

    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_single
    @pytest.mark.run(order=10)
    def test_lora_tp_col_row_ckpt(self):
        """
        Feature: check lora checkpoint
        Description: check the trainable parameters are valid.
        Expectation: test success
        """
        lora_ckpt0_init = "msrun_log_lora_tp_col_row/lora_rank0_init.ckpt"
        lora_ckpt1_init = "msrun_log_lora_tp_col_row/lora_rank1_init.ckpt"
        lora_ckpt0 = "msrun_log_lora_tp_col_row/lora_rank0.ckpt"
        lora_ckpt1 = "msrun_log_lora_tp_col_row/lora_rank1.ckpt"
        lora_params0_init = ms.load_checkpoint(lora_ckpt0_init)
        lora_params1_init = ms.load_checkpoint(lora_ckpt1_init)
        lora_params0 = ms.load_checkpoint(lora_ckpt0)
        lora_params1 = ms.load_checkpoint(lora_ckpt1)
        lora_params_lst = [
            'transformer.layers.0.attention.qkv_proj',
            'transformer.layers.0.attention.out_proj',
            'transformer.layers.0.mlp.mapping',
            'transformer.layers.0.mlp.projection',
            'transformer.layers.1.attention.qkv_proj',
            'transformer.layers.1.attention.out_proj',
            'transformer.layers.1.mlp.mapping',
            'transformer.layers.1.mlp.projection',
        ]

        # valid lora params:
        lora_params_lst_ckpt = []
        for (name, value) in lora_params0.items():
            if 'lora' in name:
                module_name = name.split('.')
                split = '.'
                lora_params_lst_ckpt.append(split.join(module_name[:-1]))
        lora_params_lst_ckpt = list(set(lora_params_lst_ckpt))
        assert len(lora_params_lst) == len(lora_params_lst_ckpt)
        for param in lora_params_lst_ckpt:
            assert param in lora_params_lst

        # valid duplicate_params in each rank are same
        duplicate_params = []
        not_duplicate_params = []
        for param in lora_params_lst:
            if 'qkv_proj' in param or 'mapping' in param:
                duplicate_params.append(f'{param}.lora_a')
                not_duplicate_params.append(f'{param}.lora_b')
            elif 'projection' in param or 'out_proj' in param:
                duplicate_params.append(f'{param}.lora_b')
                not_duplicate_params.append(f'{param}.lora_a')
            else:
                raise ValueError('lora_params_lst value error.')
        print('duplicate_params:', duplicate_params)
        for (name, value) in lora_params0.items():
            if name in duplicate_params:
                assert (value.asnumpy() == lora_params1[name].asnumpy()).all()
            if name in not_duplicate_params:
                assert not (value.asnumpy() == lora_params1[name].asnumpy()).all()

                # valid lora params are trained
        for (name, value) in lora_params0.items():
            if 'lora' in name:
                assert not (value.asnumpy() == lora_params0_init[name].asnumpy()).any()
                assert not (lora_params1[name].asnumpy() == lora_params1_init[name].asnumpy()).any()
            else:
                assert (value.asnumpy() == lora_params0_init[name].asnumpy()).all()
                assert (lora_params1[name].asnumpy() == lora_params1_init[name].asnumpy()).all()

    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_single
    @pytest.mark.run(order=11)
    def test_lora_tp_sp_col_row(self):
        """
        Feature: test transformer block finetune
        Description: run pynative mode to apply lora to pretrain model.
        Expectation: test success
        """
        os.environ['HCCL_BUFFSIZE'] = "100"
        scripts_name = "run_lora.py"
        device_num = 2

        sh_path = os.path.split(os.path.realpath(__file__))[0]
        scripts_path = os.path.join(sh_path, scripts_name)

        scripts_cmd = f"{scripts_path} --parallel_strategy tp --use_sequence_parallel"
        cmd = f"msrun --worker_num={device_num} " + \
              f"--local_worker_num={device_num} " + \
              "--master_port=8128 " + \
              "--log_dir=msrun_log_lora_tp_sp_col_row " + \
              "--join=True " + \
              "--cluster_time_out=300 " + \
              f"{scripts_cmd}"
        ret = os.system(cmd)
        os.system(f"grep -E 'ERROR|error' {sh_path}/msrun_log_lora_tp_sp_col_row/worker_0.log -C 3")
        assert ret == 0, "msrun failed, please check msrun_log_lora_tp_sp_col_row/worker_*.log"

    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_single
    @pytest.mark.run(order=12)
    def test_lora_dp_tp_sp_col(self):
        """
        Feature: test transformer block finetune
        Description: run pynative mode to apply lora to pretrain model.
        Expectation: test success
        """
        os.environ['HCCL_BUFFSIZE'] = "100"
        scripts_name = "run_lora.py"
        device_num = 4

        sh_path = os.path.split(os.path.realpath(__file__))[0]
        scripts_path = os.path.join(sh_path, scripts_name)

        scripts_cmd = f"{scripts_path} --col_row_type col --use_sequence_parallel"
        cmd = f"msrun --worker_num={device_num} " + \
              f"--local_worker_num={device_num} " + \
              "--master_port=8168 " + \
              "--log_dir=msrun_log_lora_dp_tp_sp_col " + \
              "--join=True " + \
              "--cluster_time_out=300 " + \
              f"{scripts_cmd}"
        ret = os.system(cmd)
        os.system(f"grep -E 'ERROR|error' {sh_path}/msrun_log_lora_dp_tp_sp_col/worker_0.log -C 3")
        assert ret == 0, "msrun failed, please check msrun_log_lora_dp_tp_sp_col/worker_*.log"

    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_single
    @pytest.mark.run(order=13)
    def test_lora_dp_tp_sp_col_ckpt(self):
        """
        Feature: check lora checkpoint
        Description: check the trainable parameters are valid.
        Expectation: test success
        """
        lora_ckpt0_init = "msrun_log_lora_dp_tp_sp_col/lora_rank0_init.ckpt"
        lora_ckpt1_init = "msrun_log_lora_dp_tp_sp_col/lora_rank1_init.ckpt"
        lora_ckpt2_init = "msrun_log_lora_dp_tp_sp_col/lora_rank2_init.ckpt"
        lora_ckpt3_init = "msrun_log_lora_dp_tp_sp_col/lora_rank3_init.ckpt"
        lora_ckpt0 = "msrun_log_lora_dp_tp_sp_col/lora_rank0.ckpt"
        lora_ckpt1 = "msrun_log_lora_dp_tp_sp_col/lora_rank1.ckpt"
        lora_ckpt2 = "msrun_log_lora_dp_tp_sp_col/lora_rank2.ckpt"
        lora_ckpt3 = "msrun_log_lora_dp_tp_sp_col/lora_rank3.ckpt"
        lora_params0_init = ms.load_checkpoint(lora_ckpt0_init)
        lora_params1_init = ms.load_checkpoint(lora_ckpt1_init)
        lora_params2_init = ms.load_checkpoint(lora_ckpt2_init)
        lora_params3_init = ms.load_checkpoint(lora_ckpt3_init)
        lora_params0 = ms.load_checkpoint(lora_ckpt0)
        lora_params1 = ms.load_checkpoint(lora_ckpt1)
        lora_params2 = ms.load_checkpoint(lora_ckpt2)
        lora_params3 = ms.load_checkpoint(lora_ckpt3)
        lora_params_lst = [
            'transformer.layers.0.attention.qkv_proj',
            'transformer.layers.0.mlp.mapping',
            'transformer.layers.1.attention.qkv_proj',
            'transformer.layers.1.mlp.mapping',
        ]

        # valid lora params:
        lora_params_lst_ckpt = []
        for (name, value) in lora_params0.items():
            if 'lora' in name:
                module_name = name.split('.')
                split = '.'
                lora_params_lst_ckpt.append(split.join(module_name[:-1]))
        lora_params_lst_ckpt = list(set(lora_params_lst_ckpt))
        assert len(lora_params_lst) == len(lora_params_lst_ckpt)
        for param in lora_params_lst_ckpt:
            assert param in lora_params_lst

        # valid duplicate_params in each rank are same
        duplicate_params = []
        not_duplicate_params = []
        for param in lora_params_lst:
            if 'qkv_proj' in param or 'mapping' in param:
                duplicate_params.append(f'{param}.lora_a')
                not_duplicate_params.append(f'{param}.lora_b')
            elif 'projection' in param or 'out_proj' in param:
                duplicate_params.append(f'{param}.lora_b')
                not_duplicate_params.append(f'{param}.lora_a')
            else:
                raise ValueError('lora_params_lst value error.')
        print('duplicate_params:', duplicate_params)
        for (name, value) in lora_params0.items():
            if name in duplicate_params:
                assert (value.asnumpy() == lora_params1[name].asnumpy()).all()
            if name in not_duplicate_params:
                assert not (value.asnumpy() == lora_params1[name].asnumpy()).all()

                # valid lora params are trained
        for (name, value) in lora_params0.items():
            if 'lora' in name:
                assert not (value.asnumpy() == lora_params0_init[name].asnumpy()).any()
                assert not (lora_params1[name].asnumpy() == lora_params1_init[name].asnumpy()).any()
                assert not (lora_params2[name].asnumpy() == lora_params2_init[name].asnumpy()).any()
                assert not (lora_params3[name].asnumpy() == lora_params3_init[name].asnumpy()).any()
            else:
                assert (value.asnumpy() == lora_params0_init[name].asnumpy()).all()
                assert (lora_params1[name].asnumpy() == lora_params1_init[name].asnumpy()).all()
                assert (lora_params2[name].asnumpy() == lora_params2_init[name].asnumpy()).all()
                assert (lora_params3[name].asnumpy() == lora_params3_init[name].asnumpy()).all()

    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_single
    @pytest.mark.run(order=14)
    def test_lora_dp_tp_sp_row(self):
        """
        Feature: test transformer block finetune
        Description: run pynative mode to apply lora to pretrain model.
        Expectation: test success
        """
        os.environ['HCCL_BUFFSIZE'] = "100"
        scripts_name = "run_lora.py"
        device_num = 4

        sh_path = os.path.split(os.path.realpath(__file__))[0]
        scripts_path = os.path.join(sh_path, scripts_name)

        scripts_cmd = f"{scripts_path} --col_row_type row --use_sequence_parallel"
        cmd = f"msrun --worker_num={device_num} " + \
              f"--local_worker_num={device_num} " + \
              "--master_port=8178 " + \
              "--log_dir=msrun_log_lora_dp_tp_sp_row " + \
              "--join=True " + \
              "--cluster_time_out=300 " + \
              f"{scripts_cmd}"
        ret = os.system(cmd)
        os.system(f"grep -E 'ERROR|error' {sh_path}/msrun_log_lora_dp_tp_sp_row/worker_0.log -C 3")
        assert ret == 0, "msrun failed, please check msrun_log_lora_dp_tp_sp_row/worker_*.log"

    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend910b_training
    @pytest.mark.env_single
    @pytest.mark.run(order=15)
    def test_lora_dp_tp_sp_row_ckpt(self):
        """
        Feature: check lora checkpoint
        Description: check the trainable parameters are valid.
        Expectation: test success
        """
        lora_ckpt0_init = "msrun_log_lora_dp_tp_sp_row/lora_rank0_init.ckpt"
        lora_ckpt1_init = "msrun_log_lora_dp_tp_sp_row/lora_rank1_init.ckpt"
        lora_ckpt2_init = "msrun_log_lora_dp_tp_sp_row/lora_rank2_init.ckpt"
        lora_ckpt3_init = "msrun_log_lora_dp_tp_sp_row/lora_rank3_init.ckpt"
        lora_ckpt0 = "msrun_log_lora_dp_tp_sp_row/lora_rank0.ckpt"
        lora_ckpt1 = "msrun_log_lora_dp_tp_sp_row/lora_rank1.ckpt"
        lora_ckpt2 = "msrun_log_lora_dp_tp_sp_row/lora_rank2.ckpt"
        lora_ckpt3 = "msrun_log_lora_dp_tp_sp_row/lora_rank3.ckpt"
        lora_params0_init = ms.load_checkpoint(lora_ckpt0_init)
        lora_params1_init = ms.load_checkpoint(lora_ckpt1_init)
        lora_params2_init = ms.load_checkpoint(lora_ckpt2_init)
        lora_params3_init = ms.load_checkpoint(lora_ckpt3_init)
        lora_params0 = ms.load_checkpoint(lora_ckpt0)
        lora_params1 = ms.load_checkpoint(lora_ckpt1)
        lora_params2 = ms.load_checkpoint(lora_ckpt2)
        lora_params3 = ms.load_checkpoint(lora_ckpt3)
        lora_params_lst = [
            'transformer.layers.0.attention.out_proj',
            'transformer.layers.0.mlp.projection',
            'transformer.layers.1.attention.out_proj',
            'transformer.layers.1.mlp.projection',
        ]

        # valid lora params:
        lora_params_lst_ckpt = []
        for (name, value) in lora_params0.items():
            if 'lora' in name:
                module_name = name.split('.')
                split = '.'
                lora_params_lst_ckpt.append(split.join(module_name[:-1]))
        lora_params_lst_ckpt = list(set(lora_params_lst_ckpt))
        assert len(lora_params_lst) == len(lora_params_lst_ckpt)
        for param in lora_params_lst_ckpt:
            assert param in lora_params_lst

        # valid duplicate_params in each rank are same
        duplicate_params = []
        not_duplicate_params = []
        for param in lora_params_lst:
            if 'qkv_proj' in param or 'mapping' in param:
                duplicate_params.append(f'{param}.lora_a')
                not_duplicate_params.append(f'{param}.lora_b')
            elif 'projection' in param or 'out_proj' in param:
                duplicate_params.append(f'{param}.lora_b')
                not_duplicate_params.append(f'{param}.lora_a')
            else:
                raise ValueError('lora_params_lst value error.')
        print('duplicate_params:', duplicate_params)
        for (name, value) in lora_params0.items():
            if name in duplicate_params:
                assert (value.asnumpy() == lora_params1[name].asnumpy()).all()
            if name in not_duplicate_params:
                assert not (value.asnumpy() == lora_params1[name].asnumpy()).all()

                # valid lora params are trained
        for (name, value) in lora_params0.items():
            if 'lora' in name:
                assert not (value.asnumpy() == lora_params0_init[name].asnumpy()).any()
                assert not (lora_params1[name].asnumpy() == lora_params1_init[name].asnumpy()).any()
                assert not (lora_params2[name].asnumpy() == lora_params2_init[name].asnumpy()).any()
                assert not (lora_params3[name].asnumpy() == lora_params3_init[name].asnumpy()).any()
            else:
                assert (value.asnumpy() == lora_params0_init[name].asnumpy()).all()
                assert (lora_params1[name].asnumpy() == lora_params1_init[name].asnumpy()).all()
                assert (lora_params2[name].asnumpy() == lora_params2_init[name].asnumpy()).all()
                assert (lora_params3[name].asnumpy() == lora_params3_init[name].asnumpy()).all()
