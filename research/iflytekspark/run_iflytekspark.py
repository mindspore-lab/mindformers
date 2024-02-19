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
"""iFlytekSpark model Train/Finetune/Eval/Export/Predict scripts."""
import os
import shutil
import argparse

# pylint: disable=W0611
from mindformers import Trainer, MindFormerConfig, AutoTokenizer
from mindformers import ContextConfig, ParallelContextConfig
from mindformers.core.context import build_context, build_profile_cb, init_context
from mindformers.tools.utils import check_in_modelarts, set_remote_save_url, str2bool
from mindformers.tools import get_output_root_path
from mindformers.tools.logger import logger
from mindformers.mindformer_book import MindFormerBook
from mindformers.modules.transformer.transformer import TransformerOpParallelConfig


# pylint: disable=W0611
from iflytekspark_model import IFlytekSparkModel
from iflytekspark_infer import IFlytekSparkInfer
from iflytekspark_tokenizer import IFlytekSparkTokenizer
from iflytekspark_streamer import IFlytekSparkStreamer
from optim import AdamWeightDecayX


def clear_auto_trans_output(config):
    """clear transformed_checkpoint and strategy"""
    if check_in_modelarts():
        import moxing as mox
        obs_strategy_dir = os.path.join(config.remote_save_url, "strategy")
        if mox.file.exists(obs_strategy_dir) and config.local_rank == 0:
            mox.file.remove(obs_strategy_dir, recursive=True)
        obs_transformed_ckpt_dir = os.path.join(config.remote_save_url, "transformed_checkpoint")
        if mox.file.exists(obs_transformed_ckpt_dir) and config.local_rank == 0:
            mox.file.remove(obs_transformed_ckpt_dir, recursive=True)
        mox.file.make_dirs(obs_strategy_dir)
        mox.file.make_dirs(obs_transformed_ckpt_dir)
    else:
        strategy_dir = os.path.join(get_output_root_path(), "strategy")
        if os.path.exists(strategy_dir) and config.local_rank % 8 == 0:
            shutil.rmtree(strategy_dir)
        transformed_ckpt_dir = os.path.join(get_output_root_path(), "transformed_checkpoint")
        if os.path.exists(transformed_ckpt_dir) and config.local_rank % 8 == 0:
            shutil.rmtree(transformed_ckpt_dir)
        os.makedirs(strategy_dir, exist_ok=True)
        os.makedirs(transformed_ckpt_dir, exist_ok=True)

def context_init(use_parallel=False, optimizer_parallel=False, device_id=0):
    """init context for mindspore."""
    context_config = ContextConfig(mode=0, device_target="Ascend", device_id=device_id)
    parallel_config = None
    if use_parallel:
        parallel_config = ParallelContextConfig(parallel_mode='SEMI_AUTO_PARALLEL',
                                                gradients_mean=False,
                                                enable_parallel_optimizer=optimizer_parallel,
                                                full_batch=True)
    init_context(use_parallel=use_parallel,
                 context_config=context_config,
                 parallel_config=parallel_config)


def main(task='text_generation',
         config='run_iflytekspark_13b_sft.yaml',
         run_mode='train',
         use_parallel=False,
         ckpt=None,
         auto_trans_ckpt=True,
         resume=False,
         train_dataset=None,
         eval_dataset=None,
         predict_data=None,
         max_length=512,
         remote_save_url=None,
         max_batch=1,
         op=True,
         device_id=0,
         prompt=None,
         tokenizer_file='',
         mindir_save_dir='',
         streamer=False):
    """main function."""
    # env init
    config_path = config
    if os.path.exists(config) and config.endswith(('.yaml', '.yml')):
        real_config_path = os.path.realpath(config)
        config = MindFormerConfig(real_config_path)
        # model_name = config.trainer.model_name
        # MindFormerBook._TRAINER_SUPPORT_TASKS_LIST[task][model_name] = real_config_path
        config.use_parallel = use_parallel
        build_context(config)
        # define callback and add profile callback
        if config.profile:
            config.profile_cb = build_profile_cb(config)
    else:
        context_init(use_parallel, op, device_id)

    # remote save url
    if check_in_modelarts() and remote_save_url:
        logger.info("remote_save_url is %s, the output file will be uploaded to here.", remote_save_url)
        set_remote_save_url(remote_save_url)
        config.remote_save_url = remote_save_url
    if not ckpt and config.load_checkpoint:
        ckpt = config.load_checkpoint
    if auto_trans_ckpt != config.auto_trans_ckpt:
        config.auto_trans_ckpt = auto_trans_ckpt
    if hasattr(config, 'auto_trans_ckpt') and config.auto_trans_ckpt:
        clear_auto_trans_output(config)

    # Define tasks and prepare corresponding datasets
    if run_mode == 'train':
        task = Trainer(args=config,
                       task=task,
                       train_dataset=train_dataset,
                       eval_dataset=eval_dataset)
        task.train(train_checkpoint=ckpt, auto_trans_ckpt=config.auto_trans_ckpt, resume_training=resume)

    elif run_mode == 'finetune':
        task = Trainer(args=config,
                       task=task,
                       train_dataset=train_dataset,
                       eval_dataset=eval_dataset)
        task.finetune(finetune_checkpoint=ckpt, auto_trans_ckpt=config.auto_trans_ckpt, resume_training=resume)

    elif run_mode == 'eval':
        task = Trainer(args=config,
                       task=task,
                       eval_dataset=eval_dataset)
        task.evaluate(eval_checkpoint=ckpt, auto_trans_ckpt=config.auto_trans_ckpt)

    elif run_mode == 'predict':
        # prepare data
        import jsonlines
        if predict_data.startswith('[') and predict_data.endswith(']'):
            predict_data = predict_data.lstrip('[').rstrip(']')
            predict_data = list(predict_data.split('##'))
        elif predict_data.endswith(('.json', 'jsonl')):
            predict_path = predict_data
            predict_data = []
            with jsonlines.open(predict_path, mode="r") as f:
                for line in f:
                    predict_data.append(line["input"])
        else:
            raise RuntimeError("Please check your 'predict_data' format!")

        data = []
        if prompt is not None:
            for item in predict_data:
                data.append(prompt.format(item))
        tokenizer = IFlytekSparkTokenizer(tokenizer_file)
        inputs = tokenizer(data, return_tensors=None, add_special_tokens=False)

        parallel_config = TransformerOpParallelConfig(config.parallel_config.data_parallel,
                                                      config.parallel_config.model_parallel,
                                                      vocab_emb_dp=config.parallel_config.vocab_emb_dp)
        streamer = IFlytekSparkStreamer(tokenizer) if streamer else None
        model = IFlytekSparkInfer(config_path,
                                  ckpt_path=ckpt,
                                  max_length=max_length,
                                  max_batch=max_batch,
                                  tokenizer=tokenizer,
                                  streamer=streamer,
                                  use_parallel=use_parallel,
                                  rank_id=config.local_rank,
                                  parallel_config=parallel_config)
        # infer
        if model.max_batch == 1:
            outs = model.predict(inputs["input_ids"])
        else:
            input_data = inputs["input_ids"]
            length = len(input_data) // model.max_batch
            outs = []
            for i in range(length):
                cur_input = input_data[i * model.max_batch: (i + 1) * model.max_batch]
                out = model.predict(cur_input)
                outs.extend(out)
                print(f"batch {i * model.max_batch} - {(i + 1) * model.max_batch - 1} infer output: ", out)

        # write json
        with jsonlines.open(
                f"./log/infer_result_num{len(inputs['input_ids'])}_rank{config.local_rank}.json",
                mode="w") as f:
            for idx in range(len(outs)):
                f.write({f"predict": outs[idx]})

        print("\nall of output: ", outs)

    elif run_mode == 'export':
        print("start export mindir")
        assert mindir_save_dir != '', "mindir_save_dir must be set in when export mode"
        parallel_config = TransformerOpParallelConfig(config.parallel_config.data_parallel,
                                                      config.parallel_config.model_parallel,
                                                      vocab_emb_dp=config.parallel_config.vocab_emb_dp)
        model = IFlytekSparkInfer(config_path, ckpt_path=ckpt,
                                  max_length=max_length,
                                  max_batch=max_batch,
                                  use_parallel=use_parallel,
                                  rank_id=config.local_rank,
                                  parallel_config=parallel_config)
        model.export_mindir(mindir_save_dir)

        print("\n=====================")
        print("=                   =")
        print("=  Export success!  =")
        print("=                   =")
        print("=====================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='text_generation', type=str,
                        help='set task type. Default: text_generation.')
    parser.add_argument('--config', default='run_iflytekspark_13b_sft.yaml', type=str,
                        help='config file path. Default: run_islytekspark_13b_sft.yaml.')
    parser.add_argument('--run_mode', default='train', type=str,
                        help='set run mode for model. Default: train.')
    parser.add_argument('--use_parallel', default=False, type=str2bool,
                        help='open parallel for model. Default: False.')
    parser.add_argument('--load_checkpoint', default=None, type=str,
                        help='checkpoint name or dir to load.')
    parser.add_argument('--remote_save_url', default=None, type=str,
                        help='remote save url, the output files will tansferred and stroed here. Default: None')
    parser.add_argument('--auto_trans_ckpt', default=True, type=str2bool,
                        help='whether to transform checkpoint to the checkpoint matching current distribute strategy.')
    parser.add_argument('--resume', default=False, type=str2bool,
                        help='whether resume training. Default: False.')
    parser.add_argument('--train_dataset', default=None, type=str,
                        help='set train dataset. Default: None.')
    parser.add_argument('--eval_dataset', default=None, type=str,
                        help='set eval dataset. Default: None.')
    parser.add_argument('--predict_data', default=None, type=str,
                        help='input predict data. Default: None.')
    parser.add_argument('--predict_length', default=512, type=int,
                        help='max length for predict output. Default: 512.')
    parser.add_argument('--predict_batch', default=1, type=int,
                        help='max batch for predict output. Default: 1')
    parser.add_argument('--optimizer_parallel', default=False, type=str2bool,
                        help='whether use optimizer parallel. Default: False')
    parser.add_argument('--device_id', default=0, type=int,
                        help='ID of the target device, the value must be in [0, device_num_per_host-1], Default: 0.')
    parser.add_argument('--prompt', default=None, type=str,
                        help='prompt for infer(predict)')
    parser.add_argument('--tokenizer_file', default='', type=str,
                        help='tokenizer file path')
    parser.add_argument('--mindir_save_dir', default='', type=str,
                        help='set path to save exported mindir')
    parser.add_argument('--streamer', default=False, type=str2bool,
                        help='use streamer. Default: False.')
    args = parser.parse_args()

    main(task=args.task,
         config=args.config,
         run_mode=args.run_mode,
         use_parallel=args.use_parallel,
         ckpt=args.load_checkpoint,
         auto_trans_ckpt=args.auto_trans_ckpt,
         resume=args.resume,
         train_dataset=args.train_dataset,
         eval_dataset=args.eval_dataset,
         predict_data=args.predict_data,
         max_length=args.predict_length,
         remote_save_url=args.remote_save_url,
         max_batch=args.predict_batch,
         op=args.optimizer_parallel,
         device_id=args.device_id,
         prompt=args.prompt,
         tokenizer_file=args.tokenizer_file,
         mindir_save_dir=args.mindir_save_dir,
         streamer=args.streamer)
