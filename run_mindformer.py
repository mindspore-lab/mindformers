"""Run MindFormer."""
import argparse
import os

from mindformers.tools.register import MindFormerConfig, ActionDict
from mindformers.common.parallel_config import build_parallel_config
from mindformers.tools.utils import str2bool
from mindformers.common.context import build_context
from mindformers.trainer import build_trainer
from mindformers.common.callback import build_callback
from mindformers.tools.cloud_adapter import cloud_monitor


@cloud_monitor()
def main(config):
    """main."""
    # init context
    cfts, profile_cb = build_context(config)

    # build context config
    config.logger.info(".........Build context config..........")
    build_parallel_config(config)
    config.logger.info("context config is:{}".format(config.parallel_config))
    config.logger.info("moe config is:{}".format(config.moe_config))

    # auto pull dataset if on ModelArts platform
    if config.pretrain_dataset:
        config.pretrain_dataset.data_loader.dataset_dir = cfts.get_dataset(
            config.pretrain_dataset.data_loader.dataset_dir)
    if config.eval_dataset:
        config.eval_dataset.data_loader.dataset_dir = cfts.get_dataset(
            config.eval_dataset.data_loader.dataset_dir)
    # auto pull checkpoint if on ModelArts platform
    if config.runner_config.load_checkpoint:
        config.runner_config.load_checkpoint = cfts.get_checkpoint(config.runner_config.load_checkpoint)

    # define callback
    callbacks = []
    if config.profile:
        callbacks.append(profile_cb)
    callbacks.extend(build_callback(config.callbacks))
    config.callbacks = callbacks

    trainer = build_trainer(config.trainer)
    if config.do_train:
        trainer.train(config)
    elif config.do_eval:
        trainer.eval(config)
    elif config.do_predict:
        trainer.predict(config)


if __name__ == "__main__":
    work_path = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        default=os.path.join(
            work_path, "configs/mae/pretrain_mae_vit_base_p16_224_400ep.yaml"),
        help='YAML config files')
    parser.add_argument('--device_id', default=None, type=int, help='device id')
    parser.add_argument('--do_train', default=None, type=str2bool, help='open training')
    parser.add_argument('--do_eval', default=None, type=str2bool, help='open evaluate')
    parser.add_argument('--do_predict', default=None, type=str2bool, help='open predict')
    parser.add_argument('--load_checkpoint', default=None, type=str, help='load model checkpoint')
    parser.add_argument('--seed', default=None, type=int, help='random seed')
    parser.add_argument('--use_parallel', default=None, type=str2bool, help='whether use parallel mode')
    parser.add_argument('--profile', default=None, type=str2bool, help='whether use profile analysis')
    parser.add_argument(
        '--options',
        nargs='+',
        action=ActionDict,
        help='override some settings in the used config, the key-value pair'
             'in xxx=yyy format will be merged into config file')

    args_ = parser.parse_args()
    config_ = MindFormerConfig(args_.config)
    if args_.device_id is not None:
        config_.context.device_id = args_.device_id
    if args_.do_train is not None:
        config_.do_train = args_.do_train
    if args_.do_eval is not None:
        config_.do_eval = args_.do_eval
    if args_.do_predict is not None:
        config_.do_predict = args_.do_predict
    if args_.seed is not None:
        config_.seed = args_.seed
    if args_.use_parallel is not None:
        config_.use_parallel = args_.use_parallel
    if args_.load_checkpoint is not None:
        config_.runner_config.load_checkpoint = args_.load_checkpoint
    if args_.profile is not None:
        config_.profile = args_.profile
    if args_.options is not None:
        config_.merge_from_dict(args_.options)

    main(config_)
