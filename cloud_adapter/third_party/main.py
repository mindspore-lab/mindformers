# -*- coding: utf-8 -*-
"""
功能: 通过命令行等工具(argparse, click, fire)将task及mgr等子功能集中起来, 对外提供统一的接口形式
版权信息: 华为技术有限公司, 版权所有(C) 2010-2022
"""
import click

from fm.src.aicc_tools.ailog.log import service_logger, operation_logger, set_logger_source
from fm.src.engine import run_strategy, enable_verify_options, cert_verify, \
    cache_cert, manually_input_cert, config_options, config_process, \
    show_options, list_show_process, stop_options, delete_options, \
    job_status_options, model_status_options, service_status_options, \
    finetune_options, evaluate_options, infer_options, publish_options, deploy_options
from fm.src.engine.utils import which_instance_type
from fm.src.utils.concurrency_check import concurrency_check
from fm.src.utils.constants import MX_FOUNDATION_MODEL_PROCESS_NAME
from fm.src.utils.exception_utils import log_exception_details


def cli_wrapper():
    # 并发控制检查
    try:
        if not concurrency_check(MX_FOUNDATION_MODEL_PROCESS_NAME):
            return
    except Exception as ex:
        log_exception_details(ex)
        service_logger.error('exception in concurrency check, see service log for detail error message.')
        return

    try:
        if service_logger.source is None:
            set_logger_source('cli')
        cli()
    except Exception:
        service_logger.error('exception occurred, see above info for detail error message.')


# ======MAIN======
@click.group()
def cli():
    # 并发控制检查
    try:
        if not concurrency_check(MX_FOUNDATION_MODEL_PROCESS_NAME):
            raise RuntimeError
    except Exception as ex:
        log_exception_details(ex)
        service_logger.error('exception in concurrency check, see service log for detail error message.')
        raise ex


# ======MGR======
@cli.command()
@enable_verify_options()
def enable_verify(*args, **kwargs):
    return cert_verify(*args, **kwargs)


@cli.command()
def registry(*args, **kwargs):
    return cache_cert(cert=manually_input_cert())


@cli.command()
@config_options()
def config(*args, **kwargs):
    return config_process(*args, **kwargs)


@cli.command()
@show_options()
def show(*args, **kwargs):
    display = kwargs.get('display')
    kwargs.pop('display')
    instance_id, instance_type, kwargs = which_instance_type(kwargs, is_show=True)
    if instance_type:
        kwargs["instance_type"] = instance_type
    else:
        instance_type = kwargs.get('instance_type')
    kwargs["instance_id"] = instance_id
    output, scenario = run_strategy('show', **kwargs)
    if display:
        list_show_process(output, scenario, instance_type, kwargs)
        output = ''
    operation_logger.info('show ends')
    return output


@cli.command()
@stop_options()
def stop(*args, **kwargs):
    return run_strategy('stop', **kwargs)


@cli.command()
@delete_options()
def delete(*args, **kwargs):
    instance_id, instance_type, kwargs = which_instance_type(kwargs)
    kwargs["instance_id"] = instance_id
    kwargs["instance_type"] = instance_type
    return run_strategy("delete", **kwargs)


@cli.command()
@job_status_options()
def job_status(*args, **kwargs):
    kwargs["instance_id"] = kwargs.get("job_id")
    kwargs["instance_type"] = "job"
    kwargs.pop('job_id')
    return run_strategy('get_status', **kwargs)


@cli.command()
@model_status_options()
def model_status(*args, **kwargs):
    kwargs["instance_id"] = kwargs.get("model_id")
    kwargs["instance_type"] = "model"
    kwargs.pop('model_id')
    return run_strategy('get_status', **kwargs)


@cli.command()
@service_status_options()
def service_status(*args, **kwargs):
    kwargs["instance_id"] = kwargs.get("service_id")
    kwargs["instance_type"] = "service"
    kwargs.pop('service_id')
    return run_strategy('get_status', **kwargs)


# ======TASK======
@cli.command()
@finetune_options()
def finetune(*args, **kwargs):
    return run_strategy('finetune', **kwargs)


@cli.command()
@evaluate_options()
def evaluate(*args, **kwargs):
    return run_strategy('evaluate', **kwargs)


@cli.command()
@infer_options()
def infer(*args, **kwargs):
    return run_strategy('infer', **kwargs)


@cli.command()
@publish_options()
def publish(*args, **kwargs):
    return run_strategy('publish', **kwargs)


@cli.command()
@deploy_options()
def deploy(*args, **kwargs):
    return run_strategy('deploy', **kwargs)
