# -*- coding: utf-8 -*-
"""
功能: 通过命令行等工具(argparse, click, fire)将task及mgr等子功能集中起来, 对外提供统一的接口形式
版权信息: 华为技术有限公司, 版权所有(C) 2010-2022
"""
import setproctitle

from fm.main import cli
from fm.engine import cert_param_existence_check, cache_cert
from fm.engine.utils import commands_generator
from fm.utils.concurrency_check import concurrency_check
from fm.utils.constants import MX_FOUNDATION_MODEL_PROCESS_NAME
from fm.aicc_tools.ailog.log import service_logger, set_logger_source
from fm.utils.exception_utils import log_exception_details

setproctitle.setproctitle(MX_FOUNDATION_MODEL_PROCESS_NAME)


def sdk_main(commands, function_name, error_rsp=None):
    try:
        set_logger_source('sdk')
        return cli.main(commands, standalone_mode=False)
    except Exception as ex:
        log_exception_details(ex)
        service_logger.error('exception in %s, see service log for detail error message.', function_name)
        return error_rsp if error_rsp is not None else False


# ======MGR======
def enable_verify(*args, **kwargs):
    commands = commands_generator('enable-verify', kwargs)
    return sdk_main(commands, 'enable_verify', kwargs)


def registry(registry_info):
    # concurrency control check
    try:
        if not concurrency_check(MX_FOUNDATION_MODEL_PROCESS_NAME):
            return False
    except Exception as ex:
        log_exception_details(ex)
        service_logger.error('exception in concurrency check, see service log for detail error message.')
        return False

    try:
        if not cert_param_existence_check(registry_info):
            return False
    except Exception as ex:
        log_exception_details(ex)
        service_logger.error('exception in registry param legality check, see service log for detail error message.')
        return False

    try:
        return cache_cert(cert=registry_info)
    except Exception as ex:
        log_exception_details(ex)
        service_logger.error('exception in local cache registry info, see service log for detail error message.')
        return False


def config(*args, **kwargs):
    commands = commands_generator('config', kwargs)
    return sdk_main(commands, 'config', error_rsp=False)


def show(*args, **kwargs):
    kwargs['display'] = kwargs.get('display') if kwargs.get('display') is not None else False
    commands = commands_generator('show', kwargs)
    return sdk_main(commands, 'show', error_rsp='')


def stop(*args, **kwargs):
    commands = commands_generator('stop', kwargs)
    return sdk_main(commands, 'stop', error_rsp=False)


def delete(*args, **kwargs):
    commands = commands_generator('delete', kwargs)
    return sdk_main(commands, 'delete', error_rsp=False)


def job_status(*args, **kwargs):
    commands = commands_generator('job-status', kwargs)
    return sdk_main(commands, 'job_status', error_rsp='')


def model_status(*args, **kwargs):
    commands = commands_generator('model-status', kwargs)
    return sdk_main(commands, 'model_status', error_rsp='')


def service_status(*args, **kwargs):
    commands = commands_generator('service-status', kwargs)
    return sdk_main(commands, 'service_status', error_rsp='')


# ======TASK======
def finetune(*args, **kwargs):
    commands = commands_generator('finetune', kwargs)
    return sdk_main(commands, 'finetune', error_rsp=-1)


def evaluate(*args, **kwargs):
    commands = commands_generator('evaluate', kwargs)
    return sdk_main(commands, 'evaluate', error_rsp=-1)


def infer(*args, **kwargs):
    commands = commands_generator('infer', kwargs)
    return sdk_main(commands, 'infer', error_rsp=-1)


def publish(*args, **kwargs):
    commands = commands_generator('publish', kwargs)
    return sdk_main(commands, 'publish', error_rsp=-1)


def deploy(*args, **kwargs):
    commands = commands_generator('deploy', kwargs)
    return sdk_main(commands, 'deploy', error_rsp=-1)
