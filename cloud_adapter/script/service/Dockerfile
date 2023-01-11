FROM swr.cn-north-4.myhuaweicloud.com/modelarts-job-dev-image/mindspore-ascend910-cp37-euleros2.8-aarch64-training:1.3.0-3.3.0-roma
MAINTAINER ***
WORKDIR /home/work
COPY modelarts-latest-py2.py3-none-any.whl /home/work
COPY mindxsdk_mxfoundationmodel-0.0.1-py3-none-any.whl /home/work

# root user install ascend tookkit
USER root
RUN /usr/local/Ascend/nnae/latest/script/uninstall.sh
ADD Ascend-cann-toolkit_5.0.4_linux-aarch64.run /home/work/Ascend-cann-toolkit_5.0.4_linux-aarch64.run
RUN chmod +x /home/work/Ascend-cann-toolkit_5.0.4_linux-aarch64.run
RUN /home/work/Ascend-cann-toolkit_5.0.4_linux-aarch64.run --full

# set ascend env
ENV LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/lib64:/usr/local/Ascend/ascend-toolkit/latest/compiler/lib64/plugin/opskernel:/usr/local/Ascend/ascend-toolkit/latest/compiler/lib64/plugin/nnengine:$LD_LIBRARY_PATH \
    PYTHONPATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages:/usr/local/Ascend/ascend-toolkit/latest/opp/op_impl/built-in/ai_core/tbe:$PYTHONPATH \
    PATH=/usr/local/Ascend/ascend-toolkit/latest/bin:/usr/local/Ascend/ascend-toolkit/latest/compiler/ccec_compiler/bin:$PATH \
    ASCEND_AICPU_PATH=/usr/local/Ascend/ascend-toolkit/latest \
    ASCEND_OPP_PATH=/usr/local/Ascend/ascend-toolkit/latest/opp \
    TOOLCHAIN_HOME=/usr/local/Ascend/ascend-toolkit/latest/toolkit \
    ASCEND_AUTOML_PATH=/usr/local/Ascend/ascend-toolkit/latest/tools

# work user install other packages
USER work
RUN pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/1.5.0/MindSpore/ascend/aarch64/mindspore_ascend-1.5.0-cp37-cp37m-linux_aarch64.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple \
    && pip install /home/work/modelarts-latest-py2.py3-none-any.whl -i https://pypi.tuna.tsinghua.edu.cn/simple && pip install /home/work/mindxsdk_mxfoundationmodel-0.0.1-py3-none-any.whl -i https://pypi.tuna.tsinghua.edu.cn/simple


# application layer
RUN pip install Flask -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install Flask cryptography -i https://pypi.tuna.tsinghua.edu.cn/simple

# CMD     ["/usr/sbin/sshd", "-D"]

# copy start script to container from host
COPY start_service.py /home/work/start_service.py

# defaul start script
CMD python3 /home/work/start_service.py