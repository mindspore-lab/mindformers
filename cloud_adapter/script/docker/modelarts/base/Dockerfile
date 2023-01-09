FROM swr.cn-north-4.myhuaweicloud.com/aip/mindspore_1_7_0:mindspore_1.7.0-cann_5.1.0-py_3.7-euler_2.8.3-aarch64-d910-roma-20220610092605-f0b4e9c as mid

ARG CONDA_PATH=/home/ma-user/anaconda3/bin/conda
ARG CONDA_ENV_NAME=MindSpore
ARG CANN_VERSION=5.1.RC1

########################################################################################################################
#  1. Uninstall CANN                                                                                                   #
#  2. libGL for MindSpore                                                                                              #
#  3. Java 1.8                                                                                                         #
########################################################################################################################
USER root
# install libGL
COPY EulerOS.repo /etc/yum.repos.d/EulerOS.repo
RUN yum clean all && yum makecache && yum install -y mesa-libGL java-1.8.0 && \
    bash /usr/local/Ascend/ascend-toolkit/${CANN_VERSION}/aarch64-linux/script/uninstall.sh && \
    rm -rf /usr/local/Ascend/ascend-toolkit

########################################################################################################################
#  1. Remove proxy setting of Conda                                                                                    #
#  2. Uninstall MindSpore                                                                                              #
########################################################################################################################
USER ma-user
RUN ${CONDA_PATH} config --remove-key proxy_servers && \
    pip uninstall -y mindspore-ascend


FROM scratch
ARG CWD=/home/ma-user
ARG PYTHON_VERSION=3.7
COPY --from=mid / /

########################################################################################################################
#  Reset env (scratch copy will drop env)                                                                             #
########################################################################################################################
ENV LD_LIBRARY_PATH=/usr/local/lib/:/usr/lib/aarch64-linux-gnu/hdf5/serial:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/ascend-toolkit/latest/lib64:/usr/local/Ascend/ascend-toolkit/latest/compiler/lib64/plugin/opskernel:/usr/local/Ascend/ascend-toolkit/latest/compiler/lib64/plugin/nnengine::/usr/local/seccomponent/lib/:/usr/local/seccomponent/lib/openssl/ \
    TOOLCHAIN_HOME=/usr/local/Ascend/ascend-toolkit/latest/toolkit \
    SOC_VERSION=Ascend910A \
    ENV_NAME=MindSpore \
    ASCEND_AICPU_PATH=/usr/local/Ascend/ascend-toolkit/latest \
    FE_FLAG=1 \
    PWD=/home/ma-user \
    SITE_PACKAGES_PATH=/home/ma-user/anaconda3/envs/MindSpore/lib/python${PYTHON_VERSION}/site-packages \
    IS_NEW_IMAGE_FRAMEWORK=true \
    RANK_TABLE_FILE=/user/config/nbstart_hccl.json \
    NNAE_PATH=/usr/local/Ascend/nnae/latest \
    OPTION_EXEC_EXTERN_PLUGIN_PATH=/usr/local/Ascend/fwkacllib/lib64/plugin/opskernel/libfe.so:/usr/local/Ascend/fwkacllib/lib64/plugin/opskernel/libaicpu_plugin.so:/usr/local/Ascend/fwkacllib/lib64/plugin/opskernel/librts_engine.so:/usr/local/Ascend/fwkacllib/lib64/plugin/opskernel/libge_local_engine.so \
    PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
    ASCEND_OPP_PATH=/usr/local/Ascend/ascend-toolkit/latest/opp \
    DEPLOY_MODE=1 \
    SDR_LIBRARY_PATH=/usr/local/route \
    ANACONDA_DIR=/home/ma-user/anaconda3 \
    PYTHONPATH=/usr/local/Ascend/tfplugin/latest/python/site-packages:/usr/local/Ascend/ascend-toolkit/latest/python/site-packages:/usr/local/Ascend/ascend-toolkit/latest/opp/op_impl/built-in/ai_core/tbe:/usr/local/seccomponent/lib \
    PROMPT_COMMAND=EulerOS_history \
    ME_TBE_PLUGIN_PATH=/usr/local/Ascend/opp/framework/built-in/tensorflow/ \
    PATH=/home/ma-user/anaconda3/envs/MindSpore/bin:/home/ma-user/anaconda3/bin:/usr/local/Ascend/ascend-toolkit/latest/bin:/usr/local/Ascend/ascend-toolkit/latest/compiler/ccec_compiler/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/home/ma-user/.local/bin \
    ASCEND_AUTOML_PATH=/usr/local/Ascend/ascend-toolkit/latest/tools \
    AICPU_FLAG=1

########################################################################################################################
#  Add run_ascend910                                                                                                   #
########################################################################################################################
COPY --chown=ma-user:ma-group ascend910 ${CWD}/ascend910
RUN chmod -R 550 ${CWD}/ascend910

########################################################################################################################
#  Add mxLaunchKit (includes ma/launcher.py)                                                                           #
########################################################################################################################
COPY --chown=ma-user:ma-group mxLaunchKit ${CWD}/mxLaunchKit
RUN chmod -R 550 ${CWD}/mxLaunchKit
ENV PYTHONPATH=/home/ma-user/mxLaunchKit:${PYTHONPATH}

########################################################################################################################
#  Set entrypoint                                                                                                      #
########################################################################################################################
CMD /usr/bin/bash
WORKDIR ${CWD}
USER ma-user
