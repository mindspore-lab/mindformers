ARG BASE_IMAGE_TAG

FROM swr.cn-central-221.ovaijisuan.com/mindxsdk/fmtk-ma_base:${BASE_IMAGE_TAG}

ARG TRAINING_CWD=/home/ma-user
ENV DEPLOY_CWD=/home/mind/model
ARG PKG_PATH=/home/ma-user/pkg
ARG PYTHON_VERSION
ARG MINDSPORE_PYTHON_VERSION_SUFFIX
ARG MOXING_VERSION=2.0.1.rc0.ffd1c0c8

ARG CONDA_PATH=/home/ma-user/anaconda3/bin/conda
ARG CONDA_ENV_NAME=MindSpore
ENV PIP_OPTS="--trusted-host https://repo.huaweicloud.com -i https://repo.huaweicloud.com/repository/pypi/simple --user"

ARG MXTUNINGKIT_PKG
ARG CANN_TOOLKIT_PKG
ARG MINDSPORE_VERSION


USER root
########################################################################################################################
#  Add pkgs: CANN, mxTuningKit                                                                                         #
########################################################################################################################
COPY --chown=ma-user:ma-group pkg ${PKG_PATH}

########################################################################################################################
#  Install CANN                                                                                                        #
########################################################################################################################
RUN chmod +x ${PKG_PATH}/${CANN_TOOLKIT_PKG} && ${PKG_PATH}/${CANN_TOOLKIT_PKG} --full

USER ma-user
########################################################################################################################
#  Python for MindSpore and mxTuningKit                                                                                #
#  Move MoXing                                                                                                         #
########################################################################################################################
# update Python
RUN ${CONDA_PATH} update --yes -n ${CONDA_ENV_NAME} --all && \
    ${CONDA_PATH} install --yes -n ${CONDA_ENV_NAME} python=${PYTHON_VERSION} && \
    if [ ${PYTHON_VERSION} != "3.7" ]; then \
        cp -r /home/ma-user/anaconda3/envs/MindSpore/lib/python3.7/site-packages/moxing /home/ma-user/anaconda3/envs/MindSpore/lib/python${PYTHON_VERSION}/site-packages/; \
        cp -r /home/ma-user/anaconda3/envs/MindSpore/lib/python3.7/site-packages/moxing_framework-${MOXING_VERSION}.dist-info /home/ma-user/anaconda3/envs/MindSpore/lib/python${PYTHON_VERSION}/site-packages/; \
    fi

########################################################################################################################
#  1. Install MoXing for mxLaunchKit (execute MA training task without AK/SK)                                          #
#  2. Install MindSpore and mxTuningKit                                                                                #
########################################################################################################################
# MindSpore, MoXing, mxTuningKit, VEGA[Optional]
RUN pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/${MINDSPORE_VERSION}/MindSpore/ascend/aarch64/mindspore_ascend-${MINDSPORE_VERSION}-${MINDSPORE_PYTHON_VERSION_SUFFIX}-linux_aarch64.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip install $PIP_OPTS ${PKG_PATH}/${MXTUNINGKIT_PKG} && \
    pip install $PIP_OPTS -r ${PKG_PATH}/common_requirements.txt && \
    pip install $PIP_OPTS -r ${PKG_PATH}/${PYTHON_VERSION}_requirements.txt && \
    rm -rf ${PKG_PATH}
ENV LD_PRELOAD=/home/ma-user/.local/lib/python${PYTHON_VERSION}/site-packages/scikit_learn.libs/libgomp-d22c30c5.so.1.0.0 \
    SITE_PACKAGES_PATH=/home/ma-user/anaconda3/envs/MindSpore/lib/python${PYTHON_VERSION}/site-packages


########################################################################################################################
#  deployment entrypoint                                                                                               #
########################################################################################################################
CMD /usr/bin/bash ${DEPLOY_CWD}/start.sh
WORKDIR ${DEPLOY_CWD}
USER ma-user
