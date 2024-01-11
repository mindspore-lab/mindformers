FROM ubuntu:20.04

LABEL maintainer="Huawei"
LABEL repository="mindformers"

USER root
WORKDIR /root

SHELL ["/bin/bash", "-c"]

# system requirements
RUN sed -i s@/archive.ubuntu.com/@/mirrors.tuna.tsinghua.edu.cn/@g /etc/apt/sources.list \
    && apt clean && apt update -y && apt upgrade -y \
    && apt install vim wget gcc git -y

# conda python environment
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py39_23.11.0-2-Linux-x86_64.sh \
    && bash Miniconda3-py39_23.11.0-2-Linux-x86_64.sh -b \
    && /root/miniconda3/bin/conda init bash \
    && rm Miniconda3-py39_23.11.0-2-Linux-x86_64.sh

# Make RUN commands use the conda environment:
SHELL ["/root/miniconda3/bin/conda", "run", "-n", "base", "/bin/bash", "-c"]

RUN wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%207.0.RC1/Ascend-cann-toolkit_7.0.RC1.3_linux-x86_64.run \
    && chmod 777 Ascend-cann-toolkit_7.0.RC1.3_linux-x86_64.run \
    && bash Ascend-cann-toolkit_7.0.RC1.3_linux-x86_64.run --install --install-for-all --quiet \
    && rm Ascend-cann-toolkit_7.0.RC1.3_linux-x86_64.run \
    && echo "export LD_LIBRARY_PATH=/usr/local/Ascend/driver/:/usr/local/Ascend/driver/lib64/common/:/usr/local/Ascend/driver/lib64/driver/:\$LD_LIBRARY_PATH" >> /root/.bashrc \
    && echo "source /usr/local/Ascend/ascend-toolkit/set_env.sh" >> /root/.bashrc

# create conda env for mindspore & mindformers
RUN conda create -n mindspore2.2.1_py39 python=3.9 -y \
    && echo "conda activate mindspore2.2.1_py39" >> /root/.bashrc

# Make RUN commands use the new conda environment:
SHELL ["/root/miniconda3/bin/conda", "run", "-n", "mindspore2.2.1_py39", "/bin/bash", "-c"]

# set pip source
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# install mindspore
RUN conda install mindspore=2.2.1 -c mindspore -c conda-forge -y \
    && pip install sympy absl-py \
    && pip install /usr/local/Ascend/ascend-toolkit/latest/lib64/te-0.4.0-py3-none-any.whl --force-reinstall \
    && pip install /usr/local/Ascend/ascend-toolkit/latest/lib64/hccl-0.1.0-py3-none-any.whl --force-reinstall \
    && pip list | grep mind

# install mindspore-lite
RUN wget https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.2.1/MindSpore/lite/release/linux/x86_64/cloud_fusion/python39/mindspore_lite-2.2.1-cp39-cp39-linux_x86_64.whl \
    && pip install mindspore_lite-2.2.1-cp39-cp39-linux_x86_64.whl \
    && pip list | grep mind \
    && rm mindspore_lite-2.2.1-cp39-cp39-linux_x86_64.whl

# install mindformers
RUN pip install mindformers==0.8.0 \
    && pip list | grep mind
