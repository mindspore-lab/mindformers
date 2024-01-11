FROM swr.cn-central-221.ovaijisuan.com/wuh-aicc_dxy/mindspore2_2_0:MindSpore2.2.0-cann7.0rc1_py_3.9-euler_2.8.3-D910B

# uninstall te hccl
USER root
RUN /home/ma-user/anaconda3/envs/MindSpore/bin/pip uninstall te hccl -y

USER ma-user
WORKDIR /home/ma-user

# reinstall te hccl
RUN ~/anaconda3/envs/MindSpore/bin/pip install /usr/local/Ascend/ascend-toolkit/latest/lib64/te-0.4.0-py3-none-any.whl -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    ~/anaconda3/envs/MindSpore/bin/pip install /usr/local/Ascend/ascend-toolkit/latest/lib64/hccl-0.1.0-py3-none-any.whl -i https://pypi.tuna.tsinghua.edu.cn/simple

# install mindfomers
RUN ~/anaconda3/envs/MindSpore/bin/pip install mindformers==0.8.0 -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    ~/anaconda3/envs/MindSpore/bin/pip list | grep mind
