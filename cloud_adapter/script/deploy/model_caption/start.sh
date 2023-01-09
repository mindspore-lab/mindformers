cd $(cd "$(dirname "$0")"; pwd)
# aicc host
pip install --trusted-host 100.125.0.76 -i http://100.125.0.76:32021/repository/pypi/simple -r requirements.txt
export GLOG_v=3
export ASCEND_GLOBAL_LOG_LEVEL=3
python service.py
