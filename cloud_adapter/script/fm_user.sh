#!/bin/bash
# Copyright 2022 Huawei Technologies Co., Ltd
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

fm_package_name="Ascend_mindxsdk_mxFoundationModel"
ma_package_name="modelarts"
cmd_para="$1"
whl_para="$2"
whl_path="$3"


fm_get_install_path()
{
  space_char=' '

  # get fm install path
  fm_install_info=$(pip show $fm_package_name | grep 'Location:')
  fm_install_path=$(echo $fm_install_info |sed 's/^Location: //g')

  # get fm bin file path
  fm_bin_info=$(whereis fm)
  fm_bin_path=${fm_bin_info#*$space_char}
  if [[ $fm_bin_path == *$space_char* ]];then
    fm_bin_path=${fm_bin_path%%$space_char*}
  fi

}

ma_get_install_path()
{
  ma_install_info=$(pip show $ma_package_name | grep 'Location:')
  ma_install_path=$(echo $ma_install_info |sed 's/^Location: //g')
}

fm_chmod_whl()
{
  chd_value=$1
  if [[ -d "$fm_install_path/fm" ]];then
    chmod $chd_value -R $fm_install_path/fm
    chmod 500 -R $fm_install_path/fm/src/kmc/
  fi
  if [[ -f "$fm_bin_path" ]];then
    chmod $chd_value  $fm_bin_path
  fi
}

ma_chmod_whl()
{
  chd_value=$1
  if [[ -d "$ma_install_path/modelarts" ]];then
    chmod $chd_value -R $ma_install_path/modelarts
  fi
}

fm_install_whl()
{
  if [[ -z $whl_path ]]; then
    whl_path="./"
  fi

  whl_file_name=$(find $whl_path -maxdepth 1 -type f -name $fm_package_name*.whl)
  echo "Begin to install mxFoundationModel wheel package ($whl_file_name)."
  if [[ -f "$whl_file_name" ]];then
    pip install $whl_file_name --log-file "$HOME/.cache/Huawei/mxFoundationModel/log/install-log.log"
    if test $? -ne 0; then
      echo "Install mxFoundationModel wheel package failed."
    else
      fm_get_install_path
      fm_chmod_whl 550
      echo "Install mxFoundationModel wheel package successfully."
    fi
  else
    echo "There is no mxFoundationModel wheel package to install."
  fi
}

ma_install_whl()
{
  if [[ -z $whl_path ]]; then
    whl_path="./"
  fi

  whl_file_name=$(find $whl_path -maxdepth 1 -type f -name $ma_package_name*.whl)
  echo "Begin to install modelarts wheel package ($whl_file_name)."
  if [[ -f "$whl_file_name" ]];then
    pip install $whl_file_name
    if test $? -ne 0; then
      echo "Install modelarts wheel package failed."
    else
      ma_get_install_path
      ma_chmod_whl 550
      echo "Install modelarts wheel package successfully."
    fi
  else
    echo "There is no modelarts wheel package to install."
  fi
}

fm_uninstall_whl()
{
  fm_get_install_path
  if [ -n "$fm_install_path" ];then
    fm_chmod_whl 750
    pip uninstall $fm_package_name --log-file "$HOME/.cache/Huawei/mxFoundationModel/log/uninstall-log.log"
    if test $? -ne 0; then
      echo "Uninstall mxFoundationModel wheel package failed."
    fi
  else
    echo "There is no mxFoundationModel wheel package to be uninstalled. Please check if it has been installed."
  fi
}

ma_uninstall_whl()
{
  ma_get_install_path
  if [ -n "$ma_install_path" ];then
    ma_chmod_whl 750
    pip uninstall $ma_package_name
    if test $? -ne 0; then
      echo "Uninstall modelarts wheel package failed."
    fi
  else
    echo "There is no modelarts wheel package to be uninstalled. Please check if it has been installed."
  fi
}

print_error()
{
  echo "command arguments are wrong! Valid arguments are necessary as follows: "
  echo "----1 install cmd:"
  echo "----    1) install ma: 'bash fm_user.sh install ma'"
  echo "----    2) install fm: 'bash fm_user.sh install fm'"
  echo "----2 uninstall cmd:"
  echo "----    1) uninstall ma: 'bash fm_user.sh uninstall ma'"
  echo "----    2) uninstall fm: 'bash fm_user.sh uninstall fm'"
}

main()
{
  umask 027
  case $cmd_para in
    install)
      case $whl_para in
        ma)
          ma_install_whl
          ;;
        fm)
          fm_install_whl
          ;;
        *)
          print_error
      esac
      ;;
    uninstall)
      case $whl_para in
        ma)
          ma_uninstall_whl
          ;;
        fm)
          fm_uninstall_whl
          ;;
        *)
          print_error
      esac
      ;;
    *)
      print_error
  esac
}

main