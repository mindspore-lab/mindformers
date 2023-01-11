#!/bin/bash
# Copyright (c) Huawei Technologies Co. Ltd. 2021-2021. All rights reserved.
# ===============================================================================
set -e

cd $(cd "$(dirname "$0")"; pwd)

image_prefix=
current_time=$(date +%Y%m%d%H%M%S)

python_version=
mindspore_python_version_suffix=
mxtuningkit_pkg=
cann_toolkit_pkg=
cann_toolkit_version=
mindspore_version=

AVAILABLE_PYTHON_VERSION=([1]="3.9" [2]="3.8" [3]="3.7")
AVAILABLE_MINDSPORE_VERSION=([1]="1.7.0" [2]="1.7.1" [3]="1.8.0" [4]="1.8.1" [5]="1.9.0")


function split()
{
    printf "%100s" | tr " " "="
    echo -e ""
}

function check_pkg()
{
    for filename in ${1}; do
        if [ -f ${filename} ]; then
            return 1
        else
            return 0
        fi
    done
}

function get_pkg()
{
    echo -e "\nThe following are valid ${1} packages:\n"
    index=0
    for pkg in $(ls -1 "./pkg" | grep ${2})
    do
        echo -e "${index}. ${pkg}"
        pkgs[index]=${pkg}
        index=$(expr $index + 1)
    done
    echo -e ""
    read -p "Please input the index of ${1} package you want to install (default 0): " pkg_index
    if [ -z ${pkg_index} ]; then
        pkg_index=0
    fi
    if [[ -n ${pkg_index//[0-9]/} ]]; then
        echo -e "Invalid package index."
        return 1
    elif [ "${pkg_index}" -ge 0 -a "${pkg_index}" -lt "${index}" ]; then
        if [ ${1} == "mxtuningkit" ]; then
            mxtuningkit_pkg=${pkgs[pkg_index]}
            echo -e "Use ${mxtuningkit_pkg} in current docker image building task."
            split
            return 0
        elif [ ${1} == "cann_toolkit" ]; then
            cann_toolkit_pkg=${pkgs[pkg_index]}
            echo -e "Use ${cann_toolkit_pkg} in current docker image building task."
            split
            return 0
        fi
        return 1
    else
        echo -e "Invalid package index."
        return 1
    fi
}

function set_python_version()
{
    available_versions=""
    for available_version in ${AVAILABLE_PYTHON_VERSION[@]}
    do
        available_versions="${available_versions}, ${available_version}"
    done
    echo -e "\nAvailable python version:${available_versions: 1}\n"
    read -p "Please input the version of python you want to install (default 3.9): " version
    if [ -z ${version} ]; then
        version="3.9"
    fi
    for available_version in ${AVAILABLE_PYTHON_VERSION[@]}
    do
        if [ ${version} == ${available_version} ]; then
            python_version=${version}
            mindspore_python_version_suffix="cp${version//"."/""}-cp${version//"."/""}"
            if [ ${python_version} == "3.7" ]; then
                mindspore_python_version_suffix="${mindspore_python_version_suffix}m"
            fi
            echo -e "Python_${python_version} will be installed in current docker image building task."
            split
            return 1
        fi
    done
    return 0
}

function set_mindspore_version()
{
    available_versions=""
    for available_version in ${AVAILABLE_MINDSPORE_VERSION[@]}
    do
        available_versions="${available_versions}, ${available_version}"
    done
    echo -e "\nAvailable mindspore version:${available_versions: 1}\n"
    read -p "Please input the version of mindspore you want to install (default 1.8.1): " version
    if [ -z ${version} ]; then
        version="1.8.1"
    fi
    for available_version in ${AVAILABLE_MINDSPORE_VERSION[@]}
    do
        if [ ${version} == ${available_version} ]; then
            mindspore_version=${version}
            echo -e "mindspore_ascend_${mindspore_version} will be installed in current docker image building task."
            split
            return 1
        fi
    done
    return 0
}


function set_cann_toolkit_version()
{
    array=(${cann_toolkit_pkg//_/ })
    cann_toolkit_version=${array[1]}
}

if check_pkg "./pkg/Ascend_mindxsdk_mxTuningKit-*.whl"; then
    echo -e "mxTuningKit package is required."
elif check_pkg "./pkg/Ascend-cann-toolkit_*.run"; then
    echo -e "Installation package of CANN is required."
else
    # mxTuningKit pkg
    if get_pkg "mxtuningkit" "Ascend_mindxsdk_mxTuningKit-*"; then
        # cann-toolkit pkg
        if get_pkg "cann_toolkit" "Ascend-cann-toolkit_*"; then
            set_cann_toolkit_version
            if set_mindspore_version; then
                echo -e "Invalid mindspore version."
            elif set_python_version; then
                echo -e "Invalid python version."
            else
                image_prefix="fmtk-ma:py_${python_version}-ms_${mindspore_version}_cann_${cann_toolkit_version}-euler_2.8.3-aarch64-d910-"
                docker build --rm=true --no-cache \
                             --build-arg BASE_IMAGE_TAG="2022110901" \
                             --build-arg MXTUNINGKIT_PKG=${mxtuningkit_pkg} \
                             --build-arg CANN_TOOLKIT_PKG=${cann_toolkit_pkg} \
                             --build-arg MINDSPORE_VERSION=${mindspore_version} \
                             --build-arg PYTHON_VERSION=${python_version} \
                             --build-arg MINDSPORE_PYTHON_VERSION_SUFFIX=${mindspore_python_version_suffix} \
                             -t ${image_prefix}${current_time} -f Dockerfile .
            fi
        fi
    fi
fi
