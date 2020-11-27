#!/bin/bash

# Update packages
apt-get update

# Non-interactive mode, use default answers
export DEBIAN_FRONTEND=noninteractive

# Workaround for libc6 bug - asking about service restart in non-interactive mode
# https://bugs.launchpad.net/ubuntu/+source/eglibc/+bug/935681
echo 'libc6 libraries/restart-without-asking boolean true' | debconf-set-selections

# Install Python 3.7
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt-get -y install python3.7 python3.7-dev
curl https://bootstrap.pypa.io/get-pip.py | sudo python3.7

# Add Nvidia repositories
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-repo-ubuntu1804_10.0.130-1_amd64.deb
dpkg -i cuda-repo-ubuntu1804_10.0.130-1_amd64.deb
apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
apt-get update
wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb
dpkg -i nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb
apt-get update

# Install drivers, CUDA and cuDNN
apt-get -y install --no-install-recommends nvidia-driver-418
apt-get -y install --no-install-recommends cuda-10-0 libcudnn7=\*+cuda10.0 libcudnn7-dev=\*+cuda10.0
apt-get -y install --no-install-recommends libnvinfer5=5.\*+cuda10.0 libnvinfer-dev=5.\*+cuda10.0

# Install TensorFlow
pip3.7 install tensorflow-gpu

# Install PyTorch
#pip3.7 install $(curl https://pytorch.org/assets/quick-start-module.js | grep -A1 "stable,pip,linux,cuda10.0,python3.7" | grep -oP 'https.*?\.whl')
pip3.7 install torch torchvision

# Install other Python packages
pip3.7 install numpy==1.18 pandas matplotlib pexpect opencv-contrib-python keras pillow

# Reboot
reboot
