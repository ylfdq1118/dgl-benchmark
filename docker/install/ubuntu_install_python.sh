# install python and pip, don't modify this, modify install_python_package.sh
apt-get update

# python 3.6
apt-get install -y software-properties-common

add-apt-repository ppa:deadsnakes/ppa
apt-get update
apt-get install -y python3.6 python3.6-dev

rm -f /usr/bin/python3 && ln -s /usr/bin/python3.6 /usr/bin/python3

# install pip
cd /tmp && wget -q https://bootstrap.pypa.io/get-pip.py
python3.6 get-pip.py

# santiy check
python3 --version
pip3 --version
