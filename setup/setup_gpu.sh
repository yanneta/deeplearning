sudo apt-get update
sudo apt-get --assume-yes upgrade
sudo apt-get --assume-yes install tmux build-essential gcc g++ make binutils
sudo apt-get --assume-yes install software-properties-common
sudo apt autoremove
mkdir downloads
cd downloads
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_8.0.61-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1604_8.0.61-1_amd64.deb
sudo apt-get update
sudo apt-get --assume-yes install cuda
sudo modprobe nvidia
nvidia-smi

wget "http://files.fast.ai/files/cudnn-8.0-linux-x64-v6.0.tgz"
tar -zxf cudnn-8.0-linux-x64-v6.0.tgz
cd cuda
sudo cp lib64/* /usr/local/cuda/lib64/
sudo cp include/* /usr/local/cuda/include/
echo "export CUDA_HOME=/usr/local/cuda" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=\"$LD_LIBRARY_PATH:$CUDA_HOME/lib64:$CUDA_HOME/extras/CUPTI/lib64\"" >> ~/.bashrc

wget 'https://repo.continuum.io/archive/Anaconda3-4.4.0-Linux-x86_64.sh'
bash "Anaconda3-4.4.0-Linux-x86_64.sh" -b
echo "export PATH=\"$HOME/anaconda3/bin:\$PATH\"" >> ~/.bashrc
source ~/.bashrc
echo "c.NotebookApp.ip = '*'
c.NotebookApp.open_browser = False" >> $HOME/.jupyter/jupyter_notebook_config.py
#jupass=`python -c "from notebook.auth import passwd; print(passwd())"`
#echo "c.NotebookApp.password = u'"$jupass"'" >> $HOME/.jupyter/jupyter_notebook_config.py

pip install keras tensorflow-gpu
mkdir ~/.keras
echo '{
    "epsilon": 1e-07,
    "floatx": "float32",
    "image_dim_ordering": "tf",
    "backend": "tensorflow"
}' > ~/.keras/keras.json


