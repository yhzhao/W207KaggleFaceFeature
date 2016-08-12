# Preliminary Installation and Updates
sudo apt-get -y update && sudo apt-get -y dist-upgrade

# Install dependencies
sudo apt-get -y install make cmake gcc g++ gfortran build-essential git wget freetype* linux-image-generic pkg-config  packaging-dev 

# Install LAPACK and openBLAS
sudo apt-get install -y liblapack-dev
#sudo apt-get install -y libblas-dev 
sudo apt-get install -y libopenblas-dev

# install python stuffs
sudo apt-get -y install python-dev python-pip python-setuptools 

# It is better to install virtualenv and virtualenvwrapper:
#sudo pip install virtualenv virtualenvwrapper
#Configure virtualenv and virtualenvwrapper:
#mkdir ~/.virtualenvs
#export WORKON_HOME=~/.virtualenvs
#echo ". /usr/local/bin/virtualenvwrapper.sh" >> ~/.bashrc
#source ~/.bashrc
#Create your Deep Learning environment:
#mkvirtualenv deeplearning

# A dirty installation of other python packages
# sudo pip install matplotlib
# sudo pip install cython
# sudo pip install scikit-learn
# sudo pip install scikit-image
# sudo pip install pandas
sudo apt-get -y install python-nose python-numpy python-scipy python-matplotlib cython python-pandas python-sklearn python-skimage
# Install ipython
# we need to certificate for notebook and modify iPython profile for the remote access
sudo pip install "ipython[all]"

# Install the bleeding-edge version of Theano
# sudo pip install --upgrade theano
sudo pip install --upgrade git+git://github.com/Theano/Theano.git

# Setup .theanorc config
cat <<EOF >~/.theanorc
[global]
floatX=float32
allow_gc = False
device=gpu
[cuda]
root =/usr/local/cuda
[nvcc]
fastmath = True
[lib]
cnmem=1
[blas]
ldflags = -lopenblas
EOF

# Download CUDA(7.0) toolkit
sudo wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1404/x86_64/cuda-repo-ubuntu1404_7.0-28_amd64.deb 
# Depackage CUDA 
sudo dpkg -i cuda-repo-ubuntu1404_7.0-28_amd64.deb
# Install cuda driver
sudo apt-get update
sudo apt-get install -y cuda 

# Update the path to include CUDA's lib and bin
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> .bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> .bashrc
echo 'export LIBRARY_PATH=/usr/local/cuda/lib64:$LIBRARY_PATH' >> .bashrc

#source ~/.bashrc

# Reboot to load cuda 
sudo reboot