# Preliminary Installation and Updates
sudo apt-get -y update
sudo apt-get -y upgrade
sudo apt-get -y dist-upgrade

# Install dependencies
sudo apt-get -y install make cmake gcc g++ gfortran build-essential git wget freetype* linux-image-generic pkg-config  packaging-dev python-nose python-numpy python-scipy

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

# Setting-up other Python packages 
sudo pip install matplotlib
sudo pip install cython
sudo pip install scikit-learn
sudo pip install scikit-image
sudo pip install pandas
sudo pip install ipython
# for ipython we need to certificate for notebook and modify iPython profile for the remote access

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

# Install Lasagne, pylearn2, nolearn
#sudo pip install -r https://raw.githubusercontent.com/dnouri/kfkd-tutorial/master/requirements.txt
#sudo pip install -r https://raw.githubusercontent.com/dnouri/kfkd-tutorial/master/requirements-2.txt
sudo pip install --upgrade https://github.com/Lasagne/Lasagne/archive/master.zip
sudo pip install --upgrade --no-deps git+git://github.com/dnouri/nolearn.git
sudo pip install --upgrade --no-deps git+git://github.com/lisa-lab/pylearn2.git

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
source ~/.bashrc

# download data files and save them under /data
# you have to save your own kaggle cookies to a local folder to download files  
mkdir data
cd data
wget -x --load-cookies ~/cookies.txt https://www.kaggle.com/c/facial-keypoints-detection/download/training.zip
wget -x --load-cookies ~/cookies.txt https://www.kaggle.com/c/facial-keypoints-detection/download/test.zip
wget -x --load-cookies ~/cookies.txt https://www.kaggle.com/c/facial-keypoints-detection/download/SampleSubmission.csv
wget -x --load-cookies ~/cookies.txt https://www.kaggle.com/c/facial-keypoints-detection/download/IdLookupTable.csv
mv www.kaggle.com/c/facial-keypoints-detection/download/* .
sudo apt-get install unzip
unzip training.zip
unzip test.zip
rm -rf www.kaggle.com/

cd ~

# Reboot to load cuda 
sudo reboot
