
# Install Lasagne, nolearn according to dnouri's kfkd-tutorial
#sudo pip install --upgrade https://github.com/Lasagne/Lasagne/archive/master.zip
#sudo pip install --upgrade --no-deps git+git://github.com/dnouri/nolearn.git
sudo pip install -r https://raw.githubusercontent.com/dnouri/kfkd-tutorial/master/requirements.txt

# Install pylearn2
#sudo pip install --upgrade --no-deps git+git://github.com/lisa-lab/pylearn2.git
sudo ldconfig /usr/local/cuda/lib64
sudo pip install -r https://raw.githubusercontent.com/dnouri/kfkd-tutorial/master/requirements-2.txt

# install Lasagne
sudo pip install Lasagne==0.1
sudo pip install --upgrade https://github.com/Theano/Theano/archive/master.zip
sudo pip install --upgrade https://github.com/Lasagne/Lasagne/archive/master.zip

# download data files and save them under /data
# you have to save your own kaggle cookies to a local folder to download files
# In addition may have to mount /dev/xvdb to an working folder with written permission (e.g. sudo mount -t ext3 /dev/xvdb data)
mkdir data
cd data
wget https://s3-us-west-1.amazonaws.com/btneuralnets/Kaggle/IdLookupTable.csv
wget https://s3-us-west-1.amazonaws.com/btneuralnets/Kaggle/SampleSubmission.csv
wget https://s3-us-west-1.amazonaws.com/btneuralnets/Kaggle/test.csv
wget https://s3-us-west-1.amazonaws.com/btneuralnets/Kaggle/training.csv

# wget -x --load-cookies ~/cookies.txt https://www.kaggle.com/c/facial-keypoints-detection/download/training.zip
# wget -x --load-cookies ~/cookies.txt https://www.kaggle.com/c/facial-keypoints-detection/download/test.zip
# wget -x --load-cookies ~/cookies.txt https://www.kaggle.com/c/facial-keypoints-detection/download/SampleSubmission.csv
# wget -x --load-cookies ~/cookies.txt https://www.kaggle.com/c/facial-keypoints-detection/download/IdLookupTable.csv
# mv www.kaggle.com/c/facial-keypoints-detection/download/* .
# sudo apt-get install unzip
# unzip training.zip
# unzip test.zip
# rm -rf www.kaggle.com/
# rm test.zip  training.zip

mkdir facial_keypoints_detection
mv * facial_keypoints_detection/
cd ~

