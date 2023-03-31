#install gdown 
pip install gdown

# download and extract PointDA-10 dataset
gdown https://drive.google.com/u/0/uc\?export\=download\&confirm\=nZfC\&id\=1-LfJWL5geF9h0Z2QpdTL0n4lShy8wy2J
unzip PointDA_data.zip -d data

# remove .zip
rm PointDA_data.zip
