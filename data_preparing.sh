wget http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Movies_and_TV_5.json.gz
wget http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Movies_and_TV.json.gz
gunzip reviews_Movies_and_TV_5.json.gz
gunzip meta_Movies_and_TV.json.gz
mkdir data
python preprocess/data_processing.py
python preprocess/data_generating.py
python preprocess/vocab_generating.py
