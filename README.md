## Adaptive User Modeling with Long and Short-Term Preference for Personalized Recommendation
This code provides an implementation of the SLi-Rec network for sequential recommendation in this paper.

## Data Preparation
```
sh data_preparing.sh
```
In data/, it will generate these files: 
- reviews_information
- meta_information
- train_data 
- test_data
- user_vocab.pkl 
- item_vocab.pkl 
- category_vocab.pkl 

## Model Implementation
The model is implemented in ```model.py```.
Training model:
```
python sli_rec/train.py
```

After training, run the following code to evaluate the model:
```
python sli_rec/test.py
```

The model below had been supported: 

Baselines:
- ASVD
- DIN
- LSTM
- LSTMPP
- NARM
- CARNN
- Time1LSTM
- Time2LSTM
- Time3LSTM
- DIEN

Our models:
- A2SVD
- T_SeqRec
- TC_SeqRec_I
- TC_SeqRec_G
- TC_SeqRec
- SLi_Rec_Fixed
- SLi_Rec_Adaptive

## Dependencies (other versions may also work):
- python==2.7
- tensorflow==1.4.1
- keras==2.1.5
- numpy==1.15.4
