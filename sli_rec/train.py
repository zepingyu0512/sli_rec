import random
import numpy as np
import tensorflow as tf
from iterator import Iterator
from model import *
from utils import *

SEED = 3
MAX_EPOCH = 10
TEST_FREQ = 1000
LR = 1e-3
EMBEDDING_DIM = 18
HIDDEN_SIZE = 36
ATTENTION_SIZE = 36
MODEL_TYPE = "SLi_Rec_Adaptive"

MODEL_DICT = {"ASVD":Model_ASVD, "DIN":Model_DIN, "LSTM":Model_LSTM, "LSTMPP":Model_LSTMPP, "NARM":Model_NARM, "CARNN":Model_CARNN,  #baselines
              "Time1LSTM":Model_Time1LSTM, "Time2LSTM":Model_Time2LSTM, "Time3LSTM":Model_Time3LSTM, "DIEN":Model_DIEN,               
              "A2SVD":Model_A2SVD, "T_SeqRec":Model_T_SeqRec, "TC_SeqRec_I":Model_TC_SeqRec_I, "TC_SeqRec_G":Model_TC_SeqRec_G,      #our models
              "TC_SeqRec":Model_TC_SeqRec, "SLi_Rec_Fixed":Model_SLi_Rec_Fixed, "SLi_Rec_Adaptive":Model_SLi_Rec_Adaptive}            

def train(train_file = "data/train_data", test_file = "data/test_data", save_path = "saved_model/", model_type = MODEL_TYPE, seed = SEED):
    tf.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    with tf.Session() as sess:
        if model_type in MODEL_DICT: 
            cur_model = MODEL_DICT[model_type]
        else:
            print "{0} is not implemented".format(model_type)
            return
            
        train_data, test_data = Iterator(train_file), Iterator(test_file)
        user_number, item_number, cate_number = train_data.get_id_numbers()            
        model = cur_model(user_number, item_number, cate_number, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        itr = 0
        learning_rate = LR
        best_auc = 0.0
        best_model_path = save_path + model_type
        for i in range(MAX_EPOCH):
            train_loss_sum = 0.0
            train_accuracy_sum = 0.0
            for src, tgt in train_data:
                user, targetitem, targetcategory, item_history, cate_history, timeinterval_history, timelast_history, timenow_history, mid_mask, label, seq_len = prepare_data(src, tgt)
                train_loss, train_acc= model.train(sess, [user, targetitem, targetcategory, item_history, cate_history, timeinterval_history,
                                                          timelast_history, timenow_history, mid_mask, label, seq_len, learning_rate])
                train_loss_sum += train_loss
                train_accuracy_sum += train_acc
                itr += 1
                if (itr % TEST_FREQ) == 0:
                    print "Iter: {0}, training loss = {1}, training accuracy = {2}".format(
                          itr, train_loss_sum / TEST_FREQ, train_accuracy_sum / TEST_FREQ)
                          
                    test_auc, test_loss, test_acc = evaluate_epoch(sess, test_data, model)
                    print "test_auc: {0}, testing loss = {1}, testing accuracy = {2}".format(
                          test_auc, test_loss, test_acc)
                          
                    if test_auc > best_auc:
                        best_auc = test_auc
                        model.save(sess, best_model_path)
                        print "Model saved in {0}".format(best_model_path)
                    
                    train_loss_sum = 0.0
                    train_accuracy_sum = 0.0

if __name__ == "__main__":
    train()