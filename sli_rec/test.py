import numpy as np
from iterator import Iterator
import tensorflow as tf
from model import *
import random
from utils import *
from train import *

def test(train_file = "data/train_data", test_file = "data/test_data", save_path = "saved_model/", model_type = MODEL_TYPE, seed = SEED):
    tf.set_random_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    with tf.Session() as sess:
        train_data, test_data = Iterator(train_file), Iterator(test_file)
        user_number, item_number, cate_number = train_data.get_id_numbers()

        if model_type in MODEL_DICT: 
            cur_model = MODEL_DICT[model_type]
        else:
            print "{0} is not implemented".format(model_type)
            return
        model = cur_model(user_number, item_number, cate_number, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        model_path = save_path + model_type
        model.restore(sess, model_path)
        test_auc, test_loss, test_acc = evaluate_epoch(sess, test_data, model)
        print "test_auc: {0}, testing loss = {1}, testing accuracy = {2}".format(
              test_auc, test_loss, test_acc)

if __name__ == "__main__":
    test()