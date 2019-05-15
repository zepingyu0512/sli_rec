import tensorflow as tf
from rnn_cell_impl import GRUCell, LSTMCell, Time1LSTMCell, Time2LSTMCell, Time3LSTMCell, Time4LSTMCell, CARNNCell
from rnn import dynamic_rnn
from utils import *

class Model(object):
    def __init__(self, user_number, item_number, cate_number, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE):
        with tf.name_scope('Inputs'):
            self.item_history = tf.placeholder(tf.int32, [None, None], name='item_history')
            self.cate_history = tf.placeholder(tf.int32, [None, None], name='cate_history')
            self.timeinterval_history = tf.placeholder(tf.float32, [None, None], name='timeinterval_history')
            self.timelast_history = tf.placeholder(tf.float32, [None, None], name='timelast_history')
            self.timenow_history = tf.placeholder(tf.float32, [None, None], name='timenow_history')
            self.user = tf.placeholder(tf.int32, [None, ], name='user')
            self.targetitem = tf.placeholder(tf.int32, [None, ], name='targetitem')
            self.targetcate = tf.placeholder(tf.int32, [None, ], name='targetcate')
            self.mask = tf.placeholder(tf.float32, [None, None], name='mask')
            self.sequence_length = tf.placeholder(tf.int32, [None], name='sequence_length')
            self.label = tf.placeholder(tf.float32, [None, None], name='label')
            self.lr = tf.placeholder(tf.float64, [])

        with tf.name_scope('Embedding_layer'):
            self.item_lookup = tf.get_variable("item_lookup", [item_number, EMBEDDING_DIM])
            tf.summary.histogram('item_lookup', self.item_lookup)
            self.targetitem_embedding = tf.nn.embedding_lookup(self.item_lookup, self.targetitem)
            self.itemhistory_embedding = tf.nn.embedding_lookup(self.item_lookup, self.item_history)

            self.cate_lookup = tf.get_variable("cate_lookup", [cate_number, EMBEDDING_DIM])
            tf.summary.histogram('cate_lookup', self.cate_lookup)
            self.targetcate_embedding = tf.nn.embedding_lookup(self.cate_lookup, self.targetcate)
            self.catehistory_embedding = tf.nn.embedding_lookup(self.cate_lookup, self.cate_history)

        self.target_item_embedding = tf.concat([self.targetitem_embedding, self.targetcate_embedding], 1)
        self.item_history_embedding = tf.concat([self.itemhistory_embedding, self.catehistory_embedding], 2)
            
    def fcn_net(self, inps, use_dice = False):
        bn1 = tf.layers.batch_normalization(inputs=inps, name='bn1')
        dnn1 = tf.layers.dense(bn1, 200, activation=None, name='f1')
        if use_dice:
            dnn1 = dice(dnn1, name='dice_1')
        else:
            dnn1 = prelu(dnn1, 'prelu1')

        dnn2 = tf.layers.dense(dnn1, 80, activation=None, name='f2')
        if use_dice:
            dnn2 = dice(dnn2, name='dice_2')
        else:
            dnn2 = prelu(dnn2, 'prelu2')
        dnn3 = tf.layers.dense(dnn2, 2, activation=None, name='f3')
        self.y_hat = tf.nn.softmax(dnn3) + 0.00000001

        with tf.name_scope('Metrics'):
            ctr_loss = - tf.reduce_mean(tf.log(self.y_hat) * self.label)
            self.loss = ctr_loss
            tf.summary.scalar('loss', self.loss)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(self.y_hat), self.label), tf.float32))
            tf.summary.scalar('accuracy', self.accuracy)

        self.merged = tf.summary.merge_all()

    def train(self, sess, inps):
        loss, accuracy, _ = sess.run([self.loss, self.accuracy, self.optimizer], feed_dict={
            self.user: inps[0],
            self.targetitem: inps[1],
            self.targetcate: inps[2],
            self.item_history: inps[3],
            self.cate_history: inps[4],
            self.timeinterval_history: inps[5],
            self.timelast_history: inps[6],
            self.timenow_history: inps[7],
            self.mask: inps[8],
            self.label: inps[9],
            self.sequence_length: inps[10],
            self.lr: inps[11],
        })
        return loss, accuracy

    def calculate(self, sess, inps):
        probs, loss, accuracy = sess.run([self.y_hat, self.loss, self.accuracy], feed_dict={
            self.user: inps[0],
            self.targetitem: inps[1],
            self.targetcate: inps[2],
            self.item_history: inps[3],
            self.cate_history: inps[4],
            self.timeinterval_history: inps[5],
            self.timelast_history: inps[6],
            self.timenow_history: inps[7],
            self.mask: inps[8],
            self.label: inps[9],
            self.sequence_length: inps[10],
        })
        return probs, loss, accuracy

    def save(self, sess, path):
        saver = tf.train.Saver()
        saver.save(sess, save_path=path)

    def restore(self, sess, path):
        saver = tf.train.Saver()
        saver.restore(sess, save_path=path)
        print('model restored from %s' % path)

class Model_ASVD(Model):
    def __init__(self, user_number, item_number, cate_number, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE):
        super(Model_ASVD, self).__init__(user_number, item_number, cate_number, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        
        # Sum                                                  
        self.item_history_embedding_sum = tf.reduce_sum(self.item_history_embedding, 1)
        last_inps = tf.concat([self.target_item_embedding, self.item_history_embedding_sum], 1)
        self.fcn_net(last_inps, use_dice=False)
                
class Model_DIN(Model):
    def __init__(self, user_number, item_number, cate_number, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE):
        super(Model_DIN, self).__init__(user_number, item_number, cate_number, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)

        # Attention layer
        with tf.name_scope('Attention_layer'):
            attention_output = attention_DIN(self.target_item_embedding, self.item_history_embedding, ATTENTION_SIZE, self.mask)
            att_fea = tf.reduce_sum(attention_output, 1)
            tf.summary.histogram('att_fea', att_fea)
            
        last_inps = tf.concat([self.target_item_embedding, att_fea], 1)
        self.fcn_net(last_inps, use_dice=False)

class Model_LSTM(Model):
    def __init__(self, user_number, item_number, cate_number, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE):
        super(Model_LSTM, self).__init__(user_number, item_number, cate_number, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)

        # RNN layer
        with tf.name_scope('rnn'):
            rnn_outputs, final_state = dynamic_rnn(LSTMCell(HIDDEN_SIZE), inputs=self.item_history_embedding,
                                         sequence_length=self.sequence_length, dtype=tf.float32,
                                         scope="lstm")
            tf.summary.histogram('LSTM_outputs', rnn_outputs)

        last_inps = tf.concat([self.target_item_embedding, final_state[1]], 1)
        self.fcn_net(last_inps, use_dice=False)

class Model_LSTMPP(Model):
    def __init__(self, user_number, item_number, cate_number, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE):
        super(Model_LSTMPP, self).__init__(user_number, item_number, cate_number, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        
        # Attention layer
        with tf.name_scope('Attention_layer'):
            att_outputs, alphas = attention_HAN(self.item_history_embedding, attention_size=ATTENTION_SIZE, return_alphas=True)
            att_fea = tf.reduce_sum(att_outputs, 1)
            tf.summary.histogram('att_fea', att_fea)
            
        # RNN layer
        with tf.name_scope('rnn'):
            rnn_outputs, final_state = dynamic_rnn(LSTMCell(HIDDEN_SIZE), inputs=self.item_history_embedding,
                                         sequence_length=self.sequence_length, dtype=tf.float32,
                                         scope="lstm")
            tf.summary.histogram('LSTM_outputs', rnn_outputs)
         
        # alpha            
        with tf.name_scope('User_alpha'):    
            concat_all = tf.concat([self.target_item_embedding, att_fea, final_state[1], tf.expand_dims(self.timenow_history[:,-1], -1)], 1)
            concat_att1 = tf.layers.dense(concat_all, 80, activation=tf.nn.sigmoid, name='concat_att1')
            concat_att2 = tf.layers.dense(concat_att1, 40, activation=tf.nn.sigmoid, name='concat_att2')
            user_alpha = tf.layers.dense(concat_att2, 1, activation=tf.nn.sigmoid, name='concat_att3') 
            user_embed = att_fea * user_alpha + final_state[1] * (1.0 - user_alpha)

        last_inps = tf.concat([self.target_item_embedding, user_embed], 1)
        self.fcn_net(last_inps, use_dice=True)
        
class Model_NARM(Model):
    def __init__(self, user_number, item_number, cate_number, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE):
        super(Model_NARM, self).__init__(user_number, item_number, cate_number, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        
        # RNN layer(1)
        with tf.name_scope('rnn_1'):
            rnn_outputs1, final_state1 = dynamic_rnn(LSTMCell(HIDDEN_SIZE), inputs=self.item_history_embedding,
                                         sequence_length=self.sequence_length, dtype=tf.float32,
                                         scope="lstm_1")
            tf.summary.histogram('LSTM_outputs1', rnn_outputs1)

        # RNN layer(2)
        with tf.name_scope('rnn_2'):
            rnn_outputs2, final_state2 = dynamic_rnn(LSTMCell(HIDDEN_SIZE), inputs=self.item_history_embedding,
                                         sequence_length=self.sequence_length, dtype=tf.float32,
                                         scope="lstm_2")
            tf.summary.histogram('LSTM_outputs2', rnn_outputs2)
             
        # Attention layer
        with tf.name_scope('Attention_layer'):
            att_outputs, alphas = attention_FCN(final_state1[1], rnn_outputs2, ATTENTION_SIZE, self.mask, 
                                                    softmax_stag=1, stag='1_1', mode='LIST', return_alphas=True)
            tf.summary.histogram('alpha_outputs', alphas)
            att_fea = tf.reduce_sum(att_outputs, 1)

        last_inps = tf.concat([final_state1[1], att_fea, self.target_item_embedding], 1)
        self.fcn_net(last_inps, use_dice=True)
        
class Model_CARNN(Model):
    def __init__(self, user_number, item_number, cate_number, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE):
        super(Model_CARNN, self).__init__(user_number, item_number, cate_number, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        self.item_history_embedding = tf.concat([self.item_history_embedding, tf.expand_dims(self.timeinterval_history, -1)], -1)
        
        # RNN layer
        with tf.name_scope('rnn'):
            rnn_outputs, final_state = dynamic_rnn(CARNNCell(HIDDEN_SIZE), inputs=self.item_history_embedding,
                                         sequence_length=self.sequence_length, dtype=tf.float32,
                                         scope="carnn")
            tf.summary.histogram('CARNN_outputs', rnn_outputs)

        last_inps = tf.concat([final_state, self.target_item_embedding], 1)
        self.fcn_net(last_inps, use_dice=True)
                                
class Model_Time1LSTM(Model):
    def __init__(self, user_number, item_number, cate_number, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE):
        super(Model_Time1LSTM, self).__init__(user_number, item_number, cate_number, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
                                                       
        self.item_history_embedding = tf.concat([self.item_history_embedding, tf.expand_dims(self.timeinterval_history, -1)], -1)
        
        # RNN layer
        with tf.name_scope('rnn'):
            rnn_outputs, final_state = dynamic_rnn(Time1LSTMCell(HIDDEN_SIZE), inputs=self.item_history_embedding,
                                         sequence_length=self.sequence_length, dtype=tf.float32,
                                         scope="time1lstm")
            tf.summary.histogram('LSTM_outputs', rnn_outputs)

        last_inps = tf.concat([self.target_item_embedding, final_state[1]], 1)
        self.fcn_net(last_inps, use_dice=True)
        
class Model_Time2LSTM(Model):
    def __init__(self, user_number, item_number, cate_number, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE):
        super(Model_Time2LSTM, self).__init__(user_number, item_number, cate_number, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        
        self.item_history_embedding = tf.concat([self.item_history_embedding, tf.expand_dims(self.timeinterval_history, -1)], -1)
        
        # RNN layer
        with tf.name_scope('rnn'):
            rnn_outputs, final_state = dynamic_rnn(Time2LSTMCell(HIDDEN_SIZE), inputs=self.item_history_embedding,
                                         sequence_length=self.sequence_length, dtype=tf.float32,
                                         scope="time2lstm")
            tf.summary.histogram('LSTM_outputs', rnn_outputs)

        last_inps = tf.concat([self.target_item_embedding, final_state[1]], 1)
        self.fcn_net(last_inps, use_dice=True)
        
class Model_Time3LSTM(Model):
    def __init__(self, user_number, item_number, cate_number, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE):
        super(Model_Time3LSTM, self).__init__(user_number, item_number, cate_number, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
                                                       
        self.item_history_embedding = tf.concat([self.item_history_embedding, tf.expand_dims(self.timeinterval_history, -1)], -1)
        
        # RNN layer
        with tf.name_scope('rnn'):
            rnn_outputs, final_state = dynamic_rnn(Time3LSTMCell(HIDDEN_SIZE), inputs=self.item_history_embedding,
                                         sequence_length=self.sequence_length, dtype=tf.float32,
                                         scope="time3lstm")
            tf.summary.histogram('LSTM_outputs', rnn_outputs)

        last_inps = tf.concat([self.target_item_embedding, final_state[1]], 1)
        self.fcn_net(last_inps, use_dice=True)
        
class Model_DIEN(Model):
    def __init__(self, user_number, item_number, cate_number, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE):
        super(Model_DIEN, self).__init__(user_number, item_number, cate_number, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)

        # RNN layer(1)
        with tf.name_scope('rnn_1'):
            rnn_outputs, _ = dynamic_rnn(GRUCell(HIDDEN_SIZE), inputs=self.item_history_embedding,
                                         sequence_length=self.sequence_length, dtype=tf.float32,
                                         scope="gru_1")
            tf.summary.histogram('GRU_outputs', rnn_outputs)

        # Attention layer
        with tf.name_scope('Attention_layer'):
            att_outputs, alphas = attention_FCN(self.target_item_embedding, rnn_outputs, ATTENTION_SIZE, self.mask,
                                                    softmax_stag=1, stag='1_1', mode='LIST', return_alphas=True)
            tf.summary.histogram('alpha_outputs', alphas)
        
        # RNN layer(2)
        with tf.name_scope('rnn_2'):
            rnn_outputs2, final_state2 = dynamic_rnn(VecAttGRUCell(HIDDEN_SIZE), inputs=rnn_outputs,
                                                     att_scores = tf.expand_dims(alphas, -1),
                                                     sequence_length=self.sequence_length, dtype=tf.float32,
                                                     scope="gru_2")
            tf.summary.histogram('GRU2_Final_State', final_state2)

        last_inps = tf.concat([self.target_item_embedding, final_state2], 1)
        self.fcn_net(last_inps, use_dice=True)

class Model_A2SVD(Model):
    def __init__(self, user_number, item_number, cate_number, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE):
        super(Model_A2SVD, self).__init__(user_number, item_number, cate_number, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        
        # Attention layer
        with tf.name_scope('Attention_layer'):
            att_outputs, alphas = attention_HAN(self.item_history_embedding, attention_size=ATTENTION_SIZE, return_alphas=True)
            att_fea = tf.reduce_sum(att_outputs, 1)
            tf.summary.histogram('att_fea', att_fea)

        last_inps = tf.concat([self.target_item_embedding, att_fea], 1)
        self.fcn_net(last_inps, use_dice=True)
        
class Model_T_SeqRec(Model):
    def __init__(self, user_number, item_number, cate_number, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE):
        super(Model_T_SeqRec, self).__init__(user_number, item_number, cate_number, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)

        item_history_embedding_new = tf.concat([self.item_history_embedding, tf.expand_dims(self.timelast_history, -1)], -1)
        item_history_embedding_new = tf.concat([item_history_embedding_new, tf.expand_dims(self.timenow_history, -1)], -1)
        
        # RNN layer(1)
        with tf.name_scope('rnn'):
            rnn_outputs, final_state = dynamic_rnn(Time4LSTMCell(HIDDEN_SIZE), inputs=item_history_embedding_new,
                                         sequence_length=self.sequence_length, dtype=tf.float32,
                                         scope="time4lstm")
            tf.summary.histogram('LSTM_outputs', rnn_outputs)

        last_inps = tf.concat([self.target_item_embedding, final_state[1]], 1)
        self.fcn_net(last_inps, use_dice=True)
        
class Model_TC_SeqRec_I(Model):
    def __init__(self, user_number, item_number, cate_number, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE):
        super(Model_TC_SeqRec_I, self).__init__(user_number, item_number, cate_number, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        
        # Attention layer
        with tf.name_scope('Attention_layer'):
            att_outputs, alphas = attention_FCN(self.target_item_embedding, self.item_history_embedding, ATTENTION_SIZE, self.mask,
                                                    softmax_stag=1, stag='1_1', mode='LIST', return_alphas=True)
            tf.summary.histogram('alpha_outputs', alphas)
            
        item_history_embedding_new = tf.concat([self.item_history_embedding, tf.expand_dims(self.timelast_history, -1)], -1)
        item_history_embedding_new = tf.concat([item_history_embedding_new, tf.expand_dims(self.timenow_history, -1)], -1)
        
        # RNN layer
        with tf.name_scope('rnn'):
            rnn_outputs, final_state = dynamic_rnn(Time4AILSTMCell(HIDDEN_SIZE), inputs=item_history_embedding_new,
                                         att_scores = tf.expand_dims(alphas, -1),
                                         sequence_length=self.sequence_length, dtype=tf.float32,
                                         scope="time4lstm_i")
            tf.summary.histogram('LSTM_outputs', rnn_outputs)

        last_inps = tf.concat([self.target_item_embedding, final_state[1]], 1)
        self.fcn_net(last_inps, use_dice=True)
        
class Model_TC_SeqRec_G(Model):
    def __init__(self, user_number, item_number, cate_number, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE):
        super(Model_TC_SeqRec_G, self).__init__(user_number, item_number, cate_number, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        
        # Attention layer
        with tf.name_scope('Attention_layer'):
            att_outputs, alphas = attention_FCN(self.target_item_embedding, self.item_history_embedding, ATTENTION_SIZE, self.mask,
                                                    softmax_stag=1, stag='1_1', mode='LIST', return_alphas=True)
            tf.summary.histogram('alpha_outputs', alphas)
            
        item_history_embedding_new = tf.concat([self.item_history_embedding, tf.expand_dims(self.timelast_history, -1)], -1)
        item_history_embedding_new = tf.concat([item_history_embedding_new, tf.expand_dims(self.timenow_history, -1)], -1)
        
        # RNN layer
        with tf.name_scope('rnn'):
            rnn_outputs, final_state = dynamic_rnn(Time4ALSTMCell(HIDDEN_SIZE), inputs=item_history_embedding_new,
                                         att_scores = tf.expand_dims(alphas, -1),
                                         sequence_length=self.sequence_length, dtype=tf.float32,
                                         scope="time4lstm_g")
            tf.summary.histogram('LSTM_outputs', rnn_outputs)

        last_inps = tf.concat([self.target_item_embedding, final_state[1]], 1)
        self.fcn_net(last_inps, use_dice=True)
        
class Model_TC_SeqRec(Model):
    def __init__(self, user_number, item_number, cate_number, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE):
        super(Model_TC_SeqRec, self).__init__(user_number, item_number, cate_number, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)

        item_history_embedding_new = tf.concat([self.item_history_embedding, tf.expand_dims(self.timelast_history, -1)], -1)
        item_history_embedding_new = tf.concat([item_history_embedding_new, tf.expand_dims(self.timenow_history, -1)], -1)
        
        # RNN layer
        with tf.name_scope('rnn'):
            rnn_outputs, final_state = dynamic_rnn(Time4LSTMCell(HIDDEN_SIZE), inputs=item_history_embedding_new,
                                         sequence_length=self.sequence_length, dtype=tf.float32,
                                         scope="time4lstm")
            tf.summary.histogram('LSTM_outputs', rnn_outputs)

        # Attention layer
        with tf.name_scope('Attention_layer'):
            att_outputs, alphas = attention_FCN(self.target_item_embedding, rnn_outputs, ATTENTION_SIZE, self.mask,
                                                    softmax_stag=1, stag='1_1', mode='LIST', return_alphas=True)
            tf.summary.histogram('alpha_outputs', alphas)
            att_fea = tf.reduce_sum(att_outputs, 1)
                
        last_inps = tf.concat([self.target_item_embedding, att_fea], 1)
        self.fcn_net(last_inps, use_dice=True)

class Model_SLi_Rec_Fixed(Model):
    def __init__(self, user_number, item_number, cate_number, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE):
        super(Model_SLi_Rec_Fixed, self).__init__(user_number, item_number, cate_number, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        
        # Attention layer(1)
        with tf.name_scope('Attention_layer_1'):
            att_outputs1, alphas1 = attention_HAN(self.item_history_embedding, attention_size=ATTENTION_SIZE, return_alphas=True)
            att_fea1 = tf.reduce_sum(att_outputs1, 1)
            tf.summary.histogram('att_fea1', att_fea1)
            
        item_history_embedding_new = tf.concat([self.item_history_embedding, tf.expand_dims(self.timelast_history, -1)], -1)
        item_history_embedding_new = tf.concat([item_history_embedding_new, tf.expand_dims(self.timenow_history, -1)], -1)
        
        # RNN layer
        with tf.name_scope('rnn'):
            rnn_outputs, final_state1 = dynamic_rnn(Time4LSTMCell(HIDDEN_SIZE), inputs=item_history_embedding_new,
                                         sequence_length=self.sequence_length, dtype=tf.float32,
                                         scope="time4lstm")
            tf.summary.histogram('LSTM_outputs', rnn_outputs)

        # Attention layer(2)
        with tf.name_scope('Attention_layer_2'):
            att_outputs2, alphas2 = attention_FCN(self.target_item_embedding, rnn_outputs, ATTENTION_SIZE, self.mask,
                                                    softmax_stag=1, stag='1_1', mode='LIST', return_alphas=True)
            tf.summary.histogram('alpha_outputs2', alphas2)
            att_fea2 = tf.reduce_sum(att_outputs2, 1)
            tf.summary.histogram('att_fea2', att_fea2)
                 
        # alpha    
        with tf.name_scope('User_alpha'):    
            user_alpha = 0.2 
            user_embed = att_fea1 * user_alpha + att_fea2 * (1.0 - user_alpha)
            
        last_inps = tf.concat([self.target_item_embedding, user_embed], 1)
        self.fcn_net(last_inps, use_dice=True)
        
class Model_SLi_Rec_Adaptive(Model):
    def __init__(self, user_number, item_number, cate_number, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE):
        super(Model_SLi_Rec_Adaptive, self).__init__(user_number, item_number, cate_number, EMBEDDING_DIM, HIDDEN_SIZE, ATTENTION_SIZE)
        
        # Attention layer(1)
        with tf.name_scope('Attention_layer_1'):
            att_outputs1, alphas1 = attention_HAN(self.item_history_embedding, attention_size=ATTENTION_SIZE, return_alphas=True)
            att_fea1 = tf.reduce_sum(att_outputs1, 1)
            tf.summary.histogram('att_fea1', att_fea1)
            
        item_history_embedding_new = tf.concat([self.item_history_embedding, tf.expand_dims(self.timelast_history, -1)], -1)
        item_history_embedding_new = tf.concat([item_history_embedding_new, tf.expand_dims(self.timenow_history, -1)], -1)
        
        # RNN layer(1)
        with tf.name_scope('rnn'):
            rnn_outputs, final_state = dynamic_rnn(Time4LSTMCell(HIDDEN_SIZE), inputs=item_history_embedding_new,
                                         sequence_length=self.sequence_length, dtype=tf.float32,
                                         scope="time4lstm")
            tf.summary.histogram('LSTM_outputs', rnn_outputs)

        # Attention layer(2)
        with tf.name_scope('Attention_layer_2'):
            att_outputs2, alphas2 = attention_FCN(self.target_item_embedding, rnn_outputs, ATTENTION_SIZE, self.mask,
                                                    softmax_stag=1, stag='1_1', mode='LIST', return_alphas=True, scope='1')
            tf.summary.histogram('alpha_outputs2', alphas2)
            att_fea2 = tf.reduce_sum(att_outputs2, 1)
            tf.summary.histogram('att_fea2', att_fea2)
         
        # alpha           
        with tf.name_scope('User_alpha'):    
            concat_all = tf.concat([self.target_item_embedding, att_fea1, att_fea2, tf.expand_dims(self.timenow_history[:,-1], -1)], 1)
            concat_att1 = tf.layers.dense(concat_all, 80, activation=tf.nn.sigmoid, name='concat_att1')
            concat_att2 = tf.layers.dense(concat_att1, 40, activation=tf.nn.sigmoid, name='concat_att2')
            user_alpha = tf.layers.dense(concat_att2, 1, activation=tf.nn.sigmoid, name='concat_att3') 
            user_embed = att_fea1 * user_alpha + att_fea2 * (1.0 - user_alpha)

        last_inps = tf.concat([self.target_item_embedding, user_embed], 1)
        self.fcn_net(last_inps, use_dice=True)