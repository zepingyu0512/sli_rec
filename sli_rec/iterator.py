import numpy as np
import json
import pickle as pkl
import random
import math
from utils import shuffle

def load_dict(filename):
    try:
        with open(filename, 'rb') as f:
            f_json = json.load(f)
            return dict((key.encode("UTF-8"), value) for (key,value) in f_json.items())
    except:
        with open(filename, 'rb') as f:
            f_pkl = pkl.load(f)
            return dict((key.encode("UTF-8"), value) for (key,value) in f_pkl.items())

class Iterator:
    def __init__(self, source,
                 uid_voc="data/user_vocab.pkl",
                 mid_voc="data/item_vocab.pkl",
                 cat_voc="data/category_vocab.pkl",
                 batch_size=128,
                 max_batch_size=20):
        
        self.source0 = source
        self.source = shuffle(self.source0)
        self.userdict,  self.itemdict, self.catedict = load_dict(uid_voc), load_dict(mid_voc), load_dict(cat_voc)
        self.batch_size = batch_size
        self.k = batch_size * max_batch_size
        self.end_of_data = False
        self.source_buffer = []

    def __iter__(self):
        return self

    def reset(self):
        self.source= shuffle(self.source0)
        
    def get_id_numbers(self):
        return len(self.userdict), len(self.itemdict), len(self.catedict)

    def next(self):
        if self.end_of_data:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        source = []
        target = []

        if len(self.source_buffer) == 0:
            for k_ in xrange(self.k):
                ss = self.source.readline()
                if ss == "":
                    break
                self.source_buffer.append(ss.strip("\n").split("\t"))

            his_length = np.array([len(s[5].split("")) for s in self.source_buffer])
            tidx = his_length.argsort()

            _sbuf = [self.source_buffer[i] for i in tidx]
            self.source_buffer = _sbuf

        if len(self.source_buffer) == 0:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        try:
            while True:
                try:
                    ss = self.source_buffer.pop()
                except IndexError:
                    break

                uid = self.userdict[ss[1]] if ss[1] in self.userdict else 0
                mid = self.itemdict[ss[2]] if ss[2] in self.itemdict else 0
                cat = self.catedict[ss[3]] if ss[3] in self.catedict else 0
                timestepnow = float(ss[4])
                
                tmp = []
                for fea in ss[5].split(""):
                    m = self.itemdict[fea] if fea in self.itemdict else 0
                    tmp.append(m)
                mid_list = tmp

                tmp1 = []
                for fea in ss[6].split(""):
                    c = self.catedict[fea] if fea in self.catedict else 0
                    tmp1.append(c)
                cat_list = tmp1

                tmp2 = []
                for fea in ss[7].split(""):
                    tmp2.append(float(fea))
                time_list = tmp2
                
                #Time-LSTM-123 
                tmp3 = []
                for i in range(len(time_list)-1):
                    deltatime_last = (time_list[i+1] - time_list[i])/(3600 * 24)
                    if deltatime_last <= 0.5:
                        deltatime_last = 0.5
                    tmp3.append(math.log(deltatime_last))
                deltatime_now = (timestepnow - time_list[-1])/(3600 * 24)
                if deltatime_now <= 0.5:
                    deltatime_now = 0.5    
                tmp3.append(math.log(deltatime_now))               
                timeinterval_list = tmp3

                #Time-LSTM-4
                tmp4 = []
                tmp4.append(0.0)
                for i in range(len(time_list)-1):
                    deltatime_last = (time_list[i+1] - time_list[i])/(3600 * 24)
                    if deltatime_last <= 0.5:
                        deltatime_last = 0.5
                    tmp4.append(math.log(deltatime_last))
                timelast_list = tmp4
                
                tmp5 = []
                for i in range(len(time_list)):
                    deltatime_now = (timestepnow - time_list[i])/(3600 * 24)
                    if deltatime_now <= 0.5:
                        deltatime_now = 0.5
                    tmp5.append(math.log(deltatime_now))
                timenow_list = tmp5

                source.append([uid, mid, cat, mid_list, cat_list, timeinterval_list, timelast_list, timenow_list])
                target.append([float(ss[0]), 1-float(ss[0])])

                if len(source) >= self.batch_size or len(target) >= self.batch_size:
                    break
        except IOError:
            self.end_of_data = True

        if len(source) == 0 or len(target) == 0:
            source, target = self.next()

        return source, target
