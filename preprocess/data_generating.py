import sys
import hashlib
import random

if __name__=="__main__":
    f_input = open("data/preprocessed_data", "r")
    f_train = open("data/train_data", "w")
    f_test = open("data/test_data", "w")
    
    print "data generating..."
    last_user_id = None
    for line in f_input:
        line_split = line.strip().split("\t")
        tfile = line_split[0]
        label = int(line_split[1])
        user_id = line_split[2]
        movie_id = line_split[3]
        date_time = line_split[4]
        category = line_split[5]
    
        if tfile == "train":
            fo = f_train
        else:
            fo = f_test
        if user_id != last_user_id:
            movie_id_list = []
            cate_list = []
            dt_list = []
        else:
            history_clk_num = len(movie_id_list)
            cat_str = ""
            mid_str = ""
            dt_str = ""
            for c1 in cate_list:
                cat_str += c1 + ""
            for mid in movie_id_list:
                mid_str += mid + ""
            for dt_time in dt_list:
                dt_str += dt_time + ""
            if len(cat_str) > 0: cat_str = cat_str[:-1]
            if len(mid_str) > 0: mid_str = mid_str[:-1]
            if len(dt_str) > 0: dt_str = dt_str[:-1]
            if history_clk_num >= 1: 
                fo.write(line_split[1] + "\t" + user_id + "\t" + movie_id + "\t" + category + "\t" 
                        + date_time + "\t" + mid_str + "\t" + cat_str + "\t" + dt_str + "\n")
        last_user_id = user_id
        if label:
            movie_id_list.append(movie_id)
            cate_list.append(category)  
            dt_list.append(date_time)              
