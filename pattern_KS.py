from _KS_module import *
import numpy as np
import os
import multiprocessing as mp
import time


def pattern_main(args):
    # print("Process start")
    sig_type = args[0]
    sig_tr = args[1]
    sig_te = args[2]
    iter_i = args[3]
    colname = args[4]
    index = args[5]
    # sig_type = list(tr_dic.keys())
    # sig_tr = list(tr_dic.values())[50*i:50*(i+1)]
    # sig_te = list(te_dic.values())[50*i:50*(i+1)]
    sig_df = []
    for user_iter in range(50):
        iter_type = sig_type[user_iter]
        print("iter : {}/26".format(iter_i),
                  " type : {}".format(iter_type[1][0]),
                  " figure : {}".format(iter_type[1][1]),
                  " user : {}/50".format(str(user_iter+1)))
        try:    
            tmp = sig_tr[user_iter][0]
            stroke_num = len(np.where(tmp['action.0'].values == 'DOWN')[0])
            # valid_tr_ = tr_dic[('39', ('sig-en',1))]
            valid_tr_ = sig_tr[user_iter]

            valid_tr = [[] for i in range(stroke_num)]
            for i in range(len(valid_tr_)):
                tmp = diff_dataframe(valid_tr_[i])
                for j, v in enumerate(tmp):
                    valid_tr[j].append(v)

            valid_tr = [pd.concat(valid_tr[i]) for i in range(len(valid_tr))]
            valid_te = sig_te[user_iter]

            test_te = []
            user_idx = list(range(50))
            del user_idx[user_iter]
            for j in user_idx:
                tmp = sig_te[j]
                for t in range(10):
                    test_te.append(tmp[t])

            test_df, valid_df = [], []
            for i in range(len(test_te)):
                test_df.append(KS_calculator(valid_tr, test_te[i]))
                #printProgress(i, len(test_te), 'Calculate KS :', '', 1, 50)
            test_df = pd.concat(test_df)

            for i in range(len(valid_te)):
                valid_df.append(KS_calculator(valid_tr, valid_te[i]))
                #printProgress(i, len(valid_te), 'Calculate KS :', '', 1, 50)
            valid_df = pd.concat(valid_df)

            ks_df = final_df(valid_df, test_df, index, colname)
            ks_df.to_csv(os.path.join(output_path, 
                                      "{}-{}_{}.csv".format(iter_type[1][0], iter_type[1][1],iter_type[0])),
                                     columns=colname+['Target'], index=False)
            df = get_auroc(ks_df)
            sig_df.append(df)
        except Exception as e:
            print("iter : {}/26".format(iter_i),
                  " type : {}".format(iter_type[1][0]),
                  " figure : {}".format(iter_type[1][1]),
                  " user : {}/50".format(str(user_iter+1)),
                  " Error!!!!!! ")
            print(e)
            
    result_df = pd.concat(sig_df)
    mean_std_df = pd.DataFrame([result_df.mean(), result_df.std()], index=['Mean','Std'])
    mean_std_df.to_csv(os.path.join(output_path, 
                                    "{}-{}_AUROC_results.csv".format(iter_type[1][0], iter_type[1][1])))


if __name__ == "__main__":
    output_path = os.path.join(os.getcwd(), 'Sig_KS_results')
    auroc_path = os.path.join(output_path, 'AUROC_results')

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    if not os.path.exists(auroc_path):
        os.mkdir(auroc_path)
        
    tr_dic = get_data_dic('tr')
    te_dic = get_data_dic('te')
    keys = set(list(tr_dic.keys()))

    colname = ['velocity', 'angle', 'gyro', 
            'velocity_angle', 'velocity_gyro', 'angle_gyro', 
            'velocity_angle_gyro']
    # 0 : velocity, 1 : angle, 2 3 4 : gyro
    index = [[0], [1], [2,3,4],
            [0,1], [0,2,3,4], [1,2,3,4], 
            [0,1,2,3,4]]

    args = []
    for i in range(26):
        tmp_type =  sig_type = list(tr_dic.keys())[50*i:50*(i+1)]
        tmp_tr = list(tr_dic.values())[50*i:50*(i+1)]
        tmp_te = list(te_dic.values())[50*i:50*(i+1)]
        args.append([tmp_type, tmp_tr, tmp_te, i, colname, index])

    pool = mp.Pool(os.cpu_count()-2)
    pool.map(pattern_main, args)
