import numpy as np
import pandas as pd
import sys
from sklearn.metrics import roc_auc_score


def scale_timestamp(data, scale=1e+7):
    for i in range(len(data)):
        tmp = data[i]
        tmp['timeInterval'] = tmp['timeInterval'].apply(lambda x: x/scale)
    return data

def velocity(p1,p2, time):
    p1, p2 = p1.values, p2.values
    return np.linalg.norm(p2-p1)/time #* int(10**8)

def gyro_velocity(g1, g2, time):
    return np.abs((g1-g2)/time) #* int(10**10)

def angle(p1, p2, time):
    p1, p2 = p1.values, p2.values
    ang1 = np.arctan2(*p1[::-1])
    ang2 = np.arctan2(*p2[::-1])
    ans = np.rad2deg((ang1 - ang2) % (2 * np.pi))
    if 180 <= ans < 360:
        return (360 - ans) / time
    else:
        return ans / time


def printProgress (iteration, total, prefix = '', suffix = '', decimals = 1, barLength = 100):
    formatStr = "{0:." + str(decimals) + "f}"
    percent = formatStr.format(100 * (iteration / float(total)))
    filledLength = int(round(barLength * iteration / float(total)))
    bar = '#' * filledLength + '-' * (barLength - filledLength)
    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percent, '%', suffix)),
    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()
    
def get_data_dic(partition_type):
    path = '%s-dataset%02d' % (partition_type, 0)
    dataset = pd.read_pickle('./data/datasets/sig/'+path)

    datas = dataset['data']
    datas = scale_timestamp(datas)
    targets = dataset['target']
    notes = dataset['note']

    data_dic = {(target, note): []
                for note in notes for target in targets}

    for i, df in enumerate(datas):
        # df = diff_dataframe(df)
        data_dic[targets[i], notes[i]].append(df)
        printProgress(i, len(datas), 'Load {} data :'.format(partition_type), '', 1, 50)
    print("")
    return data_dic


def diff_dataframe(table):
    stroke_idx = np.where((table['action.0']=='DOWN'))[0]
    stroke_idx = np.append(stroke_idx, len(table)-1)
    stroke_num = int(len(stroke_idx))
    stroke_list = []
    for i in range(stroke_num-1):
        tmp = table.iloc[stroke_idx[i]:stroke_idx[i+1]]
        df_tmp = pd.DataFrame()
        
        # move -> up만 가져갈 것인가? up -> down도 가져갈 것인가?
        df_tmp['axis_velocity'] = tmp.apply(lambda x: velocity(x[['x.0','y.0']],
                                                               x[['x.1','y.1']],
                                                               x['timeInterval']), axis=1)
        ## 자이로 넣자 ##
        df_tmp['angle'] = tmp.apply(lambda x: angle(x[['x.0','y.0']],
                                                    x[['x.1','y.1']],
                                                    x['timeInterval']), axis=1)
        
        df_tmp['x'] = tmp.apply(lambda x: gyro_velocity(x['gx.0'],
                                                        x['gx.1'],
                                                        x['timeInterval']), axis=1)                                    
        df_tmp['y'] = tmp.apply(lambda x: gyro_velocity(x['gy.0'],
                                                        x['gy.1'],
                                                        x['timeInterval']), axis=1)
        df_tmp['z'] = tmp.apply(lambda x: gyro_velocity(x['gz.0'],
                                                        x['gz.1'],
                                                        x['timeInterval']), axis=1)
        df_tmp = df_tmp[~(df_tmp == 0).any(axis=1)]
        stroke_list.append(df_tmp)
        #         stroke_dict[str(i+1)] = df_tmp#[:-1]
    return stroke_list


# KS 계산기. 얘를 수정해야할듯(3차원 : 속도, angle, 자이로)
def KS_calculator(a,b, unit=1):
    b = diff_dataframe(b) # test 데이터는 아직 diff 안했음!
    KS_dict = {}

    def KS_function(a, b, unit=unit):
        X_axis = np.arange(np.min(np.append(a, b)), np.max(np.append(a, b)), unit)
        tmp = np.array([])
        
        for X in X_axis:
            A_Y_Axis = len(np.where(a <= X)[0]) / len(a)
            B_Y_Axis = len(np.where(b <= X)[0]) / len(b)
            tmp = np.append(tmp, np.absolute(A_Y_Axis - B_Y_Axis))
        KS = np.max(tmp) 
        return KS

    for col in a[0].columns:
        if col == 'axis_velocity':
            unit = 0.1 
        elif col == 'angle':
            unit = 0.01
        else:
            unit = 0.001
        KS_dict[col] = np.mean([KS_function(a[i][col], b[i][col], unit=unit) for i in range(len(a))])

    df = pd.DataFrame(data=KS_dict, index=[0])
    return df

def final_df(valid_df, test_df, index, colname):
    def transform_df(df, valid=True):
        df =pd.concat([df.iloc[:,idx].mean(axis=1) for idx in index], axis=1)
        df = df.reset_index(drop=True)
        df.columns = colname

        if valid:
            target_idx = pd.DataFrame({"Target":np.ones(len(df))})
            return pd.concat([df, target_idx], axis=1)
        else:
            target_idx = pd.DataFrame({"Target":np.zeros(len(df))})
            return pd.concat([df, target_idx], axis=1)

    final_test_df = transform_df(test_df, valid=False)
    final_valid_df = transform_df(valid_df)

    df = pd.concat([final_valid_df, final_test_df], axis=0)
    return df

# EER 구하기
def get_EER(ks_df):
    se_target = ks_df['Target']
    columns = ks_df.columns[:-1]
    results = []
    for col in columns:
        se = ks_df[col]
        cut_offs = sorted(set(se), reverse=True)
        DIFF, FAR, FRR = [], [], []
        for cut_off in cut_offs[:-1]:
            conds = (se >= cut_off)
            Predict_All_True_Upper = se_target[conds]
            False_Acceptance_Rate = (sum(Predict_All_True_Upper) / len(Predict_All_True_Upper))
            Predict_All_False_Under = se_target[~conds]
            False_Rejection_Rate = (sum(Predict_All_False_Under == 0) / len(Predict_All_False_Under))
            DIFF.append(abs(False_Acceptance_Rate-False_Rejection_Rate))
            FAR.append(False_Acceptance_Rate)
            FRR.append(False_Rejection_Rate)
            #print(col, cut_off)
            #print(Predict_All_False_Under, Predict_All_True_Upper)
        DIFF = np.array(DIFF)
        FAR8FRR = np.array([FAR, FRR]).T
        result = FAR8FRR[DIFF == DIFF.min()].sum()/2
        results.append(result)
    tmp_df = pd.DataFrame([results], columns=columns)
    return tmp_df

# AUROC 구하기
def get_auroc(df):
    targets = df['Target'].values
    scores = []
    for col in df.columns[:-1]:
        vals = np.array(df[col])
        score = roc_auc_score(targets, vals)
        score = abs(score-0.5)+0.5
        scores.append(score)
    df = pd.DataFrame([scores], columns=df.columns[:-1])
    return df


# 획 나눠서 dict로 집어넣기
# def diff_dataframe(table):
#     stroke_idx = np.where((table['action'].values== 'DOWN') | (table['action'].values=='UP'))[0]
#     stroke_num = int(len(stroke_idx)/2)
#     stroke_dict = {}
#     for i in range(stroke_num):
#         try:
#             tmp = table.iloc[stroke_idx[i*2:i*2+2][0]:stroke_idx[i*2:i*2+2][1]]
#             ind = np.ones((len(tmp.columns)), bool)
#             ind[[0,2]] = False
#             tmp2 = tmp.iloc[:,ind].diff().dropna()
#             tmp3 = pd.DataFrame()
#             tmp3['axis_velocity'] = [velocity(tmp.iloc[i,3:5], tmp.iloc[i-1,3:5], tmp2['timestamp'].iloc[i-1]) 
#                             for i in range(1,len(tmp))]
#             tmp3['X']= tmp2.apply(lambda x: gyro_velocity(x['gx'], x['timestamp']),axis=1).values
#             tmp3['Y']= tmp2.apply(lambda x: gyro_velocity(x['gy'], x['timestamp']),axis=1).values
#             tmp3['Z']= tmp2.apply(lambda x: gyro_velocity(x['gz'], x['timestamp']),axis=1).values
#             tmp3['angle'] =[angle(tmp.iloc[i,3:5],tmp.iloc[i-1,3:5], tmp2['timestamp'].iloc[i-1]) for i in range(1,len(tmp))]
#             stroke_dict[str(i+1)] = tmp3
#         except:
#             continue

#     return stroke_dict
