from scipy.stats import friedmanchisquare
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import Orange

def FriedTest(measure, measure_name):

    try:
        # stat, p = friedmanchisquare(measure['RaKEL'], measure['RaKEL-d'], measure['LP'], measure['ML-kNN'], measure['BR'],
        #                             measure['PS'], measure['EPS'], measure['HOMER'], measure['CC'], measure['ECC'],
        #                             measure['MLR-max'], measure['MLR-agg'])
        stat, p = friedmanchisquare(measure['Alpha-Investing'], measure['SAOLA'], measure['Fast-OSFS'],
                                    measure['OFS-Density'], measure['OFS_A3M'], measure['Group SAOLA'],
                                    measure['OGFSS-FI'], measure['OFS-DFC'])
    except KeyError:
        # stat, p = friedmanchisquare(measure['RaKEL'], measure['RaKEL-d'], measure['LP'], measure['ML-kNN'], measure['BR'],
        #                             measure['PS'], measure['EPS'], measure['HOMER'], measure['CC'], measure['ECC'],
        #                             measure['MLR-agg'])
        stat, p = friedmanchisquare(measure['Alpha-Investing'], measure['SAOLA'], measure['Fast-OSFS'],
                                    measure['OFS-Density'], measure['OFS_A3M'], measure['Group SAOLA'],
                                    measure['OGFSS-FI'], measure['OFS-DFC'])
    print('Statistics=%.3f, p=%.3f' % (stat, p))
    # interpret
    alpha = 0.05
    if p > alpha:
        print(measure_name + ' measure: Same distributions (fail to reject H0)')
    else:
        print(measure_name + ' measure: Different distributions (reject H0)')
# Read results from KNN.csv and DT.csv files
results1 = pd.read_csv('KNN.csv')
results2 = pd.read_csv('DT.csv')
cols = results1.columns
rows = results1['Datasets'].dropna()
datasets = rows.values
methods = cols[2:].values

knn_res = results1[methods].values
acc1 = []
f1_1 = []

dt_res = results2[methods].values
acc2 = []
f1_2 = []

n_groups = round(np.shape(knn_res)[0])

for i in range(0, n_groups, 2):
    f1_1.append(knn_res[i, :])
    f1_2.append(dt_res[i, :])
    acc1.append(knn_res[i+1, :])
    acc2.append(dt_res[i + 1, :])

rank_acc1, rank_acc2 = [], []
rank_f11, rank_f12 = [], []
for j in range(len(acc1)):
    sort_idx1_1, sort_idx1_2 = np.argsort(-acc1[j]), np.argsort(-acc2[j])
    rank1_1, rank1_2 = [1] * len(methods), [1] * len(methods)
    oid1_1, oid1_2 = 1, 1
    for k1 in sort_idx1_1:
        rank1_1[k1] = oid1_1
        oid1_1 += 1
    for k2 in sort_idx1_2:
        rank1_2[k2] = oid1_2
        oid1_2 += 1

    sort_idx2_1, sort_idx2_2 = np.argsort(-f1_1[j]), np.argsort(-f1_2[j])
    rank2_1, rank2_2 = [1] * len(methods), [1] * len(methods)
    oid2_1, oid2_2 = 1, 1
    for k3 in sort_idx2_1:
        rank2_1[k3] = oid2_1
        oid2_1 += 1
    for k4 in sort_idx2_2:
        rank2_2[k4] = oid2_2
        oid2_2 += 1
    rank_acc1.append(rank1_1)
    rank_acc2.append(rank1_2)
    rank_f11.append(rank2_1)
    rank_f12.append(rank2_2)

rank_acc1 = np.reshape(rank_acc1, (len(acc1), len(methods)))
rank_f11 = np.reshape(rank_f11, (len(f1_1), len(methods)))

rank_acc2 = np.reshape(rank_acc2, (len(acc2), len(methods)))
rank_f12 = np.reshape(rank_f12, (len(f1_2), len(methods)))

measure1 = pd.DataFrame(data=rank_acc1, index=datasets, columns=methods)
avranks1 = measure1.mean().values
FriedTest(measure1, 'Accuracy')
print("Mean ranks of F1-macro on DT classifier:", avranks1)
cd1 = Orange.evaluation.compute_CD(avranks1, len(datasets), test='nemenyi', alpha='0.05')
Orange.evaluation.graph_ranks(avranks1, methods, cd=cd1, width=6, textspace=1.5)
plt.savefig('CD-accuracy-KNN.png')


measure2 = pd.DataFrame(data=rank_f11, index=datasets, columns=methods)
avranks2 = measure2.mean().values
FriedTest(measure2, 'F1-macro')
print("Mean ranks of F1-macro on KNN classifier:", avranks2)
cd2 = Orange.evaluation.compute_CD(avranks2, len(datasets), test='nemenyi', alpha='0.05')
Orange.evaluation.graph_ranks(avranks2, methods, cd=cd2, width=6, textspace=1.5)
plt.savefig('CD-F1-KNN.png')
    

measure3 = pd.DataFrame(data=rank_acc2, index=datasets, columns=methods)
avranks3 = measure3.mean().values
FriedTest(measure3, 'Accuracy')
print("Mean ranks of F1-macro on DT classifier:", avranks3)
cd3 = Orange.evaluation.compute_CD(avranks3, len(datasets), test='nemenyi', alpha='0.05')
Orange.evaluation.graph_ranks(avranks3, methods, cd=cd1, width=6, textspace=1.5)
plt.savefig('CD-accuracy-DT.png')


measure4 = pd.DataFrame(data=rank_f12, index=datasets, columns=methods)
avranks4 = measure4.mean().values
FriedTest(measure4, 'F1-macro')
print("Mean ranks of F1-macro on KNN classifier:", avranks4)
cd4 = Orange.evaluation.compute_CD(avranks4, len(datasets), test='nemenyi', alpha='0.05')
Orange.evaluation.graph_ranks(avranks2, methods, cd=cd2, width=6, textspace=1.5)
plt.savefig('CD-F1-DT.png')
