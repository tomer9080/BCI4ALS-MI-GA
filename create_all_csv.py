import os 
from metrics_wrapper import table_headers
# check priors without UNIFY
# for i in range(0,7):
#     os.system(f'python MI5_ModelTraining.py -pa paths\\paths_TK.txt -pr {i} -n TK_rec_pr{i}')


# check priors with UNIFY(for a single date)
# for i in range(0,7):
#     os.system(f'python MI5_ModelTraining.py -pa paths\\paths_TK.txt -u True -pr {i} -n TK_rec_pr_unified{i}')

# for i in range(0,7):
#     os.system(f'python MI5_ModelTraining.py -pa paths\\paths_RL.txt -pr {i} -n TK_rec_pr{i}_r2_score')
#     os.system(f'python MI5_ModelTraining.py -pa paths\\paths_RL.txt -pr {i} -n TK_rec_pr{i}_coef -m Coef_Left,Coef_Right -a True') # ascending
#     os.system(f'python MI5_ModelTraining.py -pa paths\\paths_RL.txt -pr {i} -n TK_rec_pr{i}_lasso -m Score_(LASSO)_Left,Score_(LASSO)_Right')
#     os.system(f'python MI5_ModelTraining.py -pa paths\\paths_RL.txt -pr {i} -n TK_rec_pr{i}_r2_dist -m R2_Left,R2_Right -a True') # ascending
#     os.system(f'python MI5_ModelTraining.py -pa paths\\paths_RL.txt -pr {i} -n TK_rec_pr{i}_r1_dist -m R1_Left,R1_Right -a True') # ascending

tuples_list = [(table_headers[i], table_headers[i + 1]) for i in range(1, len(table_headers), 2)]
for l, r in tuples_list:
    ascending = True
    if 'explained' in l or 'd2' in l:
        ascending = False
    os.system(f'python MI5_ModelTraining.py -pa paths\\paths_RL.txt -pr 4 -n TK_rec_pr4_{l.replace("_left", "")} -m {l},{r} -a {ascending}')

# #need to add option to unify all recordings then run on all of them with different priors

# TK_rootdir = "C:\\BCIToolBox\\MatlabCode\\class_results"
RL_rootdir = "C:\\Users\\Latzres\\Desktop\\project\\BCI-Matlab-Code\\class_results"
os.system(f'python analyze_csv.py {RL_rootdir}')