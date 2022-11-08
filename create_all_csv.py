import os 

# os.system(f'python MI5_ModelTraining.py -pa paths\\paths_RL.txt -pr 1 -n TK_rec_pr1')
# #check priors without UNIFY
for i in range(0,7):
    os.system(f'python MI5_ModelTraining.py -pa paths\\paths_RL.txt -pr {i} -n TK_rec_pr{i}')


# #check priors with UNIFY(for a single date)
for i in range(0,7):
    os.system(f'python MI5_ModelTraining.py -pa paths\\paths_RL.txt -u True -pr {i} -n TK_rec_pr_unified{i}')

# #need to add option to unify all recordings then run on all of them with different priors

#TK_rootdir
RL_rootdir = "C:\\Users\\Latzres\\Desktop\\project\\BCI-Matlab-Code\\class_results"
os.system(f'python analyze_csv.py {RL_rootdir}')