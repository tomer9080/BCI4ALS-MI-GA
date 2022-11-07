import os 

#check priors without UNIFY
for i in range(0,6):
    os.system(f'python MI5_ModelTraining.py -pa paths\\paths_RL.txt -pr {i} -n TK_rec_pr{i}')

#check priors with UNIFY(for a single date)
for i in range(0,6):
    os.system(f'python MI5_ModelTraining.py -pa paths\\paths_RL.txt -u True -pr {i} -n TK_rec_pr{i}')

#need to add option to unify all recordings then run on all of them with different priors