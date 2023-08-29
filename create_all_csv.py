import os 

# table_headers = ['Feature',
#                 'Score_(R^2)_Left', 'Score_(R^2)_Right',
#                 'Score_(LASSO)_Left', 'Score_(LASSO)_Right',
#                 'Score_(R^1)_Left', 'Score_(R^1)_Right',
#                 'R1_Left', 'R1_Right',
#                 'R2_Left', 'R2_Right',
#                 'Coef_Left', 'Coef_Right',
#                 'explained_variance_left', 'explained_variance_right',
#                 'max_error_left', 'max_error_right',
#                 'mean_absolute_error_left', 'mean_absolute_error_right',
#                 'mean_squared_error_left', 'mean_squared_error_right',
#                 'median_absolute_error_left', 'median_absolute_error_right',
#                 'mean_absolute_percentage_error_left', 'mean_absolute_percentage_error_right',
#                 'mean_pinball_loss_left', 'mean_pinball_loss_right',
#                 'd2_pinball_score_left', 'd2_pinball_score_right',
#                 'd2_absolute_error_score_left', 'd2_absolute_error_score_right'
#                 ]
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

# tuples_list = [(table_headers[i], table_headers[i + 1]) for i in range(1, len(table_headers), 2)]
# for l, r in tuples_list:
#     ascending = True
#     if 'explained' in l or 'd2' in l:
#         ascending = False
#     os.system(f'python MI5_ModelTraining.py -pa paths\\paths_RL.txt -pr 4 -n TK_rec_pr4_{l.replace("_left", "")} -m {l},{r} -a {ascending}')

# Run many GAs to get new data on fine tuned GAs.
for i in range(35):
    os.system(f"python MI5_ModelTraining.py -pa paths/paths_linux.txt -ga True -th 100 -n ga_blitz_thresh_100_{i}")

for i in range(35):
    os.system(f"python MI5_ModelTraining.py -pa paths/paths_linux.txt -ga True -th 50 -n ga_blitz_thresh_50_{i}")


# from OurUtils import get_paths
# paths = get_paths('paths/paths_linux.txt')
# for path in paths:
#     os.system(f'python OurOptuna.py {path}')

# #need to add option to unify all recordings then run on all of them with different priors
