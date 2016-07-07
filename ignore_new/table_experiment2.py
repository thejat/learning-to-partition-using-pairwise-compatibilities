print graphtype
for k in all_result_log.keys():
	print "k:",k
	print "No Estimated: ",1.0*sum([all_result_log[k][run_number]['no_estimated_weights'] for run_number in all_result_log[k].keys()])/len(all_result_log[k].keys())
	print "No Observable: ",1.0*sum([all_result_log[k][run_number]['no_observable_weights'] for run_number in all_result_log[k].keys()])/len(all_result_log[k].keys())
	print "Total weights: ",1.0*sum([all_result_log[k][run_number]['total_weights'] for run_number in all_result_log[k].keys()])/len(all_result_log[k].keys())
	print "Success: ",1.0*sum([all_result_log[k][run_number]['success'] for run_number in all_result_log[k].keys()])/len(all_result_log[k].keys())

import pickle
pickle.dump(all_result_log,open(temp_dir+'/learning_weights_'+graphtype+'_20151122_all_result_log.pickle','wb'))

'''
random
k: 8
No Estimated:  7.96666666667
No Observable:  7.96666666667
Total weights:  28.0
Success:  1.0
k: 16
No Estimated:  36.5666666667
No Observable:  36.5666666667
Total weights:  120.0
Success:  1.0
k: 64
No Estimated:  604.166666667
No Observable:  604.166666667
Total weights:  2016.0
Success:  1.0
k: 32
No Estimated:  149.633333333
No Observable:  149.633333333
Total weights:  496.0
Success:  1.0
--------------------------------------

facebook
k: 8
No Estimated:  23.4666666667
No Observable:  23.4666666667
Total weights:  28.0
Success:  1.0
k: 16
No Estimated:  107.233333333
No Observable:  111.966666667
Total weights:  120.0
Success:  0.957763417747
k: 64
No Estimated:  1673.83333333
No Observable:  1979.93333333
Total weights:  2016.0
Success:  0.845428720697
k: 32
No Estimated:  436.0
No Observable:  479.6
Total weights:  496.0
Success:  0.909122104449


-------------------------------------

collab
k: 8
No Estimated:  25.3666666667
No Observable:  25.3666666667
Total weights:  28.0
Success:  1.0
k: 16
No Estimated:  109.266666667
No Observable:  111.266666667
Total weights:  120.0
Success:  0.982144696646
k: 64
No Estimated:  1808.0
No Observable:  1990.63333333
Total weights:  2016.0
Success:  0.908257831777
k: 32
No Estimated:  451.1
No Observable:  479.566666667
Total weights:  496.0
Success:  0.940678858444

'''