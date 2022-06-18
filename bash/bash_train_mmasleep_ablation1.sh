start=0
end=19
for i in $(eval echo {$start..$end})
do
   python train_Kfold_mmasleep_MBFE.py --fold_id=$i --ch=4 --np_data_dir data_path/sleepedf-20   
   python train_Kfold_mmasleep_MBFE_TRANS.py --fold_id=$i --ch=4 --np_data_dir data_path/sleepedf-20   
done
