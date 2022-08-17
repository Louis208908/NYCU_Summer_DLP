#########################
## KL Anneal: Cyclical ##
#########################
python ./NYCU_Summer_CLP/lab4/train_fixed_prior.py --cuda --train --batch_size 20 --tfr_decay_step 0.01 --tfr_start_decay_epoch 150 --kl_anneal_cyclical 

#python ./NYCU_Summer_CLP/lab4/train_fixed_prior.py \
#	--cuda \
#	--test \
#	--test_set test \
#	--batch_size 20 \
#	--kl_anneal_cyclical \
#	--exp_name cyclical \
#	--model_dir ../logs/fp/cyclical

##########################
## KL Anneal: Monotonic ##
##########################
#python ./NYCU_Summer_CLP/lab4/train_fixed_prior.py \
#	--cuda \
#	--train \
#	--tfr_decay_step 0.01 \


#python main.py \
#	--cuda \
#	--test \
#	--test_set test \
#	--batch_size 20 \
#	--exp_name monotonic \
#	--model_dir ../logs/fp/monotonic

##############
## Plotting ##
##############
python others/utils.py


python -u ./NYCU_Summer_DLP/lab4/src/train_fixed_prior.py --train --batch_size 20 --tfr_decay_step 0.01 --tfr_start_decay_epoch 50 --lr 0.001