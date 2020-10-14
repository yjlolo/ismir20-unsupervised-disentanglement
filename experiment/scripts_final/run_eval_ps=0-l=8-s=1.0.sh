# evaluate timbre space with FID
# for both reconstructed validation set and random sampling conditioned on inferred pitch 
CUDA_VISIBLE_DEVICES=1 python experiment/rand_sample.py -m saved_final/models/0514-final_exp/ps=0-su=1.0-l=8-10-1-0-0-0-0-bf=True/ -w 10-1-0-0-0-0 -s experiment/score_final/ps=0-su=1.0-l=8-10-1-0-0-0-0-bf=True/ -c saved_final/models/0514-final_exp-clfr_instrument/

# consistency and diversity for pitch conditional generation
CUDA_VISIBLE_DEVICES=1 python experiment/cond_gen.py -m saved_final/models/0514-final_exp/ps=0-su=1.0-l=8-10-1-0-0-0-0-bf=True/ -w 10-1-0-0-0-0 -s experiment/score_final/ps=0-su=1.0-l=8-10-1-0-0-0-0-bf=True/ -c saved_final/models/0514-final_exp-clfr_pitch/

# metric during training 
CUDA_VISIBLE_DEVICES=1 python experiment/eval_metrics.py -m saved_final/models/0514-final_exp/ps=0-su=1.0-l=8-10-1-0-0-0-0-bf=True/ -w 10-1-0-0-0-0 -s experiment/score_final/ps=0-su=1.0-l=8-10-1-0-0-0-0-bf=True/
