import argparse
import os
import json
import numpy as np
from pathlib import Path
from latent_classification import latent_classifier
from utils import set_seed, read_json
from parse_config import ConfigParser 

set_seed()

# tasks = ['family', 'instrument', 'pitch']
tasks = ['instrument', 'pitch']

def eval_latent_classifier(config):
    # scores = {k: 0 for k in ['pitch', 'instrument', 'family', 'dynamic']} 
    scores = {k: 0 for k in tasks} 
    for target in tasks:
        scores[target] = latent_classifier(config, target)

    return scores


def batch_eval_latent_classifier(model_folder, w_config, save_path):
    assert len(w_config.split('-')) == 6
    print("Weight config: %s" % w_config)
    # avg_scores = {k: 0 for k in ['pitch', 'instrument', 'family', 'dynamic']}
    scores = {k: [] for k in tasks}
    avg_scores = {k: 0 for k in tasks}
    model_count = 0
    eval_models = [f for f in Path(model_folder).glob('*') if w_config in str(f)]

    txt_file_name = 'batch_eval_lc_%s.txt' % w_config
    txt_dir = os.path.join(save_path, txt_file_name)
    txt_file = open(txt_dir, 'w')

    for f in eval_models:
        # for d in f.glob('*'):
        #     m = d / 'model_best.pth'
        #     if m.is_file():
        #         f = d; txt_file.write("%s\n" % f)
        #         break
        txt_file.write('%s\n' % f)
        print("Processing model: %s" % f)
        config_file = read_json(str(f / 'config.json'))
        model_best = str(f / 'model_best.pth')
        config = ConfigParser(config_file, resume=model_best, testing=True)
        score = eval_latent_classifier(config)
        for k in scores.keys():
            scores[k].append(100 * score[k])
            avg_scores[k] += 100 * score[k]
        model_count += 1
    for k in scores.keys():
        avg_scores[k] /= model_count
 

    combine = (np.array(scores['instrument']) + (100 - np.array(scores['pitch']))) / 2
    scores.update({'combine': list(combine)})
    avg_scores.update({'combine': np.mean(combine)})

    txt_file.write(json.dumps(scores, indent=2))
    txt_file.write('\n')
    txt_file.write(json.dumps(avg_scores, indent=2))
    txt_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_folder', type=str)
    parser.add_argument('-w', '--w_config', type=str)
    parser.add_argument('-s', '--save_path', type=str)
    args = parser.parse_args()
    batch_eval_latent_classifier(args.model_folder, args.w_config, args.save_path)
