import os
import glob
import json
from tqdm import tqdm
import numpy as np

file_name = sorted(glob.glob('./mlxtend/blur/label/gen/*.json'))
file_name = ['_'.join(fi.split('/')[-1].split('_')[:-1]) for fi in file_name]

gt_path = './ground_truth/'
ml_path = './mlxtend/'
dlib_path = './dlib'

degrades = ['blur', 'noise', 'down']
statuses = ['gen', 'low']

non_detect = {'mlxtend' : {'blurgen':[], 'blurlow':[], 'noisegen':[], 'noiselow':[], 'downgen':[], 'downlow':[]},
                'dlib' : {'blurgen':[], 'blurlow':[], 'noisegen':[], 'noiselow':[], 'downgen':[], 'downlow':[]}}

mean_error = {'mlxtend' : {'blurgen':[], 'blurlow':[], 'noisegen':[], 'noiselow':[], 'downgen':[], 'downlow':[]},
                'dlib' : {'blurgen':[], 'blurlow':[], 'noisegen':[], 'noiselow':[], 'downgen':[], 'downlow':[]}}

for model in ['mlxtend', 'dlib']:
    for file_path in tqdm(file_name):

        try:
            with open(gt_path + file_path + '_label.json', 'r') as f:
                gtf = json.load(f)
            gt = []

            for i in range(9):
                gt.extend(gtf['figures'][i]['shape']['coordinates'])
        except:
            print(f'error in {file_path}  !!')
            error_file['gt'].append(file_path)

        for de in degrades:
            for status in statuses:
                try:
                    with open('./' + model + f'/{de}/label/{status}/{file_path}_crop.json', 'r') as f:
                        pred = json.load(f)

                    if model == 'mlxtend':
                        if pred['landmarks'][0] == [0, 0]:
                            continue
                        pred = np.array(pred['landmarks'])

                    else:
                        pred = np.array(pred['landmark'])
                    gt = np.array(gt)
                    mean_error[model][de+status].append((((gt - pred)**2).sum(axis = 1)**(1/2)).mean())
                except:
                    non_detect[model][de+status].append(file_path)

with open('./RMSE.json', 'w') as f:
    json.dump(mean_error, f)
with open('./non_detect.json', 'w') as f:
    json.dump(non_detect, f)