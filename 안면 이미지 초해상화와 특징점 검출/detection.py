import imageio
import matplotlib.pyplot as plt
from mlxtend.image import extract_face_landmarks

import glob
import os
import json
import numpy as np

import dlib
import json
import cv2
import glob
import os

files = glob.glob('./results/noise/test/*.*')
files = [file.split('/')[-1] for file in files]
degrades = ['blur', 'noise', 'down']

############ mlxtend ############

#from _typeshed import NoneType
# gen_nomark = 0
# low_nomakr = 0
for degrade in degrades:
  
  for file in files:
    file = f'./results/{degrade}/test/{file}'
    img = imageio.imread(file)
    low = img[:, :512]
    gen = img[: ,512:1024]

    file_name = file.split(".")[0]

    low_mark = extract_face_landmarks(low)
    gen_mark = extract_face_landmarks(gen)

    if type(gen_mark) != 'NoneType':
      for p in gen_mark:
        p = list(map(int, p))
        gen[p[1]-3:p[1]+3, p[0]-3:p[0]+3, :] = (0, 0, 255)
      
      #image with landmarks save
      imageio.imwrite(f'./mlxtend/{degrade}/img/gen/{file_name}.jpg', gen)
      
      #landmarks save
      marks = [list(map(int, i)) for i in gen_mark]
      land = {"landmarks" : marks}
      with open(f'./mlxtend/{degrade}/label/gen/{file_name}.json', 'w') as f:
        json.dump(land, f, indent=2)


    if type(low_mark) != 'NoneType':
      for p in low_mark:
        p = list(map(int, p))
        low[p[1]-3:p[1]+3, p[0]-3:p[0]+3, :] = (0, 0, 255)
      
      #image with landmarks save
      imageio.imwrite(f'/mlxtend/{degrade}/img/low/{file_name}.jpg', low)
      
      #landmarks save
      marks = [list(map(int, i)) for i in low_mark]
      land = {"landmarks" : marks}
      with open(f'./mlxtend/{degrade}/label/low/{file_name}.json', 'w') as f:
        json.dump(land, f, indent=2)

############ dlib ############

RIGHT_EYE = list(range(36, 42))
LEFT_EYE = list(range(42, 48))
MOUTH = list(range(48, 68))
NOSE = list(range(27, 36))
EYEBROWS = list(range(17, 27))
JAWLINE = list(range(1, 17))
ALL = list(range(0, 68))
EYES = list(range(36, 48))

predictor_file = '/content/shape_predictor_68_face_landmarks.dat'

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_file)

for degrade in ['blur', 'noise', 'down']:

  for file in files:
    file = f'./results/{degrade}/test/{file}'
    file_name = file.split(".")[0]

    image = cv2.imread(file)
    low = image[:, :512]
    gen = image[: ,512:1024]
  
    gray_low = cv2.cvtColor(low, cv2.COLOR_BGR2GRAY)
    gray_gen = cv2.cvtColor(gen, cv2.COLOR_BGR2GRAY)
    rects_low = detector(gray_low, 1)
    rects_gen = detector(gray_gen, 1)


  #low detection
    for (i, rect) in enumerate(rects_low):
        points_low = np.matrix([[p.x, p.y] for p in predictor(gray_low, rect).parts()])
        show_parts_low = points_low[ALL]
        print(show_parts_low)
        if len(rects_low) != 0:
          landmark_low = {"landmark" : [np.array(i).reshape(-1, ).tolist() for i in points_low]}
          with open(f'./dlib/{degrade}/label/low/{file_name}.json', 'w') as f:
            json.dump(landmark_low, f, indent=2)

        for (i, point) in enumerate(show_parts_low):
            x = point[0,0]
            y = point[0,1]
            cv2.circle(low, (x, y), 1, (0, 255, 255), -1)
    cv2.imwrite(f'./dlib/{degrade}/img/low/{file_name}.jpg', low)

    with open(f'./dlib/{degrade}/label/low/{file_name}.json', 'w') as f:
        json.dump(land, f, indent=2)

        #     cv2.putText(image, "{}".format(i + 1), (x, y - 2),
        # cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)



    for (i, rect) in enumerate(rects_gen):
        points_gen = np.matrix([[p.x, p.y] for p in predictor(gray_gen, rect).parts()])
        show_parts_gen = points_gen[ALL]
        print(show_parts_gen)
        if len(rects_gen) != 0:
          landmark_gen = {"landmark" : [np.array(i).reshape(-1, ).tolist() for i in points_gen]}
          with open(f'./dlib/{degrade}/label/gen/{file_name}.json', 'w') as f:
            json.dump(landmark_gen, f, indent=2)

        for (i, point) in enumerate(show_parts_gen):
            x = point[0,0]
            y = point[0,1]
            cv2.circle(gen, (x, y), 1, (0, 255, 255), -1)

    cv2.imwrite(f'./dlib/{degrade}/img/gen/{file_name}.jpg', gen)

    # cv2.imshow(gen)
    # cv2.imshow(low)