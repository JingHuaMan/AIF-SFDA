import os
import pickle
import random
import shutil

import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

from data.random_point_prompt_dataset import RandomPointPromptDataset


class DummyOPT:
    def __init__(self):
        self.load_size = 512
        self.no_mask = False
        self.preprocess = []
        # self.positive_num = 20  # vessel
        # self.negative_num = 20  # vessel
        self.positive_num = 5  # od
        self.negative_num = 5  # od
        self.data_dirname = '/home/huyan/liheng/data6/lihaojin/data/retinal_vessel/drive/crop_train'


if __name__ == '__main__':
    source_dir = '/home/huyan/liheng/data6/lihaojin/data/retinal_vessel/'
    dataset_list = ['avrdb', 'chasedb1', 'drhagis', 'drive', 'hrf', 'iostar', 'lesav', 'stare']
    # source_dir = '/home/huyan/liheng/data6/lihaojin/data/OCOD_segmentation/'
    # dataset_list = ['riga', 'iostar', 'adam']

    opt = DummyOPT()
    sub_list = ['crop_test', 'crop_train']
    for dataset_name in tqdm(dataset_list):
        for sub in sub_list:
            opt.data_dirname = os.path.join(source_dir, dataset_name, sub)
            dataset = RandomPointPromptDataset(opt)
            for i in range(len(dataset)):
                data = dataset[i]
                prompt = data['prompt']
                label = data['label']

                # visualize both prompt and label
                plt.imshow(label.squeeze().numpy(), cmap='gray')
                for p, p_type in zip(prompt[0], prompt[1]):
                    plt.scatter(p[0], p[1], c='r' if p_type else 'b', s=5)
                plt.show()

                if i == '3':
                    exit()



                # path = os.path.join('/'.join(data['source_path'].split('/')[:-1]), 'prompts.pickle')
                # if os.path.exists(path):
                #     os.remove(path)
                # with open(path, 'wb') as f:
                #     pickle.dump((512, prompt[0], prompt[1]), f)
