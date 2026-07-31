import torch
import os

import pandas as pd

from torch.utils.data import Dataset
from torchvision import transforms, io

current_dir = os.path.dirname(os.path.abspath(__file__))


class CardArithmetic(Dataset):

    def __init__(self, image_dir, variant='sum_2', train_val_test='train'):
        """
        Initialize Card arithmetic dataset by loading image and task data.
        :param image_dir: String of directory that contains train/val/test folders with image data
        :param variant: String denoting the variant of card arithmetic
        :param train_val_test: String indicating whether to load the train, validation or test data
        """
        if train_val_test == 'val':
            # Val data uses train images
            self.image_dir = f'{image_dir}/images_train'
        else:
            self.image_dir = f'{image_dir}/images_{train_val_test}'

        self.data = pd.read_csv(current_dir + f"/tasks/card_{variant}_{train_val_test}.csv")
        # Set latent_labels variable
        self.create_latent_dataset()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
        Get image and labels.
        :param index: Index of the example in the task dataset
        :return: Dictionary containing the input images, their latent labels and the downstream label
        """
        imgs = []
        img_idxs = self.data.iloc[index]
        latent_labels = self.latent_labels.iloc[index].tolist()
        for col_name, value in img_idxs.items():
            if col_name.startswith('card'):
                # Load image corresponding to card index
                img = io.read_image(self.image_dir + f"/{value}.jpg").float() / 255.0
                imgs.append(img)
            if col_name == 'result':
                label = value
        return {'p': (torch.stack(imgs), {'card': torch.Tensor(latent_labels)})}, f':- not result({label}).'

    def create_latent_dataset(self):
        """
        Create latent labels from task data and image labels.
        :return: Dataframe of the same shape as the card columns in data, but containing numerical latent labels
        """
        latent_ids = pd.read_csv(self.image_dir + "/playing_card_labels.csv")

        suits = ['h', 'c', 's', 'd']
        ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'j', 'q', 'k', 'a']

        # Create an index to map semantic labels to numerical labels
        semantic_to_num = {}
        for s_idx, s in enumerate(suits):
            for r_idx, r in enumerate(ranks):
                card_string = f"{r}{s}"
                semantic_to_num[card_string] = r_idx + (s_idx * len(ranks))

        # Create an index to map image IDs to numerical labels
        id_to_num = latent_ids.set_index('img')['label'].map(semantic_to_num).to_dict()
        # Replace image IDs with their corresponding numerical latent labels and drop result column
        self.latent_labels = self.data.drop(columns=['result']).replace(id_to_num)
