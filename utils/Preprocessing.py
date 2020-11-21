import albumentations as A
from albumentations.pytorch.transforms import ToTensor
import torch
import numpy as np




mean = [0.485]
std = [0.229]
normalize = {'mean': mean, 'std': std}


def complex_preprocess(normalize = normalize):
    return A.Compose([A.Cutout(num_holes=8, max_h_size=3, max_w_size=3, p=0.3),
                      A.ShiftScaleRotate(shift_limit=(-0.05, 0.05), scale_limit=(0, .01), rotate_limit=45,
                                         border_mode=0),
                      A.HorizontalFlip(),
                      A.RandomGamma((40, 120)),
                      A.GaussNoise(var_limit=0.01),
                      A.RandomContrast((-0.2, 0.2)),
                      ToTensor(normalize=normalize)

                      ])


def simple_preprocess(normalize = normalize):
    return A.Compose([ToTensor(normalize=normalize)
                      ])

class FashionDataset(torch.utils.data.Dataset):
  """ Fashion Mnist custom Dataset"""
  def __init__(self,image_array , labels_array,transform=simple_preprocess()):
        """
        Args:
            image_array (array): array of shape (nb_samples,h,w) containing train or test data
            labels_array (array): array of shape nb_samples containing labels, in case mode='train
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.image_array = image_array
        self.labels_array = labels_array
        self.transform = transform
  def __len__(self):
        return np.shape(self.image_array)[0]
  def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image = self.image_array[idx]
        image = np.expand_dims(image,-1)
        label = self.labels_array[idx]
        if self.transform:
            data = {"image": image}
            augmented = self.transform(**data)
            image = augmented['image']
        return image, label


def to_numpy(x):
    if not (isinstance(x, np.ndarray) or x is None):
        if x.is_cuda:
            x = x.data.cpu()
        x = x.numpy()
    return x
def unnormalize(img,mean = mean, std = std,image_size=(28,28)):

  "Unnormalize a given image tensor and make it plotable"
  # plt imshow only accept positive values
  unnormalized_img = std[0]*img +mean[0]
  return to_numpy(unnormalized_img)