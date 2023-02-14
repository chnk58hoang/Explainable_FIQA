from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class ExFIQA(Dataset):
    def __init__(self, df):
        super().__init__()
        self.dataframe = df
        self.image_transform = transforms.Compose([transforms.ToTensor(),
                                                   transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        img_path = self.dataframe.iloc[index]['path']
        sharpness = self.dataframe.iloc[index]['sharpness']
        illu = self.dataframe.iloc[index]['illustration']
        img_path = img_path.replace('/home/artorias/Downloads/archive/casia-webface/','/kaggle/input/casia-webface/casia-webface/')
        image = Image.open(img_path)
        image = self.image_transform(image)
        return image, sharpness, illu 

