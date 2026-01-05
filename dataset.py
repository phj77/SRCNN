import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

def save_image(image, save_dir, filename):
    """
    이미지를 지정된 경로에 저장.
    Args:
        image (PIL.Image): 저장할 이미지.
        save_dir (str): 저장할 디렉토리 경로.
        filename (str): 저장할 파일 이름.
    """
    os.makedirs(save_dir, exist_ok=True)  # 디렉토리가 없으면 생성
    save_path = os.path.join(save_dir, filename)
    image.save(save_path)

class makeDataSet(Dataset):
    def __init__(self, hr_dir, lr_save_dir, hr_cropped_dir, scale=3, transform=None):
        """
        Args:
            hr_dir (str): 고해상도 이미지 폴더 경로
            scale (int): 다운샘플링 비율 (기본값 3)
            transform (callable, optional): 변환 작업 (augmentation)
        """
        if not os.path.exists(hr_dir):
            raise FileNotFoundError(f"HR directory not found: {hr_dir}")
        self.hr_dir = hr_dir
        self.hr_dir_cropped = hr_cropped_dir
        self.lr_save_dir = lr_save_dir
        self.hr_images = sorted(os.listdir(hr_dir))
        self.scale = scale
        self.transform = transform

        os.makedirs(self.hr_dir_cropped, exist_ok=True)
        os.makedirs(self.lr_save_dir, exist_ok=True)


    def __len__(self):
        return len(self.hr_images)

    def __getitem__(self, idx):
        # 고해상도 이미지 로드
        hr_path = os.path.join(self.hr_dir, self.hr_images[idx])
        hr_img = Image.open(hr_path).convert("RGB")

        #256*256 이미지 255*255로 만들기
        hr_img_croped = hr_img.crop((0,0,hr_img.width - 1, hr_img.height - 1))

        # 저해상도 이미지 생성 (1/scale 크기로 다운샘플링)
        lr_img = hr_img_croped.resize(
            (hr_img_croped.width // self.scale, hr_img_croped.height // self.scale),
            Image.BICUBIC
        )

        #crop된 고해상도 이미지 저장
        save_image(hr_img_croped, self.hr_dir_cropped, f"hr_cropped_{self.hr_images[idx]}")

        # 저해상도 이미지 저장
        save_image(lr_img, self.lr_save_dir, f"lr_{self.hr_images[idx]}")

        # Transform 적용
        if self.transform:
            lr_img = self.transform(lr_img)
            hr_img_croped  = self.transform(hr_img_croped)

        return lr_img, hr_img_croped


class SRDataset(Dataset):
    def __init__(self, input_dir, target_dir, transform=None):
        """
        Args:
            input_dir (str): 저해상도 이미지 폴더 경로
            target_dir (str): 고해상도 이미지 폴더 경로
            transform (callable, optional): 이미지에 적용할 변환 (augmentation)
        """
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.input_images = sorted(os.listdir(input_dir))
        self.target_images = sorted(os.listdir(target_dir))
        self.transform = transform

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        # 저해상도 이미지
        input_path = os.path.join(self.input_dir, self.input_images[idx])
        input_img = Image.open(input_path).convert("RGB")

        # 고해상도 이미지
        target_path = os.path.join(self.target_dir, self.target_images[idx])
        target_img = Image.open(target_path).convert("RGB")

        # Transform 적용
        if self.transform:
            input_img = self.transform(input_img)
            target_img = self.transform(target_img)

        return input_img, target_img