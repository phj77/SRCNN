import os
import torch
import dataset
from model import SRCNN
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchvision.utils import save_image
from PIL import Image

# HYPERPARAMETER
batch_size = 1
learning_rate = 1e-3

# 경로 설정
test_hr_dir = "./myData/test/hr"
test_lr_dir = "./myData/test/lr"
test_hr_cropped_dir = "./myData/test/hr_cropped"
output_dir = "./myData/test/output"
os.makedirs(output_dir, exist_ok=True)

# Device 설정
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Transform 설정
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# 테스트 데이터셋
testSet = dataset.SRDataset(test_lr_dir, test_hr_cropped_dir, transform=transform)
test_loader = DataLoader(dataset=testSet, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

# 모델 로드
model = SRCNN(num_channels=3).to(device)
model.load_state_dict(torch.load("best_srcnn.pth", map_location=device))
model.eval()

# PSNR 및 SSIM 계산용 객체
psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

# 테스트 수행 및 결과 저장
def test(model, test_loader, output_dir, device):
    total_psnr = 0.0
    total_ssim = 0.0
    num_samples = 0

    with torch.no_grad():
        for idx, (lr_img, hr_img) in enumerate(test_loader):
            lr_img, hr_img = lr_img.to(device), hr_img.to(device)
            outputs = model(lr_img)

            # 정규화 복원
            outputs = outputs * 0.5 + 0.5
            hr_img = hr_img * 0.5 + 0.5
            lr_img = lr_img * 0.5 + 0.5

            # PSNR 및 SSIM 계산
            total_psnr += psnr_metric(outputs, hr_img).item()
            total_ssim += ssim_metric(outputs, hr_img).item()
            num_samples += 1

            # 결과 저장
            output_image_path = os.path.join(output_dir, f"output_{idx}.png")
            save_image(outputs, output_image_path)
            print(f"Saved: {output_image_path}")

    # 평균 PSNR 및 SSIM 계산
    avg_psnr = total_psnr / num_samples
    avg_ssim = total_ssim / num_samples
    return avg_psnr, avg_ssim


# 실행
if __name__ == "__main__":
    avg_psnr, avg_ssim = test(model, test_loader, output_dir, device)
    print(f"Test Results - PSNR: {avg_psnr:.2f} dB, SSIM: {avg_ssim:.4f}")
