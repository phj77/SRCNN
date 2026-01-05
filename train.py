import torch
import torch.nn.functional as F
import dataset
from torch.utils.data.dataloader import DataLoader
from model import SRCNN
from model import SRCNNResidual
from torchvision import transforms
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
import os
from torchvision.utils import save_image


if __name__ == "__main__":
    #HYPERPARAMETER--------------------------------
    batch_size = 25
    epoch =150
    #optimizer = adam
    learning_rate = 1e-3

    #----------------------------------------------
    train_hr_dir = "./myData/train/hr"
    train_lr_dir = "./myData/train/lr"
    train_hr_cropped_dir = "./myData/train/hr_cropped"

    val_hr_dir = "./myData/val/hr"
    val_lr_dir = "./myData/val/lr"
    val_hr_cropped_dir = "./myData/val/hr_cropped"

    test_hr_dir = "./myData/test/hr"
    test_lr_dir = "./myData/test/lr"
    test_hr_cropped_dir = "./myData/test/hr_cropped"

    #folder---------------------------------------

    #---------------------------------------------

    #device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    #reproducibility
    torch.manual_seed(100)
    generator = torch.Generator().manual_seed(50)
    if device == 'cuda':
        torch.cuda.manual_seed_all(100)

    #transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    #DATASET
    #case1: make dataset
    trainSet = dataset.makeDataSet("./myData/train/hr", "./myData/train/lr", "./myData/train/hr_cropped", scale=3,transform= transform)
    valSet = dataset.makeDataSet("./myData/val/hr", "./myData/val/lr", "./myData/val/hr_cropped", scale=3, transform=transform)
    testSet = dataset.makeDataSet("./myData/test/hr", "./myData/test/lr", "./myData/test/hr_cropped", scale=3, transform=transform)
    #case2: already exist lr dataset in folder
    # trainSet = dataset.SRDataset("/myData/train/lr", "/myData/train/hr_cropped",transform= transform)
    # valSet = dataset.SRDataset("/myData/val/lr", "/myData/val/hr_cropped",transform= transform)
    # testSet = dataset.SRDataset("/myData/test/lr", "/myData/test/hr_cropped",transform= transform)

    #setup loader
    train_loader = DataLoader(dataset=trainSet, batch_size= batch_size, shuffle= True, num_workers= 2, pin_memory=True,drop_last= False)
    val_loader = DataLoader(dataset=valSet, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # PSNR 및 SSIM 계산용 객체
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    #setup model
    model = SRCNNResidual(num_channels=3).to(device)
    print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    #scheduler = StepLR(optimizer, step_size=30, gamma=0.5)  # 매 30 epoch마다 lr을 50%로 감소

    #function to train one epoch.
    def train_one_epoch(model, train_loader, optimizer, device):
        model.train()
        running_loss = 0.0
        for lr_img, hr_img in train_loader:
            lr_img, hr_img = lr_img.to(device), hr_img.to(device)
            outputs = model(lr_img)
            loss = F.mse_loss(outputs, hr_img)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_loss = running_loss / len(train_loader)
        return train_loss

    #function to calculate validation 
    # Validation Function 수정
    def validate(model, val_loader, device):
        model.eval()
        running_loss = 0.0
        total_psnr = 0.0
        total_ssim = 0.0
        with torch.no_grad():
            for batch_idx, (lr_img, hr_img) in enumerate(val_loader):
                lr_img, hr_img = lr_img.to(device), hr_img.to(device)
                outputs = model(lr_img)
                loss = F.mse_loss(outputs, hr_img)
                running_loss += loss.item()
                total_psnr += psnr_metric(outputs, hr_img).item()
                total_ssim += ssim_metric(outputs, hr_img).item()
        
        val_loss = running_loss / len(val_loader)
        avg_psnr = total_psnr / len(val_loader)
        avg_ssim = total_ssim / len(val_loader)
        return val_loss, avg_psnr, avg_ssim

    
    # Train Loop 및 Metric 저장
    train_losses, val_losses, psnr_scores, ssim_scores = [], [], [], []

    best_psnr = 0.0
    #train
    # Train Loop
    for ep in range(epoch):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, avg_psnr, avg_ssim = validate(model, val_loader, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        psnr_scores.append(avg_psnr)
        ssim_scores.append(avg_ssim)
        
        print(f"Epoch {ep+1}/{epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, PSNR: {avg_psnr:.2f} dB, SSIM: {avg_ssim:.4f}")
        
        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            torch.save(model.state_dict(), "best_srcnn.pth")
            print(f"Model saved at epoch {ep+1} with PSNR: {avg_psnr:.2f} dB")


    # 결과 시각화
    epochs = range(1, epoch + 1)
    plt.figure(figsize=(12, 8))

    # Loss 그래프
    plt.subplot(3, 1, 1)  # 3행 1열에서 첫 번째 그래프
    plt.plot(epochs, train_losses, label="Train Loss", color="blue")
    plt.plot(epochs, val_losses, label="Validation Loss", color="orange")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss over Epochs")
    plt.legend()

    # PSNR 그래프
    plt.subplot(3, 1, 2)  # 3행 1열에서 두 번째 그래프
    plt.plot(epochs, psnr_scores, label="PSNR (dB)", color="green")
    plt.xlabel("Epochs")
    plt.ylabel("PSNR (dB)")
    plt.title("PSNR over Epochs")
    plt.legend()

    # SSIM 그래프
    plt.subplot(3, 1, 3)  # 3행 1열에서 세 번째 그래프
    plt.plot(epochs, ssim_scores, label="SSIM", color="red")
    plt.xlabel("Epochs")
    plt.ylabel("SSIM")
    plt.title("SSIM over Epochs")
    plt.legend()

    plt.tight_layout()
    plt.show()
