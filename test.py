from typing import List, Tuple
import torch
from datasets import tqdm
from dataset import bili
from torch.utils.data import DataLoader

from config import Config
from dataset import flickr30k_dataset1
from model import CustomModel

def test_model(model: CustomModel, dataloader: torch.utils.data.DataLoader,device) -> Tuple[float, float]:
    model.eval()
    total_img_recall = 0
    total_cap_recall = 0
    num_batches = 0


    with torch.no_grad():
        for batch in tqdm(dataloader):
            image = batch["image"].to(device)
            text = batch["caption"]
            _, img_acc, cap_acc = model(image, text)
            total_img_recall += img_acc.item()
            total_cap_recall += cap_acc.item()
            num_batches += bili

    avg_img_recall = total_img_recall / num_batches
    avg_cap_recall = total_cap_recall / num_batches

    return avg_img_recall, avg_cap_recall

def test():
    dataloader = DataLoader(flickr30k_dataset1, batch_size=Config.batch_size, shuffle=True, num_workers=4)

    model = CustomModel()
    path = 'result/model.pth'
    model.load_state_dict(torch.load(path, map_location=torch.device('cuda')))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    img_recall, cap_recall = test_model(model, dataloader,device)
    print(f"Image to Caption Recall: {img_recall}")
    print(f"Caption to Image Recall: {cap_recall}")
