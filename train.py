import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from dataset import flickr30k_dataset
from model import CustomModel


def train():
    model_path = 'result/model.pth'
    optimizer_path = 'result/optimizer.pth'

    # Create the DataLoader
    clip_dataloader = DataLoader(flickr30k_dataset, batch_size=Config.batch_size, shuffle=True, num_workers=4)

    # Create an instance of your model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CustomModel().to(device)

    # Define optimizer
    optimizer = torch.optim.Adam([
        {'params': model.vision_encoder.parameters()},
        {'params': model.caption_encoder.parameters()}
    ], lr=model.lr)

    batch_zero = True
    for epoch in range(0, Config.epochs):
        model_path1 = f"result/model_{epoch}.pth"
        model.train()
        for batch in tqdm(clip_dataloader):
            image = batch["image"].to(device)
            text = batch["caption"]
            # images,text = batch
            loss, img_acc, cap_acc = model(image, text)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_zero:
                print(f"Epoch [{0}/{Config.epochs}], Batch Loss: {loss.item()}")
                batch_zero = False

        # Print training statistics
        print(f"Epoch [{epoch+1}/{Config.epochs}], Batch Loss: {loss.item()}")

    # 保存模型和优化器的状态
    torch.save(model.state_dict(), model_path)
    torch.save(optimizer.state_dict(), optimizer_path)