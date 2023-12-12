import argparse
import torch
from torch.utils.data import DataLoader
import models
from dataloaders.sketch_dataloader import SketchDataloader
from models.transformers import Sketchformer


import json

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = json.load(file)
    return config


def main():
    parser = argparse.ArgumentParser(description='Train transformer with sketch data')
    parser.add_argument("--dataset", required=True, help="Input data folder")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--model-name", default="Sketchformer", help="Model to train")
    parser.add_argument("--data-loader", default='stroke3-distributed', help="Data loader for sketch data")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID to run on")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    
    args = parser.parse_args()

    # Set up GPU
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    dataset = SketchDataloader(args.dataset)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Initialize the model
    config = load_config("config.json")
    model = Sketchformer(config).to(device)
    model.print_config()
    
    # Define loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()  # Adjust as needed
    optimizer = torch.optim.Adam(model.parameters())

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        for batch in dataloader:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)
            
            # Compute loss
            loss = criterion(outputs, targets)

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch+1}/{args.epochs}], Loss: {loss.item()}")

    # Save the model
    torch.save(model.state_dict(), f"{args.output_dir}/model.pth")

if __name__ == '__main__':
    main()