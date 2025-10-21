from torch.utils.data import DataLoader
import itertools
from datetime import datetime

from dataset import *
from model import *


train_file = "/home/xfaron/Desktop/Code/Playground/test_construction/data/external/CubiCasa5k/data/cubicasa5k/train.txt"
val_file = "/home/xfaron/Desktop/Code/Playground/test_construction/data/external/CubiCasa5k/data/cubicasa5k/val.txt"
data_folder = "/home/xfaron/Desktop/Code/Playground/test_construction/data/external/CubiCasa5k/data/cubicasa5k"

device = "cuda" if torch.cuda.is_available() else "cpu"
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

exp_name = f"wall_seg_{timestamp}"
exp_folder = "/home/xfaron/Desktop/Code/Playground/test_construction/experiments"
EPOCHS = 5


def train_one_epoch(epoch_index):
    running_loss = 0
    last_loss = 0

    i = 0
    for i, data in enumerate(tqdm(training_loader)):
        img, mask = data
        img, mask = img.to(device, non_blocking=True), mask.to(
            device, non_blocking=True
        )
        optimizer.zero_grad()

        output_mask = model(img)

        aff_loss = affinity_loss(output_mask)
        bce_loss = F.binary_cross_entropy_with_logits(output_mask, mask)
        loss = criterion(bce_loss, aff_loss)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 500 == 499:
            last_loss = running_loss / 500
            print(f"Batch : {i + 1}, Loss : {last_loss}")

    return running_loss / i


if __name__ == "__main__":

    train_folders = []
    val_folders = []
    with open(train_file) as file, open(val_file) as val_file:
        train_folders = file.read().split("\n")
        val_folders = val_file.read().split("\n")

    train_dataset = WallSegDataset(data_folder, train_folders)
    val_dataset = WallSegDataset(data_folder, val_folders)

    training_loader = DataLoader(train_dataset, batch_size=1)
    val_loader = DataLoader(val_dataset, batch_size=1)

    model = FPN()
    model = torch.nn.DataParallel(model)
    model.to(device)
    criterion = MultiLoss().to(device)
    optimizer = torch.optim.SGD(
        itertools.chain(model.parameters(), criterion.parameters()),
        lr=0.001,
        momentum=0.9,
    )

    best_vloss = 1_000_000

    for epoch in range(EPOCHS):
        print(f"EPOCH: {epoch + 1}")

        # Training
        model.train(True)
        avg_loss = train_one_epoch(epoch)

        # Evaluation
        vrunning_loss = 0

        model.eval()
        with torch.no_grad():
            for i, vdata in val_loader:
                vimg, vmask = vdata
                vimg, vmask = vimg.to(device, non_blocking=True), vmask.to(
                    device, non_blocking=True
                )

                output_mask = model(vimg)

                aff_loss = affinity_loss(output_mask)
                bce_loss = F.binary_cross_entropy_with_logits(output_mask, vmask)
                vloss = criterion(bce_loss, aff_loss)

                vrunning_loss += vloss

        avg_vloss = (vrunning_loss) / (i + 1)
        print(f"LOSS > Train : {avg_loss}, Val : {avg_vloss}")

        if avg_loss < best_vloss:
            best_vloss = avg_vloss
            model_path = os.path.join(exp_folder, exp_name, f"model_{epoch}_{best_vloss}")
            torch.save(model.state_dict(), model_path)
