from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split


def get_loaders(root="data/archive", batch_size=128):
    train_tf = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])
    eval_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])

    full_train = datasets.ImageFolder(f"{root}/train", transform=train_tf)
    test_ds    = datasets.ImageFolder(f"{root}/test",  transform=eval_tf)

    val_size = int(0.1 * len(full_train))
    train_ds, val_ds = random_split(full_train, [len(full_train) - val_size, val_size])

    # val uses eval transforms, wrap with separate dataset object
    val_ds_eval = datasets.ImageFolder(f"{root}/train", transform=eval_tf)
    val_ds.dataset = val_ds_eval

    kw = dict(batch_size=batch_size, num_workers=4, pin_memory=True)
    return (
        DataLoader(train_ds, shuffle=True,  **kw),
        DataLoader(val_ds,   shuffle=False, **kw),
        DataLoader(test_ds,  shuffle=False, **kw),
    )
