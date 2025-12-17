from torchvision import transforms


def get_transforms(img_size: int, mean: list, std: list) -> tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        # Increase to 90 for top-down view
        transforms.RandomRotation(degrees=90),
        transforms.RandomApply([
            transforms.ColorJitter(
                brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)
        ], p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
        # transforms.RandomErasing(p=0.2, scale=(
        #     0.02, 0.1))  # Simulate occlusions
    ])


    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    return train_transform, val_transform
