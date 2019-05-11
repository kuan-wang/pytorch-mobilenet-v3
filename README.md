# A PyTorch implementation of MobileNetV3

This is a PyTorch implementation of MobileNetV3 architecture as described in the paper [Searching for MobileNetV3](https://arxiv.org/pdf/1905.02244.pdf).

Some details may be different from the original paper, welcome to discuss and help me figure it out.

**[NEW]** The pretrained model of small version mobilenet-v3 is online.

## Training & Accuracy
In progress ...

### MobileNetV3 large
|              | Madds     | Parameters | Top1-acc  | Pretrained Model                                             |
| -----------  | --------- | ---------- | --------- | ------------------------------------------------------------ |
| Offical 1.0  | 219 M     | 5.4  M     | 75.2%     | -                                                            |
| Offical 0.75 | 155 M     | 4    M     | 73.3%     | -                                                            |
| Ours    1.0  |   - M     | 5.08 M     |  -        | - |
| Ours    0.75 |   - M     | 3.69 M     |  -        | - |

### MobileNetV3 small
|              | Madds     | Parameters | Top1-acc  | Pretrained Model                                             |
| -----------  | --------- | ---------- | --------- | ------------------------------------------------------------ |
| Offical 1.0  | 66  M     | 2.9  M     | 67.4%     | -                                                            |
| Offical 0.75 | 44  M     | 2.4  M     | 65.4%     | -                                                            |
| Ours    1.0  | 57  M     | 3.11 M     | 67.218%   |  [[google drive](https://drive.google.com/open?id=1397oUs0VDgZXDn4pqKZPD5NJoDQ-vlZK)] |
| Ours    0.75 |   - M     | 2.47 M     | -         | - |

## Usage
Pretrained models are still training ...
```python
    # pytorch 1.0.1
    # large
    net_large = mobilenetv3(mode='large')
    # small
    net_small = mobilenetv3(mode='small')
    state_dict = torch.load('mobilenetv3_small_67.218.pth.tar')
    net_small.load_state_dict(state_dict)
```

## Data Pre-processing

I used the following code for data pre-processing on ImageNet:

```python
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

input_size = 224
train_dataset = datasets.ImageFolder(
    traindir,
    transforms.Compose([
        transforms.RandomResizedCrop(input_size, scale=(0.2, 1.0)), 
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]))

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True,
    num_workers=n_worker, pin_memory=True)

val_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(valdir, transforms.Compose([
        transforms.Resize(int(input_size/0.875)),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        normalize,
    ])),
    batch_size=batch_size, shuffle=False,
    num_workers=n_worker, pin_memory=True)
```

