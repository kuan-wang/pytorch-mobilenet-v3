# A PyTorch implementation of MobileNetV3

This is a PyTorch implementation of MobileNetV3 architecture as described in the paper [Searching for MobileNetV3](https://arxiv.org/pdf/1905.02244.pdf).

Some details may be different from the original paper, welcome to discuss and help me figure it out.

## Training & Accuracy
In progress ...

### MobileNetV3 large
|              | Madds     | Parameters | Top1-acc  | Pretrained Model                                             |
| -----------  | --------- | ---------- | --------- | ------------------------------------------------------------ |
| Offical 1.0  | 219 M     | 5.4  M     | 75.2%     | -                                                            |
| Offical 0.75 | 155 M     | 4    M     | 73.3%     | -                                                            |
| Ours    1.0  |   - M     | 5.14 M     |  -        | - |
| Ours    0.75 |   - M     | 3.72 M     |  -        | - |

### MobileNetV3 small
|              | Madds     | Parameters | Top1-acc  | Pretrained Model                                             |
| -----------  | --------- | ---------- | --------- | ------------------------------------------------------------ |
| Offical 1.0  | 66  M     | 2.9  M     | 67.4%     | -                                                            |
| Offical 0.75 | 44  M     | 2.4  M     | 65.4%     | -                                                            |
| Ours    1.0  |   - M     | 3.11 M     | -         | - |
| Ours    0.75 |   - M     | 2.47 M     | -         | - |

## Usage
Pretrained model are still training ...
```python
    # large
    net_large = mobilenetv3(mode='large')
    # small
    net_small = mobilenetv3(mode='small')
```
