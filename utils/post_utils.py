
from torchmetrics import ConfusionMatrix
import pandas as pd
import torch


def wgn2(x, snr):
    "加随机噪声"
    snr = 10**(snr/10.0)
    xpower = torch.sum(x**2)/(x.size(0)*x.size(1)*x.size(2))
    npower = xpower / snr
    return torch.rand(x.size()).cuda() * torch.sqrt(npower)



def test_network(network, data, y, args, noise_dblist, name):
    out = network(data.cuda())
    metric = ConfusionMatrix(num_classes=args.num_classes, task="multiclass").to(args.device)
    confusion_matrix = metric(out.argmax(dim=1), y.to(args.device))
    print(f'##### {name} #####{confusion_matrix}')

    acc_list = []
    with torch.no_grad():
        for noise in noise_dblist:
            print('noise:', noise)
            testx_noise = data.cuda() + wgn2(data, noise)
            pred = network(testx_noise.cuda())
            acc = (pred.argmax(dim=1) == y.to(args.device)).sum().item() / len(y)
            acc_list.append(acc)
            del testx_noise
            del pred
            del acc
            torch.cuda.empty_cache()

    print(f'##### {name} #####{acc_list}')

    # 保存confusion_matrix和acc_list为CSV文件
    confusion_matrix_df = pd.DataFrame(confusion_matrix.cpu().numpy())
    confusion_matrix_df.to_csv(f'{name}_confusion_matrix.csv', index=False)

    acc_list_df = pd.DataFrame(acc_list, columns=['accuracy'])
    acc_list_df.to_csv(f'{name}_acc_list.csv', index=False)