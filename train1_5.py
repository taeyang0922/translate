import torch
import os
import numpy as np
import torch.utils.data as data
from processing1_2 import ImageDataset,train_processes,device,det_args,DEBUG
from backbone1_4 import SegDetectorModel
# 学习率调整方法
class DecayLearningRate():
    def __init__(self, lr=0.004, epochs=200, factor=0.9):
        self.lr = lr
        self.epochs = epochs
        self.factor = factor

    def get_learning_rate(self, epoch):
        # 学习率随着训练过程进行不断下降
        rate = np.power(1.0 - epoch / float(self.epochs + 1), self.factor)
        return rate * self.lr

    def update_learning_rate(self, optimizer, epoch):
        lr = self.get_learning_rate(epoch)
        for group in optimizer.param_groups:
            group['lr'] = lr
        return lr


# 检测模型训练
def det_train():
    # model
    model = SegDetectorModel(device)

    # data_loader
    train_dataset = ImageDataset(data_dir=det_args.train_dir, gt_dir=det_args.train_gt_dir, is_training=True,
                                 processes=train_processes)
    train_dataloader = data.DataLoader(train_dataset, batch_size=det_args.batch_size, num_workers=det_args.num_workers,
                                       shuffle=True, drop_last=False)

    # initialize dataloader and scheduler
    optimizer = torch.optim.SGD(model.parameters(), lr=det_args.lr, momentum=0.9, weight_decay=1e-4)
    scheduler = DecayLearningRate(lr=det_args.lr, epochs=det_args.max_epoch)

    step = 0
    epoch = 0
    model.train()
    os.makedirs(det_args.save_dir, exist_ok=True)
    for epoch in range(det_args.max_epoch):
        for batch in train_dataloader:
            step += 1
            current_lr = scheduler.update_learning_rate(optimizer, epoch)  # 学习率调整

            optimizer.zero_grad()
            pred, loss, metrics = model.forward(batch, training=True)
            loss.backward()
            optimizer.step()

            if step % det_args.print_interval == 0:
                line = []
                for key, l_val in metrics.items():
                    line.append('{0}: {1:.4f}'.format(key, l_val.mean()))
                line = '\t'.join(line)
                info = '\t'.join(['step:{:6d}', 'epoch:{:3d}', 'loss: {:4f}', '{}', 'lr: {:.4f}']).format(step, epoch,
                                                                                                          loss.item(),
                                                                                                          line,
                                                                                                          current_lr)
                print(info)
                if DEBUG:  # DEBUG 这里只跑少量数据
                    break

        # 保存阶段模型
        if epoch % det_args.save_interval == 0:
            save_name = 'checkpoint_' + str(epoch)
            torch.save(model.state_dict(), os.path.join(det_args.save_dir, save_name))
        if DEBUG:
            break

    # 保存最终模型
    torch.save(model.state_dict(), det_args.saved_model_path)