import torch

DEBUG = True  # Debug模式可快速跑通代码，非Debug模式可得到更好的结果
# 检测和识别模型需要足够的训练迭代次数，因此DEBUG模式下几乎得不到最终有效结果

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


class RecOptions():
    def __init__(self):
        self.height = 32  # 图像尺寸
        self.width = 100
        self.voc_size = 21  # 字符数量 '0123456789ABCDEFGHIJ' + 'PADDING'位
        self.decoder_sdim = 512
        self.max_len = 5  # 文本长度
        self.lr = 1.0
        self.milestones = [40, 60]  # 在第 40 和 60 个 epoch 训练时降低学习率
        self.max_epoch = 80
        self.batch_size = 64
        self.num_workers = 8
        self.print_interval = 25
        self.save_interval = 125
        self.train_dir = 'temp/rec_datasets/train_imgs'
        self.test_dir = 'temp/rec_datasets/test_imgs'
        self.save_dir = 'temp/rec_models/'
        self.saved_model_path = 'temp/rec_models/checkpoint_final'
        self.rec_res_dir = 'temp/rec_res/'

    def set_(self, key, value):
        if hasattr(self, key):
            setattr(self, key, value)


rec_args = RecOptions()

if DEBUG:
    rec_args.max_epoch = 1
    rec_args.print_interval = 20
    rec_args.save_interval = 1

    rec_args.batch_size = 10
    rec_args.num_workers = 0