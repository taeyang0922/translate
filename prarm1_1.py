import torch
# 是否使用 GPU
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

DEBUG = True  # Debug模式可快速跑通代码，非Debug模式可得到更好的结果
# 检测和识别模型需要足够的训练迭代次数，因此DEBUG模式下几乎得不到最终有效结果

# 参数设置
class DetOptions():
    def __init__(self):
        self.lr = 0.004
        self.max_epoch = 200
        self.batch_size = 8
        self.num_workers = 8
        self.print_interval = 100
        self.save_interval = 10
        self.train_dir = 'datasets/data/train_imgs'
        self.train_gt_dir = 'datasets/data/train_gts'
        self.test_dir = 'datasets/data/test_imgs'
        self.save_dir = 'temp/det_models/'                            # 保存检测模型
        self.saved_model_path = 'temp/det_models/checkpoint_final'    # 保存最终检测模型
        self.det_res_dir = 'temp/det_res/'                            # 保存测试集检测结
        self.thresh = 0.3                                             # 分割后处理阈值
        self.box_thresh = 0.5                                         # 检测框阈值
        self.max_candidates = 10                                      # 候选检测框数量（本数据集每张图像只有一个文本，因此可置为1）
        self.test_img_short_side = 640                                # 测试图像最短边长度

det_args = DetOptions()

# DEBUG 模式跑通代码
if DEBUG:
    det_args.max_epoch = 1
    det_args.print_interval = 1
    det_args.save_interval = 1
    det_args.batch_size = 2
    det_args.num_workers = 0