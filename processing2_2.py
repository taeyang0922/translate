import os
from tqdm import tqdm
import numpy as np
import cv2
import torch.utils.data as data
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from prarm2_1 import rec_args

'''
标签处理：定义新字符类处理半字符的情况，比如将'0-1半字符'归到'A'类，减小歧义
识别训练数据构造：从完整图像中裁剪出文本图像作为识别模型输入数据
'''


def PreProcess():
    EXT_CHARS = {
        '01': 'A', '12': 'B', '23': 'C', '34': 'D', '45': 'E',
        '56': 'F', '67': 'G', '78': 'H', '89': 'I', '09': 'J'
    }

    train_dir = 'datasets/data/train_imgs'
    train_labels_dir = 'datasets/data/train_gts'
    word_save_dir = 'temp/rec_datasets/train_imgs'  # 保存识别训练数据集
    os.makedirs(word_save_dir, exist_ok=True)
    label_files = os.listdir(train_labels_dir)
    for label_file in tqdm(label_files):
        with open(os.path.join(train_labels_dir, label_file), 'r') as f:
            lines = f.readlines()
        line = lines[0].strip().split()
        locs = line[:8]
        words = line[8:]

        # 标签处理
        if len(words) == 1:
            ext_word = words[0]
        else:
            assert len(words) % 2 == 0
            ext_word = ''
            for i in range(len(words[0])):
                char_i = [word[i] for word in words]
                if len(set(char_i)) == 1:
                    ext_word += char_i[0]
                elif len(set(char_i)) == 2:
                    char_i = list(set(char_i))
                    char_i.sort()
                    char_i = ''.join(char_i)
                    ext_char_i = EXT_CHARS[char_i]
                    ext_word += ext_char_i

        locs = [int(t) for t in line[:8]]

        # 将倾斜文字图像调整为水平图像
        x1, y1, x2, y2, x3, y3, x4, y4 = locs
        w = int(0.5 * (((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5 + ((x4 - x3) ** 2 + (y4 - y3) ** 2) ** 0.5))
        h = int(0.5 * (((x2 - x3) ** 2 + (y2 - y3) ** 2) ** 0.5 + ((x4 - x1) ** 2 + (y4 - y1) ** 2) ** 0.5))
        src_points = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], dtype='float32')
        dst_points = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype='float32')
        M = cv2.getPerspectiveTransform(src_points, dst_points)
        image = cv2.imread(os.path.join(train_dir, label_file.replace('.txt', '.jpg')))
        word_image = cv2.warpPerspective(image, M, (w, h))

        # save images
        cv2.imwrite(os.path.join(word_save_dir, label_file.replace('.txt', '') + '_' + ext_word + '.jpg'), word_image)


# 运行识别训练数据前处理代码
PreProcess()

'''
数据集导入方法
'''


# data
class WMRDataset(data.Dataset):
    def __init__(self, data_dir=None, max_len=5, resize_shape=(32, 100), train=True):
        super(WMRDataset, self).__init__()
        self.data_dir = data_dir
        self.max_len = max_len
        self.is_train = train

        self.targets = [[os.path.join(data_dir, t), t.split('_')[-1][:5]] for t in os.listdir(data_dir) if
                        t.endswith('.jpg')]
        self.PADDING, self.char2id, self.id2char = self.gen_labelmap()

        # 数据增强
        self.transform = transforms.Compose([
            transforms.Resize(resize_shape),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
            # 可以添加更多的数据增强操作，比如 gaussian blur、shear 等
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.targets)

    @staticmethod
    def gen_labelmap(charset='0123456789ABCDEFGHIJ'):
        # 构造字符和数字标签对应字典
        PADDING = 'PADDING'
        char2id = {t: idx for t, idx in zip(charset, range(1, 1 + len(charset)))}
        char2id.update({PADDING: 0})
        id2char = {v: k for k, v in char2id.items()}
        return PADDING, char2id, id2char

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        img_path = self.targets[index][0]
        word = self.targets[index][1]
        img = Image.open(img_path)

        label = np.full((self.max_len,), self.char2id[self.PADDING], dtype=np.int)
        label_list = []
        word = word[:self.max_len]
        for char in word:
            label_list.append(self.char2id[char])

        label_len = len(label_list)
        assert len(label_list) <= self.max_len
        label[:len(label_list)] = np.array(label_list)

        if self.transform is not None and self.is_train:
            img = self.transform(img)
            img.sub_(0.5).div_(0.5)

        label_len = np.array(label_len).astype(np.int32)
        label = np.array(label).astype(np.int32)

        return img, label, label_len  # 输出图像、文本标签、标签长度, 计算 CTC loss 需要后两者信息


dataset = WMRDataset(rec_args.train_dir, max_len=5, resize_shape=(rec_args.height, rec_args.width), train=True)
train_dataloader = data.DataLoader(dataset, batch_size=2, shuffle=True, pin_memory=True, drop_last=False)
batch = next(iter(train_dataloader))

image, label, label_len = batch
image = ((image[0].permute(1, 2, 0).to('cpu').numpy() * 0.5 + 0.5) * 255).astype(np.uint8)
plt.title('image')
plt.xticks([])
plt.yticks([])
plt.imshow(image)

label_digit = label[0].to('cpu').numpy().tolist()
label_str = ''.join([dataset.id2char[t] for t in label_digit if t > 0])

print('label_digit: ', label_digit)
print('label_str: ', label_str)