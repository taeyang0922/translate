import os
import torch
import torchvision
from tqdm import tqdm
import numpy as np
import cv2
from PIL import Image
from prarm2_1 import device,rec_args
from processing2_2 import WMRDataset
from backbone2_3 import RecModelBuilder


'''
根据检测结果生成识别模型测试数据
'''
def rec_test_data_gen():
    test_dir = 'datasets/data/test_imgs'
    det_dir = 'temp/det_res'
    word_save_dir = 'temp/rec_datasets/test_imgs'


    os.makedirs(word_save_dir, exist_ok=True)
    label_files = os.listdir(det_dir)
    for label_file in tqdm(label_files):
        if not label_file.endswith('.txt'):
            continue
        with open(os.path.join(det_dir, label_file), 'r') as f:
            lines = f.readlines()
        if len(lines) == 0:
            continue
        line = lines[0].strip().split(',')
        locs = [float(t) for t in line[:8]]


        # image warp
        x1, y1, x2, y2, x3, y3, x4, y4 = locs
        w = int(0.5 * (((x2 -x1 )** 2 +(y2 -y1 )**2 )**0.5 + ((x4 -x3 )** 2 +(y4 -y3 )**2 )**0.5))
        h = int(0.5 * (((x2 -x3 )** 2 +(y2 -y3 )**2 )**0.5 + ((x4 -x1 )** 2 +(y4 -y1 )**2 )**0.5))
        src_points = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], dtype='float32')
        dst_points = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype='float32')
        M = cv2.getPerspectiveTransform(src_points, dst_points)
        image = cv2.imread(os.path.join(test_dir, label_file.replace('det_res_', '')[:-4] + '.jpg'))
        word_image = cv2.warpPerspective(image, M, (w, h))

        # save images
        cv2.imwrite(os.path.join(word_save_dir, label_file.replace('det_res_', '')[:-4 ] +'.jpg'), word_image)


# 使用检测模型获取识别测试数据
rec_test_data_gen()


# inference
# 模型输出进行CTC对应解码，去除blank，将连续同字符合并
def rec_decode(rec_prob, labelmap, blank=0):
    raw_str = torch.max(rec_prob, dim=-1)[1].data.cpu().numpy()
    res_str = []
    for b in range(len(raw_str)):
        res_b = []
        prev = -1
        for ch in raw_str[b]:
            if ch == prev or ch == blank:
                prev = ch
                continue
            res_b.append(labelmap[ch])
            prev = ch
        res_str.append(''.join(res_b))
    return res_str


def rec_load_test_image(image_path, size=(100, 32)):
    img = Image.open(image_path)
    img = img.resize(size, Image.BILINEAR)
    img = torchvision.transforms.ToTensor()(img)
    img.sub_(0.5).div_(0.5)
    return img.unsqueeze(0)


# 测试
def rec_test():
    model = RecModelBuilder(rec_num_classes=rec_args.voc_size, sDim=rec_args.decoder_sdim)
    model.load_state_dict(torch.load(rec_args.saved_model_path, map_location=device))
    model.eval()

    os.makedirs(rec_args.rec_res_dir, exist_ok=True)
    _, _, labelmap = WMRDataset().gen_labelmap()  # labelmap是类别和字符对应的字典
    with torch.no_grad():
        for file in tqdm(os.listdir(rec_args.test_dir)):
            img_path = os.path.join(rec_args.test_dir, file)
            image = rec_load_test_image(img_path)
            batch = [image, None, None]
            pred_prob = model.forward(batch)
            # todo post precess
            rec_str = rec_decode(pred_prob, labelmap)[0]
            # write to file
            with open(os.path.join(rec_args.rec_res_dir, file.replace('.jpg', '.txt')), 'w') as f:
                f.write(rec_str)
# 运行测试代码
rec_test()