import os
import cv2
import matplotlib.pyplot as plt

'''
识别结果后处理
'''
def final_postProcess():
    SPECIAL_CHARS = {k :v for k, v in zip('ABCDEFGHIJ', '1234567890')}


    test_dir = 'data/test_imgs/'
    rec_res_dir = 'temp/rec_res/'
    rec_res_files = os.listdir(rec_res_dir)


    final_res = dict()
    for file in os.listdir(test_dir):
        res_file = file.replace('.jpg', '.txt')
        if res_file not in rec_res_files:
            final_res[file] = ''
            continue

        with open(os.path.join(rec_res_dir, res_file), 'r') as f:
            rec_res = f.readline().strip()
        final_res[file] = ''.join([t if t not in 'ABCDEFGHIJ' else SPECIAL_CHARS[t] for t in rec_res])


    with open('work/final_res.txt', 'w') as f:
        for key, value in final_res.items():
            f.write(key + '\t' + value + '\n')


# 生成最终的测试结果
final_postProcess()

'''
最终结果可视化
'''
test_dir = 'data/test_imgs/'
with open('work/final_res.txt', 'r') as f:
    lines = f.readlines()

plt.figure(figsize=(60, 60))
lines = lines[:5]
for i, line in enumerate(lines):
    if len(line.strip().split()) == 1:
        image_name = line.strip()  # 没有识别出来
        word = '###'
    else:
        image_name, word = line.strip().split()
    image = cv2.imread(os.path.join(test_dir, image_name))

    plt.subplot(151 + i)
    plt.title(word, fontdict={'size': 50})
    plt.xticks([])
    plt.yticks([])
    plt.imshow(image)
    plt.show()