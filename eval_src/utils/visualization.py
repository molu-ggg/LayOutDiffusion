import numpy as np
from PIL import Image, ImageDraw,ImageFont

import torch
import torchvision.utils as vutils
import torchvision.transforms as T
en_labels =['床(正视)', '壁挂空调(正视)', '台灯(正视)', '落地床头柜(正视)', '床(左视)', '落地床头柜(左视)', 
'横向装饰画(左视)', '扫地机器人(正视)', '落地灯(右视)', '横向装饰画(正视)', '装饰画(左视)', '床与床头柜组合(正视)', 
'装饰画(正视)', '吊灯(正视)', '竖向装饰画(正视)', '衣帽架(正视)', '吸顶灯(正视)', '落地梳妆台(正视)', '床(右视)', 
'落地床头柜(右视)', '四门衣柜(左视)', '装饰画(右视)', '空气净化器(正视)', '床尾凳(正视)', '横向装饰画(右视)', 
'落地灯(左视)', '吸尘器(正视)', '加湿器(正视)', '多门衣柜(右视)', '单人床尾凳(正视)', '落地空调(正视)',
 '落地梳妆台组合(正视)', '竖向装饰画(右视)', '四门衣柜(右视)', '壁挂空调(左视)', '落地灯(正视)', 
 '空气净化器(右视)', '桌面加湿器(正视)', '多门衣柜(左视)', '落地无镜梳妆台组合(正视)', '扫地机器人(右视)', 
 '双门衣柜(正视)', '竖向装饰画(左视)', '台灯(左视)', '壁挂空调(右视)', '吊灯(右视)', '洗地机(正视)', 
 '落地梳妆台组合(右视)', '四门衣柜(正视)', '吸尘器(左视)', '落地梳妆台(右视)', '吸顶灯(左视)']

def convert_layout_to_image(boxes, labels, colors, canvas_size):
    H, W = canvas_size
    img = Image.new('RGB', (int(W), int(H)), color=(255, 255, 255))
    font_path = '/home/ubuntu/ygq/LayOutDiffusion/simsun.ttc'  # 替换为实际的DejaVu Sans字体路径
    font_size = 16
    font = ImageFont.truetype(font_path, font_size)
    draw = ImageDraw.Draw(img, 'RGBA')

    # draw from larger boxes
    area = [b[2] * b[3] for b in boxes]
    indices = sorted(range(len(area)),
                     key=lambda i: area[i],
                     reverse=True)

    for i in indices:
        bbox, color = boxes[i], colors[labels[i]]
        c_fill = color + (100,)
        x1, y1, x2, y2 = bbox
        x1, x2 = x1 * (W - 1), x2 * (W - 1)
        y1, y2 = y1 * (H - 1), y2 * (H - 1)
        if x2>x1 and y2>y1:
            draw.rectangle([x1, y1, x2, y2],
                        outline=color,
                        fill=c_fill)
            draw.text((x1, y1),en_labels[int(labels[i])-1], fill='red',font = font )
#         draw.rectangle([x1, y1, x2, y2],
#             outline=color,
#             fill=c_fill)

    return img


def save_image(batch_boxes, batch_labels, batch_mask,
               dataset_colors, out_path, canvas_size=(60, 40),
               nrow=None):
    # batch_boxes: [B, N, 4]
    # batch_labels: [B, N]
    # batch_mask: [B, N]

    imgs = []
    B = batch_boxes.size(0)
    to_tensor = T.ToTensor()
    for i in range(B):
        try:
            mask_i = batch_mask[i]
            boxes = batch_boxes[i][mask_i]
            labels = batch_labels[i][mask_i]
            img = convert_layout_to_image(boxes, labels,
                                        dataset_colors,
                                        canvas_size)
            imgs.append(to_tensor(img))
        except IndexError as e:
            print("index exception")
            continue
    image = torch.stack(imgs)

    if nrow is None:
        nrow = int(np.ceil(np.sqrt(B)))

    vutils.save_image(image, out_path, normalize=False, nrow=nrow)
