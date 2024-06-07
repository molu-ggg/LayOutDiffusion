import os.path as osp
from glob import glob
import numpy as np
import cv2
from PIL import Image
from sklearn.cluster import KMeans
from PIL import Image
from tqdm import tqdm
import json
import os

color_table = {
    "1":"""
#2F0000
#4D0000
#600000
#750000
#930000
#AE0000
#CE0000
#EA0000
#FF0000
#FF2D2D
#FF5151
#f7575
#FF9797
#FFB5B5
#FFD2D2
#FFECEC
    """,
    "2":"""
#600030
#820041
#9F0050
#BF0060
#D9006C
#F00078
#FF0080
#FF359A
#FF60AF
#FF79BC
#FF95CA
#ffaad5
#FFC1E0
#FFD9EC
#FFECF5
#FFF7FB
    """,
    "3":"""
#460046
#5E005E
#750075
#930093
#AE00AE
#D200D2
#E800E8
#FF00FF
#FF44FF
#FF77FF
#FF8EFF
#ffa6ff
#FFBFFF
#FFD0FF
#FFE6FF
#FFF7FF
    """,
    "4":"""
#28004D
#3A006F
#4B0091
#5B00AE
#6F00D2
#8600FF
#921AFF
#9F35FF
#B15BFF
#BE77FF
#CA8EFF
#d3a4ff
#DCB5FF
#E6CAFF
#F1E1FF
#FAF4FF
    """,
    "5":"""
#000079
#003D79
#004B97
#005AB5
#0066CC
#0072E3
#0080FF
#2894FF
#46A3FF
#66B3FF
#84C1FF
#97CBFF
#ACD6FF
#C4E1FF
#D2E9FF
#ECF5FF
    """,
    "6":"""
#000079
#000093
#0000C6
#0000C6
#0000E3
#2828FF
#4A4AFF
#6A6AFF
#7D7DFF
#9393FF
#AAAAFF
#B9B9FF
#CECEFF
#DDDDFF
#ECECFF
#FBFBFF
    """,
    "7":"""
#003E3E
#005757
#007979
#009393
#00AEAE
#00CACA
#00E3E3
#00FFFF
#4DFFFF
#80FFFF
#A6FFFF
#BBFFFF
#CAFFFF
#D9FFFF
#ECFFFF
#FDFFFF
    """,
    "8":"""
#006030
#01814A
#019858
#01B468
#02C874
#02DF82
#02F78E
#1AFD9C
#4EFEB3
#7AFEC6
#96FED1
#ADFEDC
#C1FFE4
#D7FFEE
#E8FFF5
#FBFFFD
    """,
    "9":"""
#006000
#007500
#009100
#00A600
#00BB00
#00DB00
#00EC00
#28FF28
#53FF53
#79FF79
#93FF93
#A6FFA6
#BBFFBB
#CEFFCE
#DFFFDF
#F0FFF0
    """,
    "10":"""
#467500
#548C00
#64A600
#73BF00
#82D900
#8CEA00
#9AFF02
#A8FF24
#B7FF4A
#C2FF68
#CCFF80
#D3FF93
#DEFFAC
#E8FFC4
#EFFFD7
#F5FFE8
    """,
    "11":"""
#424200
#5B5B00
#737300
#8C8C00
#A6A600
#C4C400
#E1E100
#F9F900
#FFFF37
#FFFF6F
#FFFF93
#FFFFAA
#FFFFB9
#FFFFCE
#FFFFDF
#FFFFF4
    """,
    "12":"""
#5B4B00
#796400
#977C00
#AE8F00
#C6A300
#D9B300
#EAC100
#FFD306
#FFDC35
#FFE153
#FFE66F
#FFED97
#FFF0AC
#FFF4C1
#FFF8D7
#FFFCEC
    """,
    "13":"""
#844200
#9F5000
#BB5E00
#D26900
#EA7500
#FF8000
#FF9224
#FFA042
#FFAF60
#FFBB77
#FFC78E
#FFD1A4
#FFDCB9
#FFE4CA
#FFEEDD
#FFFAF4
    """,
    "14":"""
#642100
#842B00
#A23400
#BB3D00
#D94600
#F75000
#FF5809
#FF8040
#FF8F59
#FF9D6F
#FFAD86
#FFBD9D
#FFCBB3
#FFDAC8
#FFE6D9
#FFF3EE
    """,
    "15":"""
#613030
#743A3A
#804040
#984B4B
#AD5A5A
#B87070
#C48888
#CF9E9E
#D9B3B3
#E1C4C4
#EBD6D6
#F2E6E6
    """,
    "16":"""
#616130
#707038
#808040
#949449
#A5A552
#AFAF61
#B9B973
#C2C287
#CDCD9A
#D6D6AD
#DEDEBE
#E8E8D0
    """,
    "17":"""
#336666
#3D7878
#408080
#4F9D9D
#5CADAD
#6FB7B7
#81C0C0
#95CACA
#A3D1D1
#B3D9D9
#C4E1E1
#D1E9E9
    """,
    "18":"""
#484891
#5151A2
#5A5AAD
#7373B9
#8080C0
#9999CC
#A6A6D2
#B8B8DC
#C7C7E2
#D8D8EB
#E6E6F2
#F3F3FA
    """,
    "19":"""
#6C3365
#7E3D76
#8F4586
#9F4D95
#AE57A4
#B766AD
#C07AB8
#CA8EC2
#D2A2CC
#DAB1D5
#E2C2DE
#EBD3E8
    """
}

def color_extract(image):
    if isinstance(image, Image.Image):
        image = np.array(image)
    image = cv2.resize(image, None, None, fx=1/25, fy=1/25)
    h, w, channel = image.shape
    image = image.reshape((w * h, channel))
    k = 2
    estimator = KMeans(n_clusters=k, max_iter=200, init="k-means++", n_init=25)
    estimator.fit(image)
    centroids = estimator.cluster_centers_
    if len(centroids) > 1:
      white_idx = np.argmax(np.sum(centroids, axis=1))
      main_colors = np.delete(centroids, white_idx, axis=0)
      main_color = main_colors[0]
    else:
      main_color = centroids[0]
    return main_color

def get_info(data_path="color_data.json"):
    with open(data_path, encoding="utf-8") as f:
        data = json.load(f)

    color_meta = {}
    for sku_id, color_id in data.items():
        if color_id not in color_meta:
            color_meta[color_id] = []

        color_meta[color_id].append(sku_id)

    return color_meta, data

def hex_to_rgb(hex_color):
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return np.array([r, g, b])

def rgb_to_hex(r, g, b):
    return "#{:02x}{:02x}{:02x}".format(r, g, b)

def read_color_table():
    color_table2 = {}
    for color_id, color_str in color_table.items():
        colors = color_str.split("\n")[1:-1]
        for color in colors:
            color_table2[color] = color_id

    color_list = []
    for color_str in color_table2:
        color_hex = color_str.split("#")[1]
        color = hex_to_rgb(color_hex)
        color_list.append(color)

    color_list = np.array(color_list)

    return color_table2, color_list

def find_color_idx(color, color_list):
    idx = np.argmin(np.sum(np.abs(color-color_list), axis=1))
    return idx

def get_filelist(path):
    Filelist = []
    for home, dirs, files in os.walk(path):
        for filename in files:
            Filelist.append(os.path.join(home, filename))
    return Filelist

# def make_color_meta(root1="../../images_with_cates", root2="../../细化标注/images_with_cates", root3="../../细化标注/Q1细化标注"):
# def make_color_meta(root1="", root2="/export2/face/code/PTM/aigc/家部家装/7wskus/add_images_0123_with_cate/", root3=""):
# def make_color_meta(root1="", root2="/home/shanxinyuan/common_disk/code/PTM/jiabu/0224_跑批/2月生产素材_透底图细化标签汇总/", root3="/home/shanxinyuan/common_disk/code/PTM/jiabu/0224_跑批/2月生产素材_透底图细化标签补充/"):

# 202403-202404
# def make_color_meta(root1="", root2="/home/shanxinyuan/common_disk/data/jiabu/20240322/images/", root3=""):
def make_color_meta(root1="", root2="/home/shanxinyuan/common_disk/data/jiabu/20240521/images/", root3=""):
    color_table2, color_list = read_color_table()
    '''
    paths = glob(osp.join(root3, "*", "*.png")) + \
            glob(osp.join(root2, "*", "*.png")) + \
            glob(osp.join(root1, "*", "*.png"))
    '''
    ## add by sxy
#     skus_list = []
#     with open("no_color_info_skus.txt") as f:
#         cons = f.readlines()
#         for con in cons:
#             skus_list.append(con.strip())
#     print(len(skus_list))

    paths = get_filelist(root2) + get_filelist(root3)
    print(len(paths))
    rcolor_meta = {}
    for path in tqdm(paths):
        if not path.endswith(".jpg") and not path.endswith(".png"):
            continue
        sku_id = osp.basename(path)[:-4]
#         if sku_id not in set(skus_list):
#             continue
        if sku_id in rcolor_meta:
            continue
        print(sku_id)
        try:
            image = Image.open(path).convert("RGBA")
            color = color_extract(image)[None, :3]
            idx = find_color_idx(color, color_list)
            key = rgb_to_hex(*color_list[idx]).upper()
            color_id = color_table2[key]
            rcolor_meta[sku_id] = color_id
        except Exception as e:
            print("Skip {path} due to {e}")
    
    with open("color_data_202405.json", "w", encoding="utf-8") as f:
    # with open("color_data_202403.json", "w", encoding="utf-8") as f:
    # with open("color_data_add_part2.json", "w", encoding="utf-8") as f:
    # with open("color_data_v3.json", "w", encoding="utf-8") as f:
        f.write(json.dumps(rcolor_meta, indent=2, ensure_ascii=False))
        
        
        

if __name__ == "__main__":
    make_color_meta()
