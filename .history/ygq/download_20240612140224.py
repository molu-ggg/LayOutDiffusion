import os
# 注意os.environ得在import huggingface库相关语句之前执行。
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from huggingface_hub import hf_hub_download

def download_model(source_url,local_dir):

    # 使用huggingface原地址
    # source_url ="https://huggingface.co/BlinkDL/rwkv-4-novel/blob/main/RWKV-4-Novel-7B-v1-ChnEng-20230426-ctx8192.pth"
    # 使用huggingface-镜像地址
    # source_url = "https://hf-mirror.com/BlinkDL/rwkv-4-novel/blob/main/RWKV-4-Novel-7B-v1-ChnEng-20230426-ctx8192.pth"

    if 'blob' in source_url:
        sp = '/blob/main/'
    else:
        sp = '/resolve/main/'

    if 'huggingface.co' in source_url:
        url = 'https://huggingface.co/'
    else:
        url = 'https://hf-mirror.com'

    location = source_url.split(sp)
    repo_id = location[0].strip(url) # 仓库ID，例如："BlinkDL/rwkv-4-world" 

        cache_dir = local_dir + "/cache"
    filename = location[1]# 大模型文件，例如："RWKV-4-World-CHNtuned-7B-v1-20230709-ctx4096.pth"

    print(f'开始下载\n仓库：{repo_id}\n大模型：{filename}\n如超时不用管，会自定继续下载，直至完成。中途中断，再次运行将继续下载。')
    while True:   
        try:
            hf_hub_download(cache_dir=cache_dir,
            local_dir=local_dir,
            repo_id=repo_id,
            filename=filename,
            local_dir_use_symlinks=False,
            resume_download=True,
            etag_timeout=100
            )
        except Exception as e :
            print(e)
        else:
            print(f'下载完成，大模型保存在：{local_dir}\{filename}')
            break
            
if __name__ == '__main__':
    source_url = "https://huggingface.co/BlinkDL/rwkv-4-novel/blob/main/RWKV-4-Novel-7B-v1-ChnEng-20230426-ctx8192.pth"
    source_url = "https://hf-mirror.com/BlinkDL/rwkv-4-novel/blob/main/RWKV-4-Novel-7B-v1-ChnEng-20230426-ctx8192.pth"
    # source_url = "https://huggingface.co/BlinkDL/rwkv-5-world/resolve/main/RWKV-5-World-1B5-v2-20231025-ctx4096.pth"
    download_model(source_url,local_dir = r'D:\ProgramData\RWKV\models')
    
