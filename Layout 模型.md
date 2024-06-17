
# SD
在推理阶段，stable diffusion 是 随机噪声 + text prompt 后 反向恢复
# 论文阅读
## 领域信息
什么是：Layout？<br />通常指的是仅预测出每一张图片几个元素的最佳的相对位置（type+bounding boxes)。也有一些论文是生成图片。<br />![image.png](https://cdn.nlark.com/yuque/0/2024/png/29540509/1717987411829-a45745c0-e2ee-4c9e-b445-2d055a4a6726.png#averageHue=%23a8a6a2&clientId=u7e84b3ce-c973-4&from=paste&height=442&id=u389f0655&originHeight=884&originWidth=1182&originalType=binary&ratio=2&rotation=0&showTitle=false&size=473973&status=done&style=none&taskId=uc62aa97e-3caa-4f6b-8d33-c480c4c862a&title=&width=591)<br />这类方法的主要任务可以分成:<br />![image.png](https://cdn.nlark.com/yuque/0/2024/png/29540509/1717987759964-760a1362-3c41-4603-acf0-3718271b00f4.png#averageHue=%23e2e1df&clientId=u7e84b3ce-c973-4&from=paste&height=212&id=ucb2aabfe&originHeight=424&originWidth=918&originalType=binary&ratio=2&rotation=0&showTitle=false&size=83271&status=done&style=none&taskId=u0c2c8a74-4b95-4c5b-aaf7-000b3e51ce0&title=&width=459)<br />1.仅提供标签，生成每一个元素的坐标<br />2.标签+大小（长宽）， 生成每一个元素的坐标<br />3.标签+ 语言描述相对位置等信息， 生成每一个元素的坐标<br />4.对当前的标签+坐标做更合理的布局，生成每一个元素的坐标<br />5.给定一个元素的标签+坐标，填充其他元素，生成合适元素类型+坐标<br />6.没有提供任何内容，生成合适元素类型+坐标<br />其实本质来说，这是CV任务用NLP方法去做：<br />![image.png](https://cdn.nlark.com/yuque/0/2024/png/29540509/1717988290996-cbf138b5-46f0-4377-95ad-589c93baeef0.png#averageHue=%23e3e3e3&clientId=u7e84b3ce-c973-4&from=paste&height=215&id=ue7e3537a&originHeight=430&originWidth=706&originalType=binary&ratio=2&rotation=0&showTitle=false&size=73299&status=done&style=none&taskId=u8226eb17-f599-4913-8c7c-edea7d88d82&title=&width=353)

## LayerOutVAE
## LayerGAN
![image.png](https://cdn.nlark.com/yuque/0/2024/png/29540509/1717987440922-393e80ee-3837-4027-8ed1-ac3ee42ccc40.png#averageHue=%23f2f0ee&clientId=u7e84b3ce-c973-4&from=paste&height=397&id=ue832a3bd&originHeight=794&originWidth=1396&originalType=binary&ratio=2&rotation=0&showTitle=false&size=581983&status=done&style=none&taskId=u1e9f04da-1704-46f7-a4e5-d135db69d00&title=&width=698)<br />**如果有时间，可以看看这篇了论文**（有必要！）<br />这篇论文比较疑惑的是：<br />为什么输入是：随机每个元素都有一组几何参数θ和一个类概率向量p。这是训练过程吗？<br />如果涉及到不同风格的照片，他是怎么根据不同类别的标签的信息进行生成的？<br />**输入：均匀分布和高斯分布中采样的具有随机类概率和几何参数的一组图形元素**
## LayerOutGAN++
Constrained Graphic Layout Generation via Latent Optimization<br />布局对图像设计和场景生成非常重要。作者提出了一种基于LayoutGAN 的[生成对抗网络](https://so.csdn.net/so/search?q=%E7%94%9F%E6%88%90%E5%AF%B9%E6%8A%97%E7%BD%91%E7%BB%9C&spm=1001.2101.3001.7020)，称为Layout GAN++，它通过建模不同类型的2D元素的几何关系来综合**布局**。<br />主要内容是：<br />算法的框架：<br />![image.png](https://cdn.nlark.com/yuque/0/2024/png/29540509/1717987484426-f9ea68aa-4217-4bb7-afc6-92416c40682f.png#averageHue=%23f3f0ef&clientId=u7e84b3ce-c973-4&from=paste&height=395&id=u71969edd&originHeight=790&originWidth=2516&originalType=binary&ratio=2&rotation=0&showTitle=false&size=531535&status=done&style=none&taskId=u1239ca11-3b25-4ba3-a3d7-1e6b7f0a76d&title=&width=1258)<br />G 的输入： 噪音 + labels 输出：bounding boxes B<br />D : generated bounding boxes 𝐵 and conditional labels 𝐿.输出是一个标量值，它量化了布局的真实性，并尝试从内部表示中重建给定的边界框，输出的向量还是比较长的<br />Aux : 作者经验发现，在文档对齐良好的不居中，判断别被训练为对对齐的敏感，并且对位置的趋势不太敏感，就只是关系物体是否对齐，但是不关心不寻常的布局。<br />这里的**对齐**是什么意思？ 就是某个物体是否在画面中出现过吗？<br />疑问：<br />1.输入是： 随机噪声+ 不同模态的数据的labels(a conditional multiset of labels L )？ （why??）

1. 这个他没提，需要看代码，大概率不是同一个？？？
2. 我认为也不是同一个？？？

4.这个需要进一步了解？？？<br />5.这里是不是判断输入的是G生成的 还是 直接编码的 ？ 貌似 不太对<br />**实验的评估指标：**<br />[FID](https://www.baidu.com/s?wd=FID&rsv_idx=2&tn=baiduhome_pg&usm=1&ie=utf-8&rsv_pq=a954b521000e13d6&oq=FID%20%E6%8C%87%E6%A0%87&rsv_t=8becgk7WWFWoceInhWifIVJldggef%2BbPwawB90TKMB26aetWyWDqWjOjtmyjp7YKVNE1&sa=re_dqa_zy&icon=1)[](https://www.baidu.com/s?wd=FID&rsv_idx=2&tn=baiduhome_pg&usm=1&ie=utf-8&rsv_pq=a954b521000e13d6&oq=FID%20%E6%8C%87%E6%A0%87&rsv_t=8becgk7WWFWoceInhWifIVJldggef%2BbPwawB90TKMB26aetWyWDqWjOjtmyjp7YKVNE1&sa=re_dqa_zy&icon=1)_（Fréchet Inception Distance）是一种用于评估生成模型和真实数据分布之间差异的指标 如何即可算：_2个分布之间的**Fréchet距离**来衡量[生成模型](https://so.csdn.net/so/search?q=%E7%94%9F%E6%88%90%E6%A8%A1%E5%9E%8B&spm=1001.2101.3001.7020)和真实数据分布之间的差异。Fréchet距离是一种度量两个分布之间距离的方法，它考虑到了两个分布的均值和[协方差矩阵](https://so.csdn.net/so/search?q=%E5%8D%8F%E6%96%B9%E5%B7%AE%E7%9F%A9%E9%98%B5&spm=1001.2101.3001.7020)，可以更好地描述两个分布之间的差异。<br />FID^2 = ||\mu_1 - \mu_2||^2 + Tr(\Sigma_1 + \Sigma_2 - 2(\Sigma_1\Sigma_2)^{1/2})$<br />Maximum IoU: 两个生成的布局和参考集合之间定义。<br />Alignment and overlap ：这个指标是他们之前的工作提出来的，可以看看他们的LayoutGAN
## LayoutDiffusion
LayoutDiffusion: Improving Graphic Layout Generation by Discrete Diffusion Probabilistic Models<br />**这个方法跟LayerGAN 有什么不同？**<br />![image.png](https://cdn.nlark.com/yuque/0/2024/png/29540509/1717987523351-19bc3dfd-8f14-4d03-b037-4be3122e3e25.png#averageHue=%23f1ebe7&clientId=u7e84b3ce-c973-4&from=paste&height=441&id=u02f6c321&originHeight=882&originWidth=2540&originalType=binary&ratio=2&rotation=0&showTitle=false&size=924810&status=done&style=none&taskId=u1daf7a6f-6c7b-4966-b3c3-9c824aa430e&title=&width=1270)

### 主要思想：
采用了diffusion的方法：<br />1.forward process：坐标被轻微破坏为平稳分布， type tokens在**后期**被吸收到MASK中。<br />2.reverse process: types 首先被恢复，然后对坐标逐渐细化<br />这个过程是坐标细化的过程，对于其他任务，可以**控制输入**的不同来实现不同的任务，因为他们最终的输出都是一样的。

### 主要贡献：
1.设计的算法只是能够在 type elements 内部 或者 坐标内部之间转换<br />2.设计了近端转换，更倾向于转换到相邻坐标<br />3.针对type，设计了所有元素有一定概率到mask转换，且限制在diffusion最后一个阶段。<br />4.即插即用，即控制反向之前的输入信息，即可完成不同的任务。


### 模型详情
在反向过程中，模型是Transformer，模型输入的数据构造是： xt的编码向量，位置编码，时间t编码


从training set's prior distribution 中sampling 一个n，<br />对于构造xT，我们为类型分配MASK标记，并为前n个元素的边界框分配随机坐标标记。对于剩余的(N−n)个元素，使用PAD令牌来确保长度一致.
### 代码解读

确定training_data，也就是TextDataset 
```python
###- 模型的输入： training_data 
def get_corpus_rocstory(data_args, model, seq_length, padding_mode='block',
                        split='train', load_vocab=None):
    print("_______________________________")
    print(result_train_lst[:1])
    return {'train': result_train_lst}, model
    [{'input_ids': [0, 8, 20, 24, 125, 69, 4, 8, 20, 77, 125, 84, 4, 6, 20, 
                    88, 71, 129, 4, 6, 75, 88, 125, 129, 4, 6, 41, 21, 
                    104, 23, 4, 6, 20, 73, 125, 77, 1, 3, 3, 3, 3, 3, 3, 3, 3, 
                    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 
                    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 
                    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 
                    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 
    'hidden_states': [[-1.6222010850906372, 1.7222366333007812, 0.8530371785163879, 0.0408187098801136, -0.5346421003341675, 0.8283897042274475, -2.47684907913208, -0.3852204382419586], 


if task_mode == 'e2e-tgt':
    training_data, model = get_corpus_rocstory(data_args, model, seq_length,
                    padding_mode=padding_mode, split=split,
                    load_vocab=load_vocab)
'''
src1_valid.txt 三个文件， 然后分词，统计词频，取词频大于一定阈值（我设置了0)的东西，然后保存到 vocab.json
        '''
### 下面的self.text_datasets 就是  上面的training_data
def __getitem__(self, idx):

    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    if self.model_arch != 'conv-unet':
        arr = np.array(self.text_datasets['train'][idx]['hidden_states'],
                       dtype=np.float32)
        if self.eigen_transform  is not None:
            old_shape = arr.shape
            # arr = arr.reshape(1, -1) @ self.eigen_transform
            arr = arr.reshape(1, -1) - self.eigen_transform['mean']
            arr = arr @ self.eigen_transform['map']
            arr = arr.reshape(old_shape)
            
        if hasattr(self.data_args, 'noise_level') and self.data_args.noise_level > 0:
            # print(arr.dtype)
            # print(self.data_args.noise_level, 'using the noise level.')
            arr = arr + self.data_args.noise_level * np.random.randn(*arr.shape).astype(arr.dtype)
            # print(arr.dtype)

        out_dict = {}
        out_dict['input_ids'] = np.array(self.text_datasets['train'][idx]['input_ids'])
        # out_dict['mapping_func'] = self.mapping_func
        if self.data_args.experiment_mode == 'conditional_gen':
            out_dict['src_ids'] = np.array(self.text_datasets['train'][idx]['src_ids'])
            out_dict['src_mask'] = np.array(self.text_datasets['train'][idx]['src_mask'])
        # if self.local_classes is not None:
        #     out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
        # print(arr[0],'arr0')
        return arr, out_dict
 batch, types = next(data) ###- 相当于是一个data_loader ，dataset是 TextDataset，相当于arr, out_dict
```
#### 训练阶段
train.py
```python
model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
model2, tokenizer = load_models(args.modality, args.experiment, args.model_name_or_path, args.in_channel,
                                    args.checkpoint_path, extra_args=args)
 data = load_data_text(...)  ###- 这里的data 是否可以根据任务区调整，这个是关键？
TrainLoop(...).run_loop()
```

#### 推理阶段
batch_decoded.py<br />model : DiscreteTransformerModel （69）
```python
    # load configurations.
config_path = os.path.join(os.path.split(args.model_path)[0], "training_args.json")

model, diffusion = create_model_and_diffusion(
    **args_to_dict(args, model_and_diffusion_defaults().keys())
)
   model2, tokenizer = load_models(args.modality, args.experiment, args.model_name_or_path, args.in_channel,
                                    os.path.split(args.model_path)[0])




###  model 在这里用做推理，同时对于不同的任务的输入做了处理。
sample = sample_fn(
    model,
    sample_shape,
    sample_start_step=(T_refine if args.constrained=='refine' else args.diffusion_steps),
    content_token=(model_kwargs['y'] if args.constrained is not None else None),
    multistep=args.multistep,
    constrained=args.constrained,
)
# sample 在哪里用：
#sample from  sample_fn  ---> gathered_samples,   all_images ---> arrs  ---> arr ---> word_lst

# 将word_lst 有关的内容：会执行：'written the decoded output to ？ 会的：
# written the decoded output to ../results/generation_outputs/publaynet_lex_pretrained\ungen\publaynet_lex_pretrained.ema_0.9999_400000.pt.samples_-1.0_elem1.txt
# written the decoded output to ../results/generation_outputs/publaynet_lex_pretrained\ungen\publaynet_lex_pretrained.ema_0.9999_400000.pt.samples_-1.0_elem1.json
```


**model 的输入与输出：**
```python
print("============ model input and output============")
# if flag == 1 :
#     print(log_z) # torch.Size([20, 139, 121]) torch.Size([20, 139, 121])  batch , 139 = 128+5+5 ,121 是tensor长度
#     print(log_x_recon)
#     flag = 0 
#     return 
'''
tensor([[[  0.0000, -69.0776, -69.0776,  ..., -69.0776, -69.0776, -69.0776],
[-69.0776, -69.0776, -69.0776,  ..., -69.0776, -69.0776, -69.0776],
[-69.0776, -69.0776, -69.0776,  ..., -69.0776, -69.0776, -69.0776],
...,
[-69.0776, -69.0776, -69.0776,  ..., -69.0776, -69.0776, -69.0776],
[-69.0776, -69.0776, -69.0776,  ..., -69.0776, -69.0776, -69.0776],
[-69.0776,   0.0000,   0.0000,  ..., -69.0776, -69.0776, -69.0776]]],
device='cuda:0')
tensor([[[-1.2212e-14, -2.7718e+01, -2.5994e+01,  ..., -2.5921e+01,
    -2.5921e+01, -2.5921e+01],
    [-4.0051e+01, -2.5973e+01, -2.4939e+01,  ..., -2.7537e+01,
    -2.7537e+01, -2.7537e+01],
    [-3.5370e+01, -2.8573e+01, -3.3060e+01,  ..., -3.3323e+01,
    -3.3323e+01, -3.3323e+01],
    ...,
    [-3.4353e+01, -2.1478e+01, -1.9007e+01,  ..., -2.5094e+01,
    -2.5094e+01, -2.5094e+01],
    [-4.1782e+01, -1.8795e+01, -1.5572e+01,  ..., -3.2441e+01,
    -3.2441e+01, -3.2441e+01],
    [-7.0000e+01, -7.0000e+01, -7.0000e+01,  ..., -7.0000e+01,
    -7.0000e+01, -7.0000e+01]]], device='cuda:0')
    
'''
```
**vocab:**
```python
load_vocab {'START': 0, 'END': 1, 'UNK': 2, 'PAD': 3, '|': 4, 'figure': 5, 'text': 6, 'title': 7, 'table': 8, 'list': 9, '0': 10, '1': 11, '2': 12, '3': 13, '4': 14, '5': 15, '6': 16, '7': 17, '8': 18, '9': 19, '10': 20, '11': 21, '12': 22, '13': 23, '14': 24, '15': 25, '16': 26, '17': 27, '18': 28, '19': 29, '20': 30, '21': 31, '22': 32, '23': 33, '24': 34, '25': 35, '26': 36, '27': 37, '28': 38, '29': 39, '30': 40, '31': 41, '32': 42, '33': 43, '34': 44, '35': 45, '36': 46, '37': 47, '38': 48, '39': 49, '40': 50, '41': 51, '42': 52, '43': 53, '44': 54, '45': 55, '46': 56, '47': 57, '48': 58, '49': 59, '50': 60, '51': 61, '52': 62, '53': 63, '54': 64, '55': 65, '56': 66, '57': 67, '58': 68, '59': 69, '60': 70, '61': 71, '62': 72, '63': 73, '64': 74, '65': 75, '66': 76, '67': 77, '68': 78, '69': 79, '70': 80, '71': 81, '72': 82, '73': 83, '74': 84, '75': 85, '76': 86, '77': 87, '78': 88, '79': 89, '80': 90, '81': 91, '82': 92, '83': 93, '84': 94, '85': 95, '86': 96, '87': 97, '88': 98, '89': 99, '90': 100, '91': 101, '92': 102, '93': 103, '94': 104, '95': 105, '96': 106, '97': 107, '98': 108, '99': 109, '100': 110, '101': 111, '102': 112, '103': 113, '104': 114, '105': 115, '106': 116, '107': 117, '108': 118, '109': 119, '110': 120, '111': 121, '112': 122, '113': 123, '114': 124, '115': 125, '116': 126, '117': 127, '118': 128, '119': 129, '120': 130, '121': 131, '122': 132, '123': 133, '124': 134, '125': 135, '126': 136, '127': 137}
```
**为什么要这样做？**<br />既然 标签没有加噪，但是为什么后来生成的内容很乱？ 请你弄清楚每个阶段的后输入与输出<br />1.这四个数字代表的是bounding boxs 的 框的信息<br />left li, top ti, right ri, and bottom bi coordinates<br />2.input 是什么？<br />3.mask ？<br />4.不同的符号表示什么？ 他表示加入不同程度的噪音<br />这三个Q为什么是矩阵？<br />该实验的指标是什么？<br />该实验的主要思想是什么？<br />与 stable diffusion的区别？<br />与LayerOUtGAN 的区别？<br />知识回顾：<br />SD: 正向加噪，反向生图<br />正向过程一般不训练模型，反向过程需要训练噪音生成器。<br />正向数据集：<br />正向过程：<br />input: 为大量不同噪点强度级别的图像 (image + 一个级别的噪音）<br />output: 该噪音图像对应 的 级别噪音<br />这样可以构成： 数据对，但是并不进行训练<br />反向过程：<br />实际在过程中，并不一定是从一张完全噪音图片演变成无噪音图片。<br />![image.png](https://cdn.nlark.com/yuque/0/2024/png/29540509/1717987549159-7c787e90-168d-48c0-8964-e4234683f044.png#averageHue=%2342403a&clientId=u7e84b3ce-c973-4&from=paste&height=335&id=ub35173ab&originHeight=670&originWidth=1192&originalType=binary&ratio=2&rotation=0&showTitle=false&size=764126&status=done&style=none&taskId=u95b3654c-5888-422a-9a12-5848e51429b&title=&width=596)<br />所以扩散过程的风格，完全取决于在正向过程中_U-Net，也就是训练数据集的风格。_
## LayoutDiffusion
还有一篇国内版的，与上述是同期工作。<br />LayoutDiffusion: Controllable Diffusion Model for Layout-to-image Generation
# 实验
![[https://apijoyspace.jd.com/v1/files/XOUQgRmQG4h4nXllimZy/link](https://apijoyspace.jd.com/v1/files/XOUQgRmQG4h4nXllimZy/link)](SD+a3438060-e2d4-4c9a-9736-a72a272b75e1/link 5)<br />labelme: 这个分别是左上，右下坐标<br />文心一言的解释：<br />bboxes是一个包含多个边界框（bounding boxes）的tensor。每个边界框由四个数值表示，通常这些数值分别代表：

1. **x_min**(或称为xmin或left)：边界框左上角的x坐标（水平方向的起始位置）。
2. **y_min**(或称为ymin或top)：边界框左上角的y坐标（垂直方向的起始位置）。
3. **width**：边界框的宽度。
4. **height**：边界框的高度。

这里表示什么意思？ 怎么计算？ 看论文？<br />![[https://apijoyspace.jd.com/v1/files/dLP5ewwzdRDr4QZE8fRi/link](https://apijoyspace.jd.com/v1/files/dLP5ewwzdRDr4QZE8fRi/link)](SD+a3438060-e2d4-4c9a-9736-a72a272b75e1/link 6)<br />training set's prior distribution. F<br />python scripts/batch_decode.py ../results/checkpoint/rico_1 -1.0 ema 20 3728 False -1 type<br />python scripts/train.py --checkpoint_path ../results/checkpoint/rico_2 --model_arch transformer --modality e2e-tgt --save_interval 500 --lr 1e-5 --batch_size 4 --diffusion_steps 200 --noise_schedule gaussian_refine_pow2.5 --use_kl False --learn_sigma False --aux_loss True --rescale_timesteps False --seq_length 121 --num_channels 128 --seed 102 --dropout 0.1 --padding_mode pad --experiment random --lr_anneal_steps 20000 --weight_decay 0.0 --predict_xstart True --training_mode discrete1 --vocab_size 186 --submit False --e2e_train ../data/processed_datasets/RICO_ltrb_lex --alignment_loss False<br />python eval_src/tools/draw_from_results.py -d rico -p results/generation_outputs/rico_1/ungen/processed.pt -s results/generation_outputs/rico_1/ungen/pics -n 100
