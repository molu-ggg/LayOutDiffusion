键值对：这是pond的：
```python
{"START": 0, "END": 1, "UNK": 2, "PAD": 3, "|": 4, "figure": 5, "text": 6, "title": 7, "table": 8, "list": 9, "0": 10, "1": 11, "2": 12, "3": 13, "4": 14, "5": 15, "6": 16, "7": 17, "8": 18, "9": 19, "10": 20, "11": 21, "12": 22, "13": 23, "14": 24, "15": 25, "16": 26, "17": 27, "18": 28, "19": 29, "20": 30, "21": 31, "22": 32, "23": 33, "24": 34, "25": 35, "26": 36, "27": 37, "28": 38, "29": 39, "30": 40, "31": 41, "32": 42, "33": 43, "34": 44, "35": 45, "36": 46, "37": 47, "38": 48, "39": 49, "40": 50, "41": 51, "42": 52, "43": 53, "44": 54, "45": 55, "46": 56, "47": 57, "48": 58, "49": 59, "50": 60, "51": 61, "52": 62, "53": 63, "54": 64, "55": 65, "56": 66, "57": 67, "58": 68, "59": 69, "60": 70, "61": 71, "62": 72, "63": 73, "64": 74, "65": 75, "66": 76, "67": 77, "68": 78, "69": 79, "70": 80, "71": 81, "72": 82, "73": 83, "74": 84, "75": 85, "76": 86, "77": 87, "78": 88, "79": 89, "80": 90, "81": 91, "82": 92, "83": 93, "84": 94, "85": 95, "86": 96, "87": 97, "88": 98, "89": 99, "90": 100, "91": 101, "92": 102, "93": 103, "94": 104, "95": 105, "96": 106, "97": 107, "98": 108, "99": 109, "100": 110, "101": 111, "102": 112, "103": 113, "104": 114, "105": 115, "106": 116, "107": 117, "108": 118, "109": 119, "110": 120, "111": 121, "112": 122, "113": 123, "114": 124, "115": 125, "116": 126, "117": 127, "118": 128, "119": 129, "120": 130, "121": 131, "122": 132, "123": 133, "124": 134, "125": 135, "126": 136, "127": 137}
138 MASK 
```



## 代码解读
```python
training_data ,model = get_corpus_rocstory(....)
    return result_train_lst,model
dataset = TextDataset(
    training_data,
    seq_length,
    data_args,
    model_arch=data_args.model_arch,
    shard=MPI.COMM_WORLD.Get_rank(),
    num_shards=MPI.COMM_WORLD.Get_size(),
)
TextDataset __getitem__:
    return arr, out_dict
    # arr = np.array(self.text_datasets['train'][idx]['hidden_states'],dtype=np.float32)
    #out_dict : out_dict['input_ids'] = np.array(self.text_datasets['train'][idx]['input_ids'])

# 'hidden_states' 的来源：
model = torch.nn.Embedding(len(vocab_dict), data_args.in_channel) len(vocab_dict),8
hidden_state = model(torch.tensor(input_ids))
```
result_train_lst：
```python
[{'input_ids': [0, 8, 20, 24, 125, 69, 4, 8, 20, 77, 125, 84, 4, 6, 20, 88, 71, 129, 4, 6, 75, 88, 125, 129, 4, 6, 41, 21, 104, 23, 4, 6, 20, 73, 125, 77, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 'hidden_states': [[-1.6222010850906372, 1.7222366333007812, 0.8530371785163879, 0.0408187098801136, -0.5346421003341675, 0.8283897042274475, -2.47684907913208, -0.3852204382419586], [0.5329328775405884, 1.5567169189453125, 0.15599234402179718, 1.5328624248504639, -1.3395750522613525, 0.09910503029823303, -0.42658814787864685, -0.4373644292354584], [1.2780152559280396, -0.028085479512810707,

 (121,)
 (121,8) 
 
```
![hidden_states](../improved-diffusion/improved_diffusion/text_datasets.py#L287)

model = torch.nn.Embedding(len(tokenizer), emb_dim) 

train : ![train](../improved-diffusion/scripts/train.py#L148) Line 148 
batch, cond = next(self.data)


## 时间步的采样：
![schedule_sampler](../improved-diffusion/improved_diffusion/train_util.py#L277) 277 行
self.schedule_sampler="uniform",
UniformSampler的策略非常简单：它为每个时间步长分配相同的权重。这意味着在采样过程中，每个时间步长被选中的概率是相等的
![UniformSampler](../improved-diffusion/improved_diffusion/resample.py#L61) # 也有可能跟Q有关系

## noise的生成

![training_losses](../improved-diffusion/improved_diffusion/discrete_diffusion.py#L763) 


噪音采样，从$q(xt|x0)$ 中采样，概率分布为：$q(xt|x0)= cat(x_t;Q_tx_{t-1})$ where Cat(x_t; Q_tx_{t-1}) is a categorical distribution over the one-hot row vector $x_t$ with probabilities given by the row vector $Q_tx_{t-1}$.

噪音采样方法：Gumbel-Max 采样方法: https://cloud.tencent.com/developer/article/1901340 

![公式9,概率分布中参数随着时间t的选择](../improved-diffusion/improved_diffusion/discrete_diffusion.py)
    at,at1, bt1,bt2, ct,ct1, att,att1, btt1,btt2, ctt,ctt1 = alpha_schedule(self.num_timesteps, N=self.num_classes-1, matrix_policy=1) gaussian_refine_pow2.5 

![什么时候 Mask 阶段 ](../improved-diffusion/improved_diffusion/discrete_diffusion.py)

        
![计算Qcoord](../improved-diffusion/improved_diffusion/discrete_diffusion.py#L50) 计算Q


  
  




