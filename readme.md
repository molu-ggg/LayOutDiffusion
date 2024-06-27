

# Layout diffusion

适配卧室数据集(350+图片场景，52类标签)，在代码中代号为"rico" 

## 数据集情况
    350+ 数据集 按照 train:val:test = 7:1.5:1.5 的比例划分，并且保证了test和val的所有标签在train中包含有
    
    最后得到的train 有270张，val有36张，test有50张场景
## 训练与测试
### 训练数据命令：

```cmd
python scripts/train.py --checkpoint_path ../results/checkpoint/rico_5 --model_arch transformer --modality e2e-tgt --save_interval 5000 --lr 1e-5 --batch_size  4  --diffusion_steps  200  --noise_schedule gaussian_refine_pow2.5  --use_kl False --learn_sigma False  --aux_loss True --rescale_timesteps False --seq_length 121 --num_channels 128 --seed 102 --dropout 0.1 --padding_mode pad --experiment random  --lr_anneal_steps 200000 --weight_decay 0.0 --predict_xstart True --training_mode discrete1 --vocab_size 186 --submit False --e2e_train ../data/processed_datasets/RICO_ltrb_lex --alignment_loss False

```



### 测试命令：

在 推理阶段有'type' 和 'ungen' 两种类型

type: 仅仅输入标签，预测坐标

ungen: 随机生成标签和坐标

```cmd
python scripts/batch_decode.py ../results/checkpoint/rico_5 -1.0 ema 20 50 False -1 type

python json2metrics.py ./results/generation_outputs/rico_5/type/rico_5.ema_0.9999_035000.pt.samples_-1.0_elem1.json

python eval_src/tools/draw_from_results.py -d rico -p results/generation_outputs/rico_5/type/processed.pt -s results/generation_outputs/rico_5/type/pics -n 100

```


## 源README
跳转这个文件 ![analysis](ygq/README.md)

## 代码解析
跳转这个文件 ![analysis](ygq/analysis.md)


