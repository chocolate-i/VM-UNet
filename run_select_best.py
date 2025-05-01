import os
import torch
from torch.utils.data import DataLoader
import numpy as np
from datasets.dataset import NPY_datasets
from models.vmunet.vmunet import VMUNet
from select_best_samples import select_best_samples
from configs.config_setting import setting_config
from utils import get_logger, set_seed
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='选择最佳分割样本')
    parser.add_argument('--ratio', type=float, default=0.05, help='选择的样本比例')
    parser.add_argument('--metric', type=str, default='dice', choices=['dice', 'iou'], help='评估指标')
    parser.add_argument('--output', type=str, default=None, help='输出目录（默认为results/best_samples）')
    return parser.parse_args()

def main(config):
    # 解析命令行参数
    args = parse_args()
    ratio = args.ratio
    metric = args.metric
    best_samples_dir = args.output if args.output else os.path.join(config.work_dir, 'best_samples/')
    
    # 设置保存目录
    if not os.path.exists(best_samples_dir):
        os.makedirs(best_samples_dir)
    
    # 创建logger
    log_dir = os.path.join(config.work_dir, 'log')
    logger = get_logger('select_best', log_dir)
    
    # GPU设置
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id
    set_seed(config.seed)
    torch.cuda.empty_cache()
    
    # 准备数据集 - 使用测试集或验证集
    print('加载数据集...')
    val_dataset = NPY_datasets(config.data_path, config, train=False)
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        num_workers=config.num_workers
    )
    
    # 加载模型
    print('加载训练好的模型...')
    model_cfg = config.model_config
    model = VMUNet(
        num_classes=model_cfg['num_classes'],
        input_channels=model_cfg['input_channels'],
        depths=model_cfg['depths'],
        depths_decoder=model_cfg['depths_decoder'],
        drop_path_rate=model_cfg['drop_path_rate'],
        load_ckpt_path=model_cfg['load_ckpt_path'],
    )
    model.load_from()
    
    # 加载最佳权重
    checkpoint_dir = os.path.join(config.work_dir, 'checkpoints')
    best_weight_path = os.path.join(checkpoint_dir, 'best.pth')

    # 创建检查点目录（如果不存在）
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
        print(f'创建检查点目录: {checkpoint_dir}')
        logger.info(f'创建检查点目录: {checkpoint_dir}')
    
    # 修改查找逻辑，添加目录存在检查
    best_weight_found = False
    if os.path.exists(best_weight_path):
        best_weight_found = True
    else:
        # 仅当目录存在时才列出目录内容
        if os.path.exists(checkpoint_dir) and os.listdir(checkpoint_dir):
            for file in os.listdir(checkpoint_dir):
                if file.startswith('best-epoch'):
                    best_weight_path = os.path.join(checkpoint_dir, file)
                    best_weight_found = True
                    break
    
    if best_weight_found:
        print(f'加载最佳模型权重: {best_weight_path}')
        best_weight = torch.load(best_weight_path, map_location=torch.device('cpu'))
        model.load_state_dict(best_weight)
    else:
        print('未找到训练好的模型权重，使用初始化模型')
    
    model = model.cuda()
    model.eval()
    
    # 选择最佳样本
    print('选择分割效果最佳的样本...')
    best_samples = select_best_samples(
        model, 
        val_loader, 
        config, 
        best_samples_dir, 
        ratio=ratio,  
        metric_name=metric  # 修改为metric_name
    )
    
    print(f'已完成! 选择了 {len(best_samples)} 个最佳样本并保存到 {best_samples_dir}')
    logger.info(f'已完成! 选择了 {len(best_samples)} 个最佳样本并保存到 {best_samples_dir}')

if __name__ == '__main__':
    config = setting_config
    main(config)