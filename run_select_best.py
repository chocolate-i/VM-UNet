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
    parser.add_argument('--work_dir', type=str, help='训练好的模型所在工作目录')
    return parser.parse_args()

def main(config):
    # 解析命令行参数
    args = parse_args()
    ratio = args.ratio
    metric = args.metric
    
    # 如果指定了work_dir，则更新config.work_dir
    if args.work_dir:
        config.work_dir = args.work_dir
        
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

    # 检查是否已包含checkpoints路径
    if 'checkpoints' in checkpoint_dir and checkpoint_dir.endswith('checkpoints/checkpoints'):
        # 修复嵌套路径问题
        checkpoint_dir = checkpoint_dir.replace('/checkpoints/checkpoints', '/checkpoints')
        best_weight_path = os.path.join(checkpoint_dir, 'best.pth')
    
    print(f'尝试从以下路径加载模型: {checkpoint_dir}')

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
        
        # 过滤掉额外的键
        keys_to_remove = []
        for key in list(best_weight.keys()):
            if "total_ops" in key or "total_params" in key:
                keys_to_remove.append(key)
        
        if keys_to_remove:
            print(f"正在移除{len(keys_to_remove)}个统计信息键...")
            for key in keys_to_remove:
                del best_weight[key]
        
        # 调试代码：打印模型期望的键和权重文件中的键
        model_keys = set(model.state_dict().keys())
        weight_keys = set(best_weight.keys())
        print(f"额外的键: {weight_keys - model_keys}")
        print(f"缺失的键: {model_keys - weight_keys}")
        
        # 加载过滤后的权重
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
    # 指定包含训练模型的目录 - 确保不包含checkpoints部分
    config.work_dir = 'results/vmunet_isic18_Thursday_01_May_2025_16h_55m_01s/'
    
    # 在main函数中修改checkpoint_dir的创建方式
    main(config)

