import torch

config = {
    # 网络训练部分
    'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    'batch_size': 1,
    'epochs': 21,
    'save_epoch': 1,
    'learning_rate': 1e-3,
    'min_learning_rate': 1e-5,

    'train_data': 'train/train_data.txt',
    'test_data': 'train/pathPointList.txt',

    'show_output_path': 'out',
    'show_test_path': 'test_result',
    'ckpt': 'ckpt/ckpt_for_test.pth',
    'save_file': 'ckpt/ckpt',
    'save_min_loss_file': 'ckpt/min_loss_ckpt.pth'

}