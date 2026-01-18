"""
Fashion-MNIST服装图像分类 - CNN模型
使用本地已下载的数据文件和PyTorch
一个.py文件 + 四个数据文件即可运行
"""

import os
import gzip
import struct
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 检查并导入PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    print("正在安装PyTorch...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "torch", "torchvision", "-i", "https://pypi.tuna.tsinghua.edu.cn/simple"])
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 1. 数据加载 ====================
def load_fashion_mnist():
    """从本地文件加载Fashion-MNIST数据集"""
    # 检查文件是否存在
    required_files = [
        'train-images-idx3-ubyte.gz',
        'train-labels-idx1-ubyte.gz', 
        't10k-images-idx3-ubyte.gz',
        't10k-labels-idx1-ubyte.gz'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"错误: 找不到以下文件:")
        for file in missing_files:
            print(f"  - {file}")
        print("\n请将四个数据文件放在当前目录下:")
        print("  1. train-images-idx3-ubyte.gz")
        print("  2. train-labels-idx1-ubyte.gz")
        print("  3. t10k-images-idx3-ubyte.gz")
        print("  4. t10k-labels-idx1-ubyte.gz")
        return None
    
    print("正在从本地文件加载数据集...")
    
    # 加载训练数据
    with gzip.open('train-images-idx3-ubyte.gz', 'rb') as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        train_images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows, cols)
    
    with gzip.open('train-labels-idx1-ubyte.gz', 'rb') as f:
        magic, num = struct.unpack(">II", f.read(8))
        train_labels = np.frombuffer(f.read(), dtype=np.uint8)
    
    # 加载测试数据
    with gzip.open('t10k-images-idx3-ubyte.gz', 'rb') as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        test_images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows, cols)
    
    with gzip.open('t10k-labels-idx1-ubyte.gz', 'rb') as f:
        magic, num = struct.unpack(">II", f.read(8))
        test_labels = np.frombuffer(f.read(), dtype=np.uint8)
    
    print(f"加载完成:")
    print(f"  训练集: {len(train_images)} 张图片")
    print(f"  测试集: {len(test_images)} 张图片")
    
    return (train_images, train_labels), (test_images, test_labels)

# ==================== 2. CNN模型定义 ====================
class FashionCNN(nn.Module):
    """包含3个卷积层的CNN模型"""
    def __init__(self):
        super(FashionCNN, self).__init__()
        
        # 第一个卷积层: 输入1通道 -> 输出32通道
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        
        # 第二个卷积层: 32通道 -> 64通道
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        # 第三个卷积层: 64通道 -> 128通道
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # 池化层
        self.pool = nn.MaxPool2d(2, 2)
        
        # 全连接层
        # 经过3次池化后: 28x28 -> 14x14 -> 7x7 -> 3x3 (向下取整)
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, 10)
        
    def forward(self, x):
        # 添加通道维度 (batch_size, 1, 28, 28)
        if x.dim() == 3:
            x = x.unsqueeze(1)
        
        # 卷积块1: 28x28 -> 14x14
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        
        # 卷积块2: 14x14 -> 7x7
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        
        # 卷积块3: 7x7 -> 3x3
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        
        # 展平
        x = x.view(x.size(0), -1)
        
        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x

# ==================== 3. 训练和评估函数 ====================
def train_model(model, train_loader, device, epochs=10):
    """训练模型"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    losses = []
    accuracies = []
    
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 统计
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        losses.append(epoch_loss)
        accuracies.append(epoch_acc)
        
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')
    
    return losses, accuracies

def evaluate_model(model, test_loader, device):
    """评估模型"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy

# ==================== 4. 可视化函数 ====================
def visualize_samples(images, labels, class_names):
    """可视化数据集样本"""
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    for i in range(10):
        row, col = i // 5, i % 5
        # 找到每个类别的第一个样本
        idx = np.where(labels == i)[0][0]
        axes[row, col].imshow(images[idx], cmap='gray')
        axes[row, col].set_title(f'{class_names[i]}')
        axes[row, col].axis('off')
    plt.suptitle('Fashion-MNIST数据集样本', fontsize=14)
    plt.tight_layout()
    plt.show()

def plot_training_curves(losses, accuracies):
    """绘制训练曲线"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # 损失曲线
    ax1.plot(losses, 'b-', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('训练损失曲线')
    ax1.grid(True, alpha=0.3)
    
    # 准确率曲线
    ax2.plot(accuracies, 'r-', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('训练准确率曲线')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# ==================== 5. 主程序 ====================
def main():
    print("=" * 60)
    print("Fashion-MNIST服装图像分类 - CNN模型")
    print("=" * 60)
    
    # 类别名称
    class_names = ['T恤/上衣', '裤子', '套头衫', '连衣裙', '外套',
                   '凉鞋', '衬衫', '运动鞋', '包包', '踝靴']
    
    # 1. 加载数据
    print("\n[1] 加载数据集...")
    data = load_fashion_mnist()
    if data is None:
        return
    
    (train_images, train_labels), (test_images, test_labels) = data
    
    # 2. 可视化数据
    print("\n[2] 可视化数据集样本...")
    visualize_samples(train_images, train_labels, class_names)
    
    # 3. 准备数据
    print("\n[3] 准备训练数据...")
    
    # 转换为PyTorch张量并归一化
    train_images_tensor = torch.FloatTensor(train_images).unsqueeze(1) / 255.0
    train_labels_tensor = torch.LongTensor(train_labels)
    test_images_tensor = torch.FloatTensor(test_images).unsqueeze(1) / 255.0
    test_labels_tensor = torch.LongTensor(test_labels)
    
    # 创建数据集和数据加载器
    train_dataset = TensorDataset(train_images_tensor, train_labels_tensor)
    test_dataset = TensorDataset(test_images_tensor, test_labels_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    print(f"  训练批次: {len(train_loader)}")
    print(f"  测试批次: {len(test_loader)}")
    
    # 4. 创建和训练模型
    print("\n[4] 创建CNN模型...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  使用设备: {device}")
    
    model = FashionCNN().to(device)
    
    # 打印模型结构
    print("\n  模型结构:")
    print("  " + "-" * 40)
    print("  输入: 1x28x28 灰度图像")
    print("  Conv1: 1->32通道, 3x3卷积, padding=1")
    print("  MaxPool: 2x2, stride=2")
    print("  Conv2: 32->64通道, 3x3卷积, padding=1")
    print("  MaxPool: 2x2, stride=2")
    print("  Conv3: 64->128通道, 3x3卷积, padding=1")
    print("  MaxPool: 2x2, stride=2")
    print("  FC1: 1152 -> 256 (128*3*3=1152)")
    print("  FC2: 256 -> 10")
    print("  " + "-" * 40)
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  总参数量: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    
    # 5. 训练模型
    print("\n[5] 开始训练模型...")
    print("  " + "-" * 40)
    
    losses, accuracies = train_model(model, train_loader, device, epochs=10)
    
    # 绘制训练曲线
    plot_training_curves(losses, accuracies)
    
    # 6. 评估模型
    print("\n[6] 评估模型...")
    train_accuracy = evaluate_model(model, train_loader, device)
    test_accuracy = evaluate_model(model, test_loader, device)
    
    print(f"  训练集准确率: {train_accuracy:.2f}%")
    print(f"  测试集准确率: {test_accuracy:.2f}%")
    
    # 7. 可视化预测结果
    print("\n[7] 可视化预测结果...")
    model.eval()
    
    # 获取一批测试数据
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    images, labels = images.to(device), labels.to(device)
    
    # 进行预测
    with torch.no_grad():
        outputs = model(images)
        _, predictions = torch.max(outputs, 1)
    
    # 显示预测结果
    fig, axes = plt.subplots(4, 5, figsize=(12, 9))
    for i in range(20):
        row, col = i // 5, i % 5
        axes[row, col].imshow(images[i].cpu().squeeze(), cmap='gray')
        
        # 判断预测是否正确
        if predictions[i] == labels[i]:
            color = 'green'
            status = "✓"
        else:
            color = 'red'
            status = "✗"
        
        axes[row, col].set_title(f'{class_names[predictions[i]]} {status}', 
                               color=color, fontsize=9)
        axes[row, col].axis('off')
    
    plt.suptitle('测试集预测结果（绿色=正确，红色=错误）', fontsize=14)
    plt.tight_layout()
    plt.show()
    
    # 8. 保存模型
    print("\n[8] 保存模型...")
    torch.save(model.state_dict(), 'fashion_cnn_model.pth')
    print("  模型已保存为: fashion_cnn_model.pth")
    
    # 9. 总结
    print("\n" + "=" * 60)
    print("实验完成!")
    print("=" * 60)
    print(f"最终测试准确率: {test_accuracy:.2f}%")
    print("\n文件清单:")
    print("  1. 数据文件 (4个):")
    print("     - train-images-idx3-ubyte.gz")
    print("     - train-labels-idx1-ubyte.gz")
    print("     - t10k-images-idx3-ubyte.gz")
    print("     - t10k-labels-idx1-ubyte.gz")
    print("  2. 代码文件: 当前.py文件")
    print("  3. 模型文件: fashion_cnn_model.pth")
    print("=" * 60)

# ==================== 运行程序 ====================
if __name__ == '__main__':
    main()