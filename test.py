"""
Fashion-MNIST服装图像分类 - CNN模型（包含实际图片预测功能）
使用本地已下载的数据文件和PyTorch
一个.py文件 + 四个数据文件即可运行
"""

import os
import gzip
import struct
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
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
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, 10)
        
    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ==================== 3. 实际图片预测功能 ====================
def predict_custom_image(model, image_path, class_names, device):
    """
    预测自己拍摄的图片
    image_path: 自己图片的路径
    """
    print(f"\n正在处理图片: {image_path}")
    
    try:
        # 1. 加载图片
        img = Image.open(image_path)
        print(f"  原始图片大小: {img.size}, 模式: {img.mode}")
        
        # 显示原始图片
        plt.figure(figsize=(8, 8))
        plt.subplot(2, 2, 1)
        plt.imshow(img)
        plt.title('原始图片')
        plt.axis('off')
        
        # 2. 转换为灰度图
        if img.mode != 'L':
            img_gray = img.convert('L')
            print(f"  已转换为灰度图")
        else:
            img_gray = img
        
        plt.subplot(2, 2, 2)
        plt.imshow(img_gray, cmap='gray')
        plt.title('灰度图')
        plt.axis('off')
        
        # 3. 调整大小为28x28
        img_resized = img_gray.resize((28, 28))
        print(f"  调整大小到: {img_resized.size}")
        
        plt.subplot(2, 2, 3)
        plt.imshow(img_resized, cmap='gray')
        plt.title('28x28大小')
        plt.axis('off')
        
        # 4. 转换为numpy数组
        img_array = np.array(img_resized, dtype=np.float32)
        
        # 5. 反转颜色（如果背景是白色）
        # Fashion-MNIST是黑底白字，如果您的图片是白底黑字，需要反转
        if np.mean(img_array) > 127:  # 如果平均像素值大于127，说明是白底
            img_array = 255 - img_array
            print("  检测到白底图片，已进行颜色反转")
        
        # 6. 归一化到[-1, 1]（与训练数据相同）
        img_array = (img_array - 127.5) / 127.5
        
        # 显示预处理后的图片
        plt.subplot(2, 2, 4)
        plt.imshow(img_array, cmap='gray')
        plt.title('预处理后')
        plt.axis('off')
        
        plt.suptitle('图片预处理步骤', fontsize=14)
        plt.tight_layout()
        plt.show()
        
        # 7. 转换为PyTorch tensor
        img_tensor = torch.FloatTensor(img_array).unsqueeze(0).unsqueeze(0)  # (1, 1, 28, 28)
        img_tensor = img_tensor.to(device)
        
        # 8. 预测
        model.eval()
        with torch.no_grad():
            output = model(img_tensor)
            probabilities = F.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        pred_class = predicted.item()
        confidence_value = confidence.item()
        
        # 9. 显示预测结果
        print(f"\n预测结果:")
        print(f"  预测类别: {class_names[pred_class]}")
        print(f"  置信度: {confidence_value:.2%}")
        
        # 获取所有类别的置信度
        all_probs = probabilities.squeeze().cpu().numpy()
        
        # 可视化置信度分布
        plt.figure(figsize=(10, 6))
        colors = ['red' if i == pred_class else 'blue' for i in range(len(class_names))]
        bars = plt.bar(range(len(class_names)), all_probs, color=colors)
        
        plt.xticks(range(len(class_names)), class_names, rotation=45, ha='right')
        plt.ylabel('置信度')
        plt.title(f'预测结果: {class_names[pred_class]} (置信度: {confidence_value:.2%})')
        plt.ylim([0, 1])
        plt.grid(True, alpha=0.3)
        
        # 在柱状图上添加数值
        for i, (bar, prob) in enumerate(zip(bars, all_probs)):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{prob:.2%}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.show()
        
        # 10. 打印详细的置信度
        print("\n详细置信度分布:")
        for i, prob in enumerate(all_probs):
            print(f"  {class_names[i]:15s}: {prob:.2%}")
        
        return pred_class, confidence_value
        
    except Exception as e:
        print(f"处理图片时出错: {e}")
        return None, None

# ==================== 4. 训练和评估函数 ====================
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
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
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

# ==================== 5. 可视化函数 ====================
def visualize_samples(images, labels, class_names):
    """可视化数据集样本"""
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    for i in range(10):
        row, col = i // 5, i % 5
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
    
    ax1.plot(losses, 'b-', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('训练损失曲线')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(accuracies, 'r-', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('训练准确率曲线')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# ==================== 6. 主程序 ====================
def main():
    print("=" * 70)
    print("Fashion-MNIST服装图像分类 - CNN模型（含实际图片预测）")
    print("=" * 70)
    
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
    
    train_images_tensor = torch.FloatTensor(train_images).unsqueeze(1) / 255.0
    train_labels_tensor = torch.LongTensor(train_labels)
    test_images_tensor = torch.FloatTensor(test_images).unsqueeze(1) / 255.0
    test_labels_tensor = torch.LongTensor(test_labels)
    
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
    print("  Conv1: 1->32通道, 3x3卷积, padding=1")
    print("  Conv2: 32->64通道, 3x3卷积, padding=1")
    print("  Conv3: 64->128通道, 3x3卷积, padding=1")
    print("  FC1: 1152 -> 256")
    print("  FC2: 256 -> 10")
    print("  " + "-" * 40)
    
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
    
    # 7. 测试集预测可视化
    print("\n[7] 测试集预测结果...")
    model.eval()
    
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    images, labels = images.to(device), labels.to(device)
    
    with torch.no_grad():
        outputs = model(images)
        _, predictions = torch.max(outputs, 1)
    
    fig, axes = plt.subplots(4, 5, figsize=(12, 9))
    for i in range(20):
        row, col = i // 5, i % 5
        axes[row, col].imshow(images[i].cpu().squeeze(), cmap='gray')
        
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
    torch.save({
        'model_state_dict': model.state_dict(),
        'test_accuracy': test_accuracy,
    }, 'fashion_cnn_model.pth')
    print("  模型已保存为: fashion_cnn_model.pth")
    
    # 9. 实际图片预测（新增功能）
    print("\n" + "=" * 70)
    print("[9] 实际图片预测功能")
    print("=" * 70)
    
    # 检查是否有测试图片
    test_images_list = ['test_image.jpg', 'my_clothes.jpg', 'custom.jpg']
    found_images = []
    
    for img_file in test_images_list:
        if os.path.exists(img_file):
            found_images.append(img_file)
    
    if found_images:
        print("找到以下测试图片:")
        for i, img_file in enumerate(found_images, 1):
            print(f"  {i}. {img_file}")
        
        # 预测所有找到的图片
        for img_file in found_images:
            predict_custom_image(model, img_file, class_names, device)
    else:
        print("未找到测试图片，请将您的服装图片放在当前目录下并命名为:")
        print("  - test_image.jpg")
        print("  - my_clothes.jpg")
        print("  - custom.jpg")
        print("\n支持的图片格式: JPG, PNG, BMP等")
        
        # 询问用户是否要输入其他图片路径
        user_input = input("\n是否要输入其他图片路径? (y/n): ").lower()
        if user_input == 'y':
            custom_path = input("请输入图片路径: ").strip()
            if os.path.exists(custom_path):
                predict_custom_image(model, custom_path, class_names, device)
            else:
                print(f"图片不存在: {custom_path}")
    
    # 10. 总结
    print("\n" + "=" * 70)
    print("实验完成!")
    print("=" * 70)
    print(f"最终测试准确率: {test_accuracy:.2f}%")
    print("\n使用说明:")
    print("  1. 将您的服装图片重命名为 'test_image.jpg' 放在当前目录")
    print("  2. 重新运行程序，会自动进行预测")
    print("  3. 或运行以下代码单独预测图片:")
    print("     predict_custom_image(model, '您的图片.jpg', class_names, device)")
    print("=" * 70)

# ==================== 运行程序 ====================
if __name__ == '__main__':
    main()