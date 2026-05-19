# Object detection dataset(目标检测数据集)

目标检测领域没有像 MNIST 和 Fashion-MNIST 那样的小数据集。为了快速测试目标检测模型，d2l 收集并标记了一个小型香蕉检测数据集：

- 拍摄一组香蕉照片，生成 1000 张不同角度和大小的香蕉图像；
- 将香蕉图像放置在背景图片的随机位置；
- 在图片上为这些香蕉标记边界框。



## 读取数据集

数据集包含一个 CSV 文件，记录每张图像的目标类别标签和真实边界框左上角与右下角的坐标。

- `read_data_bananas(is_train)`：读取香蕉检测数据集中的图像和标签。
- `BananasDataset(is_train)`：继承 `torch.utils.data.Dataset` 的自定义数据集类，用于加载香蕉检测数据集。
- `load_data_bananas(batch_size)`：返回训练集和测试集的两个 DataLoader 实例。



## 数据格式

读取一个小批量时，可以观察到：

- 图像 batch 形状为 *(batch_size, channels, height, width)*，与图像分类任务相同。
- 标签 batch 形状为 *(batch_size, m, 5)*，其中 $m$ 是数据集中任意图像可能包含的最大边界框数量。

标签为长度为 5 的数组：

- 第一个元素：对象类别（-1 表示用于填充的非法边界框）。
- 后四个元素：边界框左上角和右下角的 *(x, y)* 坐标（值域 0~1）。

填充机制：小批量计算要求每张图像有相同数量的边界框，因此边界框少于 $m$ 的图像将被非法边界框（类别为 -1）填充至 $m$。对于香蕉数据集，$m = 1$。



## 小结

- 香蕉检测数据集可用于演示目标检测模型。
- 目标检测的数据加载与图像分类类似，但标签还需要包含真实边界框信息。






# SSD(单发多框检测)

*Single Shot MultiBox Detection*（SSD）是一种简单、快速且被广泛使用的目标检测模型。该模型由 Liu 等人于 2016 年提出。



## 模型结构

<div align="center">
<img src="https://zh-v2.d2l.ai/_images/ssd.svg" alt="SSD" align="center">
</div>

SSD 由一个 *base network*（基础网络块）和若干*多尺度特征块*串联而成：

- *base network*：从输入图像中提取特征（如截断的 VGG 或 ResNet），输出尺寸较大的特征图，用于检测较小目标。
- *multi-scale feature block*：将上一层特征图的高和宽缩小（如减半），扩大每个单元的感受野，用于检测较大目标。

通过多尺度特征块，SSD 生成不同大小的锚框并预测其类别和偏移量，从而检测大小不同的目标。



## 类别预测层

设目标类别数为 $q$，则锚框有 $q+1$ 个类别（含背景类）。设特征图高宽为 $h$、$w$，每单元生成 $a$ 个锚框，则需对 $hwa$ 个锚框分类。

SSD 借鉴 NiN 思想，使用卷积层通道输出类别预测，避免使用全连接层带来的参数过多：

- 使用 $3 \times 3$ 卷积层（padding=1，保持空间尺寸）。
- 输出通道数为 $a(q+1)$，其中索引 $i(q+1)+j$ 的通道代表第 $i$ 个锚框关于第 $j$ 类的预测。

- `cls_predictor(num_inputs, num_anchors, num_classes)`：返回类别预测层。



## 边界框预测层

设计与类别预测层类似，只是每个锚框预测 4 个偏移量而非 $q+1$ 个类别：

- `bbox_predictor(num_inputs, num_anchors)`：返回边界框预测层，输出通道数为 $a \times 4$。



## 连结多尺度的预测

不同尺度特征图形状不同，使其预测输出形状也不同。为了在维度 1 上连结：

- 将通道维移到最后，转成二维形状 *(batch_size, height × width × channels)*。
- `flatten_pred(pred)`：将预测张量按上述方式压平。
- `concat_preds(preds)`：将多个尺度的预测连结起来。



## 高宽减半块

`down_sample_blk(in_channels, out_channels)`：将输入特征图的高度和宽度减半。

- 应用了 VGG 模块设计。
- 由两个 $3 \times 3$ 卷积层（padding=1，含 BatchNorm + ReLU）和一个步幅为 2 的 $2 \times 2$ 最大池化层组成。
- 每个输出单元在输入上的感受野为 $6 \times 6$。



## 基本网络块

`base_net()`：从输入图像中抽取特征。

- 串联 3 个高宽减半块，通道数逐步翻倍（3 → 16 → 32 → 64）。
- 给定输入 $256 \times 256$，输出特征图形状为 $32 \times 32$。



## 完整的 TinySSD 模型

完整模型由 5 个块组成：

1. 基本网络块（`base_net`）；
2. 到 4. 三个高宽减半块（`down_sample_blk`）；
3. 全局最大池化块（`AdaptiveMaxPool2d((1,1))`），将高度和宽度都降到 1。

每个块的特征图都用于生成锚框，并预测类别和偏移量。

- `get_blk(i)`：返回第 $i$ 个块。
- `blk_forward(X, blk, size, ratio, cls_predictor, bbox_predictor)`：前向传播单个块，返回 *(特征图 Y, 锚框, 类别预测, 边界框预测)*。
- `TinySSD(num_classes)`：完整的 SSD 模型类。

锚框尺度配置：在区间 [0.2, 1.05] 上均匀划分得到 5 个较小值（0.2、0.37、0.54、0.71、0.88），较大值由相邻两小值的几何平均给出（如 $\sqrt{0.2 \times 0.37} = 0.272$）。

```python
sizes = [[0.2, 0.272], [0.37, 0.447], [0.54, 0.619],
         [0.71, 0.79], [0.88, 0.961]]
ratios = [[1, 2, 0.5]] * 5
num_anchors = len(sizes[0]) + len(ratios[0]) - 1  # 4
```

给定 $256 \times 256$ 输入，五个尺度的特征图分别为 $32^2, 16^2, 8^2, 4^2, 1$，每个单元 4 个锚框，共 $(32^2 + 16^2 + 8^2 + 4^2 + 1) \times 4 = 5444$ 个锚框。



## 训练模型

### 损失函数

目标检测有两类损失：

- *class loss*：使用 cross-entropy loss。
- *offset loss*（仅对正类锚框）：使用 *L1 loss*（预测值与真实值之差的绝对值），相比平方损失对异常值更稳健。掩码 `bbox_masks` 用于让负类锚框和填充锚框不参与损失计算。

```python
cls_loss = nn.CrossEntropyLoss(reduction='none')
bbox_loss = nn.L1Loss(reduction='none')
```

- `calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks)`：返回两类损失之和。


### 评价函数

- `cls_eval(cls_preds, cls_labels)`：准确率。
- `bbox_eval(bbox_preds, bbox_labels, bbox_masks)`：*mean absolute error*（MAE，平均绝对误差）。


### 训练流程

1. 前向传播生成多尺度锚框（`anchors`），并预测类别（`cls_preds`）和偏移量（`bbox_preds`）。
2. 根据标签 `Y` 为生成的锚框标注类别（`cls_labels`）和偏移量（`bbox_labels`），调用 `multibox_target`。
3. 根据预测和标注计算损失并反向传播更新参数。



## 预测目标

- 对模型输出做 softmax 得到类别概率 `cls_probs`。
- 调用 `multibox_detection` 应用 *NMS* 得到最终预测边界框。
- 筛选置信度高于阈值（如 0.9）的预测框作为最终输出。



## 改进方法（练习）

- 损失函数改进：
  - *smooth L1 loss*：在零点附近用平方函数从而更加平滑，通过 $\sigma$ 控制平滑区域。
  - *focal loss*：$-\alpha(1 - p_j)^\gamma \log p_j$，增大 $\gamma$ 可有效减少易分样本的相对损失，让训练更集中在难分样本上。
- 输入图像放大、负锚框减半、类别和偏移损失加权等。



## 小结

- SSD 是一种多尺度目标检测模型，基于基础网络块和多尺度特征块生成不同数量和大小的锚框，并预测类别和偏移量来检测不同大小的目标。
- 训练 SSD 模型时，损失函数由类别 cross-entropy loss 和偏移量 L1 loss 加权而成。






# R-CNN系列

*Region-based CNN*（R-CNN）是将深度模型应用于目标检测的开创性工作之一。本节介绍 R-CNN 及其改进：*Fast R-CNN*、*Faster R-CNN* 和 *Mask R-CNN*。



## R-CNN

<div align="center">
<img src="https://zh-v2.d2l.ai/_images/r-cnn.svg" alt="R-CNN" align="center">
</div>

R-CNN 的四个步骤：

1. 对输入图像使用 *selective search*（选择性搜索）选取若干（如 2000 个）*region proposal*（提议区域），并标注类别和真实边界框；
2. 选择一个预训练的 CNN，截断在输出层之前。将每个提议区域变形为网络需要的输入尺寸，前向传播得到提议区域的特征；
3. 训练多个 *SVM*（support vector machine）对每个提议区域的特征做类别预测；
4. 训练线性回归模型，根据提议区域特征预测真实边界框。

**缺点**：每张图可能产生上千个提议区域，每个区域都要独立做一次 CNN 前向传播，计算量巨大，速度很慢。



## Fast R-CNN

<div align="center">
<img src="https://zh-v2.d2l.ai/_images/fast-rcnn.svg" alt="Fast R-CNN" align="center">
</div>

R-CNN 的主要性能瓶颈：对每个提议区域独立做卷积神经网络的前向传播，没有共享计算。Fast R-CNN 的主要改进：**仅在整张图像上执行一次 CNN 前向传播**。

计算流程：

1. 整张图像输入 CNN，输出特征图，形状为 $1 \times c \times h_1 \times w_1$；
2. 假设选择性搜索生成 $n$ 个提议区域。这些区域在 CNN 特征图上对应不同形状的*兴趣区域*（region of interest, RoI）。Fast R-CNN 引入 *RoI pooling*（兴趣区域汇聚层），将不同形状的 RoI 转换为相同形状的特征，输出形状为 $n \times c \times h_2 \times w_2$；
3. 全连接层将形状变换为 $n \times d$；
4. 分别预测每个 RoI 的类别（输出形状 $n \times q$）和边界框（输出形状 $n \times 4$）。

*RoI pooling*：

- 与普通 pooling 不同，RoI pooling 可以直接指定输出形状。
- 对任意形状 $h \times w$ 的 RoI，划分为 $h_2 \times w_2$ 个子窗口，每个子窗口大小约为 $(h/h_2) \times (w/w_2)$（向上取整），取最大值作为输出。
- 因此可以从形状各异的 RoI 中提取出形状相同的特征。
- PyTorch 实现：`torchvision.ops.roi_pool(X, rois, output_size, spatial_scale)`。



## Faster R-CNN

<div align="center">
<img src="https://zh-v2.d2l.ai/_images/faster-rcnn.svg" alt="Faster R-CNN" align="center">
</div>

Fast R-CNN 仍需要选择性搜索生成大量提议区域，速度较慢。Faster R-CNN 的关键改进：将选择性搜索替换为 *region proposal network*（RPN，区域提议网络）。

*RPN* 的计算步骤：

1. 使用 padding=1 的 $3 \times 3$ 卷积层变换 CNN 输出，输出通道数为 $c$。
2. 以特征图每个像素为中心，生成多个不同大小和宽高比的锚框并标注。
3. 使用锚框中心单元长度为 $c$ 的特征，预测该锚框的二元类别（含目标 / 背景）和边界框偏移量。
4. 使用 *NMS*，从预测为目标的边界框中移除相似结果。剩余的预测边界框即作为 RoI pooling 所需的提议区域。

RPN 与整个网络一起端到端训练，因此能学到如何生成高质量的提议区域，从而在减少提议区域数量的情况下保证检测精度。



## Mask R-CNN

<div align="center">
<img src="https://zh-v2.d2l.ai/_images/mask-rcnn.svg" alt="Mask R-CNN" align="center">
</div>

如果训练集还标注了每个目标在图像上的*像素级位置*，Mask R-CNN 能利用这些详尽的标注信息进一步提升精度。基于 Faster R-CNN 修改：

- 将 *RoI pooling* 替换为 *RoI Align*：使用*双线性插值*（bilinear interpolation）来保留特征图上的空间信息，更适于像素级预测。
- RoI Align 输出不仅用于预测类别和边界框，还通过额外的 *FCN*（全卷积网络）预测目标的像素级位置(mask)。



## 小结

- R-CNN：对图像选取若干提议区域，分别用 CNN 提取特征，再预测类别和边界框。慢。
- Fast R-CNN：只对整张图像做一次 CNN 前向传播，引入 RoI pooling 提取定长特征。
- Faster R-CNN：将选择性搜索替换为 RPN，端到端训练。
- Mask R-CNN：在 Faster R-CNN 基础上引入 FCN，借助目标像素级位置进一步提升精度。






# Semantic segmentation(语义分割)

之前讨论的目标检测一直使用方形边界框来标注和预测目标。*Semantic segmentation*（语义分割）则关注如何将图像分割成属于不同语义类别的区域，其标注和预测都是*像素级*的。



## 图像分割相关概念区分

- *Image segmentation*（图像分割）：将图像划分为若干组成区域，通常利用像素之间的相关性，**无须像素标签**。预测时无法保证分割出的区域具有期望的语义。
- *Semantic segmentation*（语义分割）：识别并理解图像中每个像素的语义类别。
- *Instance segmentation*（实例分割）：又称 *simultaneous detection and segmentation*。不仅区分语义，还区分不同的目标实例（如同一图像中的两条狗）。




## Pascal VOC2012 数据集

最重要的语义分割数据集之一，包含 21 个类别（含 background）。

数据集组件位于 `../data/VOCdevkit/VOC2012`：

- `ImageSets/Segmentation`：训练和测试样本的文本文件。
- `JPEGImages`：输入图像。
- `SegmentationClass`：标签图像，尺寸与输入图像相同。**标签中颜色相同的像素属于同一个语义类别**。

- `read_voc_images(voc_dir, is_train)`：将所有 VOC 图像和标签读入内存。
- `voc_colormap2label()`：构建从 RGB 颜色值到 VOC 类别索引的映射。
- `voc_label_indices(colormap, colormap2label)`：将 VOC 标签中的 RGB 值映射到类别索引。

预定义常量：

```python
VOC_COLORMAP = [[0, 0, 0], [128, 0, 0], ...]
VOC_CLASSES = ['background', 'aeroplane', 'bicycle', ...]
```



## 预处理数据

与图像分类不同，语义分割的输入图像与标签在像素上一一对应，**因此使用随机裁剪而非缩放**，避免重新映射带来的精度损失。

- `voc_rand_crop(feature, label, height, width)`：对特征图和标签图做相同区域的随机裁剪。
- `VOCSegDataset(is_train, crop_size, voc_dir)`：自定义语义分割数据集类，提供：
  - `normalize_image`：对 RGB 三通道做标准化。
  - `filter`：移除尺寸小于裁剪尺寸的图像。
  - `__getitem__`：返回 *(归一化后的特征, 类别索引标签)*。
- `load_data_voc(batch_size, crop_size)`：下载并读取 Pascal VOC2012 数据集，返回训练集和测试集迭代器。

读取一个小批量可以发现：与图像分类或目标检测不同，**这里的标签是一个三维数组**（batch_size × height × width）。



## 小结

- *Semantic segmentation* 通过将图像划分为属于不同语义类别的区域，识别并理解图像中像素级别的内容。
- 重要数据集：Pascal VOC2012。
- 语义分割的输入图像和标签在像素上一一对应，因此使用随机裁剪而非缩放。






# FCN(全卷积网络)

*Fully Convolutional Network*（FCN，全卷积网络）采用卷积神经网络实现了从图像像素到像素类别的变换，是语义分割的经典模型。

与图像分类或目标检测中的 CNN 不同，FCN 将中间层特征图的高宽**变换回输入图像的尺寸**（通过 *transposed convolution* 实现）。因此输出的类别预测与输入图像在像素级别上一一对应。



## 构造模型

<div align="center">
<img src="https://zh-v2.d2l.ai/_images/fcn.svg" alt="FCN" align="center">
</div>

最基本的 FCN 设计：

1. 使用 CNN 抽取图像特征（如截断的 ResNet-18，去掉最后的全局平均池化层和全连接层）；
2. 通过 $1 \times 1$ 卷积层将通道数变换为类别个数；
3. 通过 *transposed convolution* 层将特征图的高和宽变换回输入图像尺寸。

```python
pretrained_net = torchvision.models.resnet18(pretrained=True)
net = nn.Sequential(*list(pretrained_net.children())[:-2])
net.add_module('final_conv', nn.Conv2d(512, num_classes, kernel_size=1))
net.add_module('transpose_conv',
    nn.ConvTranspose2d(num_classes, num_classes,
                       kernel_size=64, padding=16, stride=32))
```

规律：若步幅为 $s$，填充为 $s/2$（假设 $s/2$ 是整数），卷积核高宽为 $2s$，则转置卷积层将输入高宽分别放大 $s$ 倍。



## 初始化转置卷积层

*Bilinear interpolation*（双线性插值）是上采样的常用方法，也常用于初始化转置卷积层。其计算过程：

1. 将输出图像坐标 $(x, y)$ 映射到输入图像坐标 $(x', y')$（实数）；
2. 在输入图像上找到离 $(x', y')$ 最近的 4 个像素；
3. 根据这 4 个像素及其与 $(x', y')$ 的相对距离计算输出像素值。

- `bilinear_kernel(in_channels, out_channels, kernel_size)`：构造双线性插值核张量。

FCN 中转置卷积层用 `bilinear_kernel` 初始化；$1 \times 1$ 卷积层用 Xavier 初始化。



## 训练和预测

- 损失函数：每像素的 cross-entropy loss，需要在 channel 维上计算：

```python
def loss(inputs, targets):
    return F.cross_entropy(inputs, targets, reduction='none').mean(1).mean(1)
```

- `predict(img)`：对输入图像做标准化和形状变换，前向传播后取 argmax 得到类别预测。
- `label2image(pred)`：将类别索引映射回 RGB 颜色用于可视化。

测试时如图像高宽不是 32 的整数倍，可在图像中截取多个高宽为 32 的整数倍的矩形区域分别预测，再取重叠区域的平均值。



## 小结

- FCN 先用 CNN 抽取特征，再用 $1 \times 1$ 卷积转换通道数为类别个数，最后通过转置卷积变换回输入尺寸。
- FCN 中转置卷积层可以初始化为双线性插值的上采样。






# Neural style transfer(风格迁移)

*Style transfer*（风格迁移）使用卷积神经网络自动将一张图像的风格应用到另一张图像之上。需要两张输入图像：*content image*（内容图像）和 *style image*（风格图像），输出 *composite image*（合成图像）保留内容图像的物体形状，同时具有风格图像的色彩和笔触。



## 方法

<div align="center">
<img src="https://zh-v2.d2l.ai/_images/neural-style.svg" alt="风格迁移" align="center">
</div>

1. 初始化*合成图像*，例如初始化为内容图像。合成图像是风格迁移过程中**唯一需要更新的变量**（即模型参数）。
2. 选择一个预训练的 CNN 抽取图像特征，**模型参数在训练中无须更新**。
3. 选择某些层的输出作为内容特征或风格特征：
   - *content layer*：靠近输出层，输出图像的内容特征。
   - *style layer*：每个卷积块的第一个卷积层，输出图像的风格特征。
4. 通过前向传播计算损失，通过反向传播迭代更新合成图像。

损失函数由 3 部分组成：

- *content loss*：合成图像与内容图像在内容特征上接近；
- *style loss*：合成图像与风格图像在风格特征上接近；
- *total variation loss*：减少合成图像的噪点。



## 预处理与后处理

- `preprocess(img, image_shape)`：对输入图像在 RGB 三个通道分别做标准化，并转换为 CNN 输入格式。
- `postprocess(img)`：还原标准化前的像素值，并裁剪到 [0, 1] 范围内用于显示。



## 抽取图像特征

使用基于 ImageNet 预训练的 *VGG-19* 模型：

```python
style_layers, content_layers = [0, 5, 10, 19, 28], [25]
net = nn.Sequential(*[pretrained_net.features[i]
                      for i in range(max(content_layers + style_layers) + 1)])
```

- `extract_features(X, content_layers, style_layers)`：逐层前向传播，保留内容层和风格层的输出。
- `get_contents(image_shape, device)`：对内容图像抽取内容特征。
- `get_styles(image_shape, device)`：对风格图像抽取风格特征。

由于训练时无须改变 VGG 参数，**内容和风格特征可在训练开始前预先提取**，合成图像的特征才需在训练中通过 `extract_features` 动态抽取。



## 损失函数

### Content loss

通过平方误差函数衡量合成图像与内容图像在内容特征上的差异：

- `content_loss(Y_hat, Y)`：内容损失，注意从计算图中分离 `Y`（detach），因为它是一个规定的值而非变量。


### Style loss

使用 *Gram matrix*（格拉姆矩阵）表达风格层输出的风格。

将风格层输出（通道数 $c$、高宽 $h$、$w$）变换为矩阵 $\mathbf{X} \in \mathbb{R}^{c \times hw}$，则 Gram 矩阵 $\mathbf{X}\mathbf{X}^\top \in \mathbb{R}^{c \times c}$ 中第 $(i, j)$ 个元素是通道 $i$ 与通道 $j$ 上风格特征的内积，表达了通道之间风格特征的相关性。

为避免数值过大，将 Gram 矩阵除以矩阵元素个数 $chw$。

- `gram(X)`：计算归一化的 Gram 矩阵。
- `style_loss(Y_hat, gram_Y)`：通过平方误差衡量合成图像与风格图像 Gram 矩阵的差异。


### Total variation loss

合成图像中可能出现高频噪点（即特别亮或暗的颗粒像素）。*Total variation denoising*（全变分去噪）通过最小化

$$\sum_{i, j} \left|x_{i, j} - x_{i+1, j}\right| + \left|x_{i, j} - x_{i, j+1}\right|$$

使邻近像素值尽可能相似。

- `tv_loss(Y_hat)`：计算全变分损失。


### 总损失

```python
content_weight, style_weight, tv_weight = 1, 1e3, 10
```

- `compute_loss(X, contents_Y_hat, styles_Y_hat, contents_Y, styles_Y_gram)`：返回三部分损失及其加权和。
- 通过调整权重超参数权衡保留内容、迁移风格和去噪的相对重要性。



## 初始化合成图像

合成图像是训练期间唯一需要更新的变量，将其包装为 `nn.Module`：

- `SynthesizedImage(img_shape)`：合成图像模型，其参数即为合成图像本身。
- `get_inits(X, device, lr, styles_Y)`：创建合成图像模型实例并初始化为 `X`，预计算风格图像的 Gram 矩阵。



## 训练模型

- `train(X, contents_Y, styles_Y, device, lr, num_epochs, lr_decay_epoch)`：风格迁移训练循环。每一步抽取合成图像的内容特征和风格特征，计算损失并反向传播更新合成图像。



## 小结

- 风格迁移的损失函数由 *content loss*、*style loss* 和 *total variation loss* 三部分组成。
- 通过预训练 CNN 抽取图像特征，最小化损失函数来更新合成图像作为模型参数。
- 使用 *Gram matrix* 表达风格层输出的风格。






# 实战 Kaggle 比赛：CIFAR-10 图像分类

之前章节中我们一直使用深度学习框架的高级 API 直接获取张量格式的图像数据集。实践中图像数据集通常以图像文件形式出现，本节展示从原始图像文件开始，组织、读取并转换为张量格式的完整流程。



## 数据集组织

CIFAR-10 包含 50000 张训练图像和 300000 张测试图像，均为 $32 \times 32$ 的 RGB png 图像，共 10 个类别。

原始数据集结构：

- `../data/cifar-10/train/[1-50000].png`
- `../data/cifar-10/test/[1-300000].png`
- `../data/cifar-10/trainLabels.csv`
- `../data/cifar-10/sampleSubmission.csv`

将原始数据集整理为子文件夹形式，方便使用 `ImageFolder` 加载：

- `read_csv_labels(fname)`：读取 CSV 文件中的标签，返回 *{文件名: 标签}* 字典。
- `copyfile(filename, target_dir)`：将文件复制到目标目录。
- `reorg_train_valid(data_dir, labels, valid_ratio)`：将验证集从原始训练集中按比例拆分出来。每个类别拆分出 $\max(\lfloor nr \rfloor, 1)$ 张图像作为验证集（$n$ 为样本最少类别的图像数）。
- `reorg_test(data_dir)`：整理测试集到 `unknown` 子目录下，方便 `ImageFolder` 读取。
- `reorg_cifar10_data(data_dir, valid_ratio)`：组合上述函数完成整体整理。



## 图像增广

训练时：

```python
transform_train = torchvision.transforms.Compose([
    torchvision.transforms.Resize(40),
    torchvision.transforms.RandomResizedCrop(32, scale=(0.64, 1.0),
                                                 ratio=(1.0, 1.0)),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465],
                                     [0.2023, 0.1994, 0.2010])])
```

测试时只做标准化（避免随机性影响评估结果）。



## 读取数据集

使用 `torchvision.datasets.ImageFolder` 分别加载 `train`、`train_valid`、`valid`、`test` 四个目录，再用 `DataLoader` 创建迭代器。



## 模型与训练

- `get_net()`：使用 d2l 实现的 ResNet-18 作为模型。
- 损失函数：`nn.CrossEntropyLoss(reduction="none")`。
- `train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period, lr_decay)`：训练函数，使用 SGD + Momentum 优化器，每 `lr_period` 个 epoch 学习率乘以 `lr_decay`。



## 提交结果

获得满意的模型后，使用所有标记数据（包括验证集，即 `train_valid_iter`）重新训练模型，对测试集分类并生成 `submission.csv` 提交。



## 小结

- 将原始图像文件组织为所需目录结构后，可以使用 `ImageFolder` 读取。
- 图像分类竞赛中可以综合使用 CNN 和图像增广。






# 实战 Kaggle 比赛：狗品种识别（ImageNet Dogs）

在 Kaggle 上识别 120 种不同品种的狗。该数据集是 ImageNet 的子集，图像比 CIFAR-10 中的图像更大，且尺寸不一。



## 数据集组织

原始数据集结构：

- `../data/dog-breed-identification/labels.csv`
- `../data/dog-breed-identification/sample_submission.csv`
- `../data/dog-breed-identification/train`
- `../data/dog-breed-identification/test`

- `reorg_dog_data(data_dir, valid_ratio)`：与 CIFAR-10 类似，调用 `read_csv_labels`、`reorg_train_valid`、`reorg_test` 完成数据集整理。



## 图像增广

由于图像较大，训练时图像增广更丰富：

```python
transform_train = torchvision.transforms.Compose([
    torchvision.transforms.RandomResizedCrop(224, scale=(0.08, 1.0),
                                             ratio=(3.0/4.0, 4.0/3.0)),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ColorJitter(brightness=0.4,
                                       contrast=0.4,
                                       saturation=0.4),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])])
```

测试时使用 `Resize(256) + CenterCrop(224)` 的确定性预处理。



## 微调预训练模型

由于该数据集是 ImageNet 的子集，可以使用在完整 ImageNet 数据集上预训练的模型作为特征提取器，仅训练自定义的小型输出网络：

```python
def get_net(devices):
    finetune_net = nn.Sequential()
    finetune_net.features = torchvision.models.resnet34(pretrained=True)
    finetune_net.output_new = nn.Sequential(
        nn.Linear(1000, 256), nn.ReLU(), nn.Linear(256, 120))
    finetune_net = finetune_net.to(devices[0])
    for param in finetune_net.features.parameters():
        param.requires_grad = False
    return finetune_net
```

关键点：

- 选择 ResNet-34 作为特征提取器，**冻结其参数**（`requires_grad = False`）。
- 在原模型 1000 维输出之上添加新的输出网络：`Linear(1000, 256) → ReLU → Linear(256, 120)`。
- 仅训练自定义输出网络，节省梯度下降的时间和内存空间。
- 标准化使用 ImageNet 三通道均值和标准差，符合预训练模型的标准化操作。



## 训练与提交

- `evaluate_loss(data_iter, net, devices)`：评估损失。
- `train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period, lr_decay)`：训练函数，**只优化 `requires_grad=True` 的参数**。
- 提交：与 CIFAR-10 不同，狗品种识别需要对每张图像输出 120 个类别的概率分布（softmax 后），写入 `submission.csv`。



## 小结

- ImageNet 数据集中的图像比 CIFAR-10 图像尺寸大，图像增广操作也需要相应调整。
- 对于 ImageNet 的子集分类任务，可以利用完整 ImageNet 数据集上的预训练模型提取特征，仅训练小型自定义输出网络，减少计算和内存开销。
