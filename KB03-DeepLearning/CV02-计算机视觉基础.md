# Image augmentation(图像增广)

大型数据集是成功应用深度神经网络的先决条件。 图像增广在对训练图像进行一系列的随机变化之后，生成相似但不同的训练样本，从而扩大了训练集的规模。 此外，应用图像增广的原因是，随机改变训练样本可以减少模型对某些属性的依赖，从而提高模型的泛化能力。 例如，我们可以以不同的方式裁剪图像，使感兴趣的对象出现在不同的位置，减少模型对于对象出现位置的依赖。 我们还可以调整亮度、颜色等因素来降低模型对颜色的敏感度。 可以说，图像增广技术对于AlexNet的成功是必不可少的。


## 常用的图像增广方法

### 翻转和裁剪

- [**左右翻转图像**]：通常不会改变对象的类别。这是最早且最广泛使用的图像增广方法之一。 可以使用`transforms`模块来创建`RandomFlipLeftRight`实例，这样就各有50%的几率使图像向左或向右翻转。
- [**上下翻转图像**]：不如左右图像翻转那样常用。但是，至少对于这个示例图像，上下翻转不会妨碍识别。接下来，我们创建一个`RandomFlipTopBottom`实例，使图像各有50%的几率向上或向下翻转。
- [**随机裁剪**]：将一个面积为原始面积10%到100%的区域，该区域的宽高比从0.5～2之间随机取值。 然后，区域的宽度和高度都被缩放到200像素。




### 改变颜色

另一种增广方法是改变颜色。 我们可以改变图像颜色的四个方面：亮度、对比度、饱和度和色调。

- [**随机更改图像的亮度**]，随机值为原始图像的50%（$1-0.5$）到150%（$1+0.5$）之间。
- [**随机更改图像的色调**]。
- [**同时更改图像的亮度（`brightness`）、对比度（`contrast`）、饱和度（`saturation`）和色调（`hue`）**]




### 结合多种图像增广方法

在实践中，我们将结合多种图像增广方法。比如，我们可以通过使用一个`Compose`实例来综合上面定义的不同的图像增广方法，并将它们应用到每个图像。例如：

```python
augs = torchvision.transforms.Compose([
    torchvision.transforms.RandomHorizontalFlip(), color_aug, shape_aug])
```




## 小结

- 图像增广基于现有的训练数据生成随机图像，来提高模型的泛化能力。
- 为了在预测过程中得到确切的结果，我们通常对训练样本只进行图像增广，而在预测过程中不使用带随机操作的图像增广。
- 深度学习框架提供了许多不同的图像增广方法，这些方法可以被同时应用。






# Fine-tuning(微调)

## 引子

> 一些常见的数据集：
> - **Fashion-MNIST训练数据集**: 有6万张图像，Fashion-MNIST是一个小型服装数据集，它包括10个类别，每个类别有6000张图像。
> - **ImageNet数据集**: 有超过1000万的图像和1000类的物体。


**一个需求示例**：假如我们想识别图片中不同类型的椅子，然后向用户推荐购买链接。


**解决方案一**：首先识别100把普通椅子，为每把椅子拍摄1000张不同角度的图像，然后在收集的图像数据集上训练一个分类模型。
- 尽管这个椅子数据集可能大于Fashion-MNIST数据集，但实例数量仍然不到ImageNet中的十分之一。
- 适合ImageNet的复杂模型可能会在这个椅子数据集上过拟合。
- 此外，由于训练样本数量有限，训练模型的准确性可能无法满足实际要求。
- 一个显而易见的解决方案是收集更多的数据，但是收集和标记数据可能需要大量的时间和金钱。例如，为了收集ImageNet数据集，研究人员花费了数百万美元的研究资金。尽管目前的数据收集成本已大幅降低，但这一成本仍不能忽视。


**解决方案二**：应用[**迁移学习**]（transfer learning）将从[**源数据集**]学到的知识迁移到[**目标数据集**]。
- 例如，尽管ImageNet数据集中的大多数图像与椅子无关，但在此数据集上训练的模型可能会提取更通用的图像特征，这有助于识别边缘、纹理、形状和对象组合。
- 这些类似的特征也可能有效地识别椅子。





## Fine-tuning的步骤

迁移学习中的常见技巧: **微调（fine-tuning）**。如 图13.2.1: 所示，微调包括以下四个步骤。

<div align="center">
<img src="https://zh-v2.d2l.ai/_images/finetune.svg" alt="微调" align="center">
</div>

1. 在源数据集（例如ImageNet数据集）上预训练神经网络模型，即*源模型*。
2. 创建一个新的神经网络模型，即*目标模型*。这将复制源模型上的所有模型设计及其参数（输出层除外）。我们假定这些模型参数包含从源数据集中学到的知识，这些知识也将适用于目标数据集。我们还假设源模型的输出层与源数据集的标签密切相关；因此不在目标模型中使用该层。
3. 向目标模型添加输出层，其输出数是目标数据集中的类别数。然后随机初始化该层的模型参数。
4. 在目标数据集（如椅子数据集）上训练目标模型。输出层将从头开始进行训练，而所有其他层的参数将根据源模型的参数进行微调。


当目标数据集比源数据集小得多时，微调有助于提高模型的泛化能力。






## 实践案例：热狗识别

我们将在一个小型数据集上微调ResNet模型。该模型已在ImageNet数据集上进行了预训练。 这个小型数据集包含数千张包含热狗和不包含热狗的图像，我们将使用微调模型来识别图像中是否包含热狗。




### 获取数据集

我们使用的[**热狗数据集来源于网络**]。 该数据集包含1400张热狗的“正类”图像，以及包含尽可能多的其他食物的“负类”图像。 含着两个类别的1000张图片用于训练，其余的则用于测试。


解压下载的数据集，我们获得了两个文件夹`hotdog/train`和`hotdog/test`。 这两个文件夹都有`hotdog`（有热狗）和`not-hotdog`（无热狗）两个子文件夹， 子文件夹内都包含相应类的图像。

```python
# download data
d2l.DATA_HUB['hotdog'] = (d2l.DATA_URL + 'hotdog.zip',
                         'fba480ffa8aa7e0febbb511d181409f899b9baa5')
data_dir = d2l.download_extract('hotdog')

# get train and test images
train_imgs = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'train'))
test_imgs = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'test'))

# show images
hotdogs = [train_imgs[i][0] for i in range(8)]
not_hotdogs = [train_imgs[-i - 1][0] for i in range(8)]
d2l.show_images(hotdogs + not_hotdogs, 2, 8, scale=1.4)
```



在训练期间，我们首先从图像中裁切随机大小和随机长宽比的区域，然后将该区域缩放为$224 \times 224$输入图像。 在测试过程中，我们将图像的高度和宽度都缩放到256像素，然后裁剪中央$224 \times 224$区域作为输入。 此外，对于RGB（红、绿和蓝）颜色通道，我们分别*标准化*每个通道。 具体而言，该通道的每个值减去该通道的平均值，然后将结果除以该通道的标准差。


```python
# 使用RGB通道的均值和标准差，以标准化每个通道
normalize = torchvision.transforms.Normalize(
    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

train_augs = torchvision.transforms.Compose([
    torchvision.transforms.RandomResizedCrop(224),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    normalize])

test_augs = torchvision.transforms.Compose([
    torchvision.transforms.Resize([256, 256]),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    normalize])
```




### 定义和初始化模型

我们使用在ImageNet数据集上预训练的ResNet-18作为源模型。 在这里，我们指定`pretrained=True`以自动下载预训练的模型参数。


预训练的源模型实例包含许多特征层和一个输出层`fc`。 此划分的主要目的是促进对除输出层以外所有层的模型参数进行微调。 下面给出了源模型的成员变量`fc`。


在ResNet的全局平均汇聚层后，全连接层转换为ImageNet数据集的1000个类输出。 之后，我们构建一个新的神经网络作为目标模型。 它的定义方式与预训练源模型的定义方式相同，只是最终层中的输出数量被设置为目标数据集中的类数（而不是1000个）。

在下面的代码中，目标模型`finetune_net`中成员变量`features`的参数被初始化为源模型相应层的模型参数。 由于模型参数是在ImageNet数据集上预训练的，并且足够好，因此通常只需要较小的学习率即可微调这些参数。

成员变量`output`的参数是随机初始化的，通常需要更高的学习率才能从头开始训练。 假设`Trainer`实例中的学习率为$\eta$，我们将成员变量`output`中参数的学习率设置为$10\eta$。


```python
pretrained_net = torchvision.models.resnet18(pretrained=True)

# `fc` of pretrained model
pretrained_net.fc

# construct target model
finetune_net = torchvision.models.resnet18(pretrained=True)
finetune_net.fc = nn.Linear(finetune_net.fc.in_features, 2)
nn.init.xavier_uniform_(finetune_net.fc.weight);
```




### 微调模型

1. 首先，我们定义了一个训练函数 `train_fine_tuning`，该函数使用微调，因此可以多次调用。
2. 我们[**使用较小的学习率**]，通过*微调*预训练获得的模型参数。
3. [**为了进行比较，**]我们定义了一个相同的模型，但是将其(**所有模型参数初始化为随机值**)。由于整个模型需要从头开始训练，因此我们需要使用更大的学习率。


```python
# 如果param_group=True，输出层中的模型参数将使用十倍的学习率
def train_fine_tuning(net, learning_rate, batch_size=128, num_epochs=5,
                      param_group=True):
    train_iter = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(
        os.path.join(data_dir, 'train'), transform=train_augs),
        batch_size=batch_size, shuffle=True)
    test_iter = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(
        os.path.join(data_dir, 'test'), transform=test_augs),
        batch_size=batch_size)
    devices = d2l.try_all_gpus()
    loss = nn.CrossEntropyLoss(reduction="none")
    if param_group:
        params_1x = [param for name, param in net.named_parameters()
             if name not in ["fc.weight", "fc.bias"]]
        trainer = torch.optim.SGD([{'params': params_1x},
                                   {'params': net.fc.parameters(),
                                    'lr': learning_rate * 10}],
                                lr=learning_rate, weight_decay=0.001)
    else:
        trainer = torch.optim.SGD(net.parameters(), lr=learning_rate,
                                  weight_decay=0.001)
    d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)


# fine-tuning
train_fine_tuning(finetune_net, 5e-5)

# scratch training
scratch_net = torchvision.models.resnet18()
scratch_net.fc = nn.Linear(scratch_net.fc.in_features, 2)
train_fine_tuning(scratch_net, 5e-4, param_group=False)
```

意料之中，微调模型往往表现更好，因为它的初始参数值更有效。




## 小结

- 迁移学习将从源数据集中学到的知识*迁移*到目标数据集，微调是迁移学习的常见技巧。
- 除输出层外，目标模型从源模型中复制所有模型设计及其参数，并根据目标数据集对这些参数进行微调。但是，目标模型的输出层需要从头开始训练。
- 通常，微调参数使用较小的学习率，而从头开始训练输出层可以使用更大的学习率。






# Bounding-box(边界框)

在图像分类任务中，我们假设图像中只有一个主要物体对象，我们只关注如何识别其类别。

然而，很多时候图像里有多个我们感兴趣的目标，我们不仅想知道它们的类别，还想得到它们在图像中的具体位置。 在计算机视觉里，我们将这类任务称为[**目标检测**]（object detection）或[**目标识别**]（object recognition）。

目标检测在多个领域中被广泛使用。 例如，在无人驾驶里，我们需要通过识别拍摄到的视频图像里的车辆、行人、道路和障碍物的位置来规划行进线路。 机器人也常通过该任务来检测感兴趣的目标。安防领域则需要检测异常目标，如歹徒或者炸弹。





## 边界框

在目标检测中，我们通常使用*边界框*（bounding box）来描述对象的空间位置。 边界框是矩形的，由矩形左上角的以及右下角的$x$和$y$坐标决定。 另一种常用的边界框表示方法是边界框中心的$(x, y)$轴坐标以及框的宽度和高度。

在这里，我们[**定义在这两种表示法之间进行转换的函数**]：`box_corner_to_center`从两角表示法转换为中心宽度表示法，而`box_center_to_corner`反之亦然。 输入参数`boxes`可以是长度为4的张量，也可以是形状为（$n$，4）的二维张量，其中$n$是边界框的数量。


```python
#@save
def box_corner_to_center(boxes):
    """从（左上，右下）转换到（中间，宽度，高度）"""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    boxes = torch.stack((cx, cy, w, h), axis=-1)
    return boxes


#@save
def box_center_to_corner(boxes):
    """从（中间，宽度，高度）转换到（左上，右下）"""
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    boxes = torch.stack((x1, y1, x2, y2), axis=-1)
    return boxes


#@save
def bbox_to_rect(bbox, color):
    # 将边界框(左上x,左上y,右下x,右下y)格式转换成matplotlib格式：
    # ((左上x,左上y),宽,高)
    return d2l.plt.Rectangle(
        xy=(bbox[0], bbox[1]), width=bbox[2]-bbox[0], height=bbox[3]-bbox[1],
        fill=False, edgecolor=color, linewidth=2)
```


我们将根据坐标信息[**定义图像中狗和猫的边界框**]。 图像中坐标的原点是图像的左上角，向右的方向为$x$轴的正方向，向下的方向为$y$轴的正方向。

我们可以[**将边界框在图中画出**]，以检查其是否准确。 画之前，我们定义一个辅助函数`bbox_to_rect`。 它将边界框表示成`matplotlib`的边界框格式。


```python
d2l.set_figsize()
img = d2l.plt.imread('../img/catdog.jpg')
d2l.plt.imshow(img)

# bbox是边界框的英文缩写
dog_bbox, cat_bbox = [60.0, 45.0, 378.0, 516.0], [400.0, 112.0, 655.0, 493.0]

boxes = torch.tensor((dog_bbox, cat_bbox))
box_center_to_corner(box_corner_to_center(boxes)) == boxes

fig = d2l.plt.imshow(img)
fig.axes.add_patch(bbox_to_rect(dog_bbox, 'blue'))
fig.axes.add_patch(bbox_to_rect(cat_bbox, 'red'))
```





## 小结

- 目标检测不仅可以识别图像中所有感兴趣的物体，还能识别它们的位置，该位置通常由矩形边界框表示。
- 我们可以在两种常用的边界框表示（中间，宽度，高度）和（左上，右下）坐标之间进行转换。




# Anchor box(锚框)

目标检测算法通常会在输入图像中采样大量的区域，然后判断这些区域中是否包含我们感兴趣的目标，并调整区域边界从而更准确地预测目标的*真实边界框*（ground-truth bounding box）。

这里介绍其中一种常用方法：以每个像素为中心，生成多个 scale 和 aspect ratio 不同的边界框，这些边界框被称为 *anchor box*（锚框）。




## 生成多个锚框

假设输入图像高度为 $h$，宽度为 $w$。我们以图像的每个像素为中心生成不同形状的锚框：

- 缩放比为 $s \in (0, 1]$，宽高比为 $r > 0$。
- 锚框的宽度和高度分别是 $hs\sqrt{r}$ 和 $hs/\sqrt{r}$。

要生成多个不同形状的锚框，设置许多 scale 取值 $s_1, \ldots, s_n$ 和许多 aspect ratio 取值 $r_1, \ldots, r_m$。

如果使用全部组合，则每个像素中心有 $nm$ 个锚框，全图共 $whnm$ 个，计算复杂度过高。**实践中只考虑包含 $s_1$ 或 $r_1$ 的组合：**

$$(s_1, r_1), (s_1, r_2), \ldots, (s_1, r_m), (s_2, r_1), (s_3, r_1), \ldots, (s_n, r_1).$$

即以同一像素为中心的锚框数量是 $n+m-1$，对于整个输入图像总共生成 $wh(n+m-1)$ 个锚框。

- `multibox_prior(data, sizes, ratios)`：生成以每个像素为中心具有不同形状的锚框。返回的张量形状为 *(批量大小, 锚框数量, 4)*。
- `show_bboxes(axes, bboxes, labels, colors)`：在图像上绘制多个边界框。




## 交并比（IoU）

衡量 anchor box 与 ground-truth bounding box 之间相似性的常用指标。其本质是 *Jaccard 系数*：

$$J(\mathcal{A},\mathcal{B}) = \frac{\left|\mathcal{A} \cap \mathcal{B}\right|}{\left| \mathcal{A} \cup \mathcal{B}\right|}.$$

对于两个边界框，它们的 Jaccard 系数通常称为*交并比*（intersection over union, IoU），即两个边界框相交面积与相并面积之比。

- 取值范围 $[0, 1]$：0 表示完全不重合，1 表示完全重合。
- `box_iou(boxes1, boxes2)`：计算两个锚框或边界框列表中成对的 IoU。




## 在训练数据中标注锚框

在训练集中，将每个锚框视为一个训练样本。为了训练目标检测模型，需要为每个锚框标注两类标签：

- *class*：锚框相关对象的类别；
- *offset*：真实边界框相对于锚框的偏移量。


### 将真实边界框分配给锚框

给定锚框 $A_1, \ldots, A_{n_a}$ 和真实边界框 $B_1, \ldots, B_{n_b}$（$n_a \geq n_b$），定义矩阵 $\mathbf{X} \in \mathbb{R}^{n_a \times n_b}$，其中 $x_{ij}$ 为锚框 $A_i$ 和真实边界框 $B_j$ 的 IoU。分配算法步骤：

1. 找到矩阵 $\mathbf{X}$ 中最大的元素 $x_{i_1 j_1}$，将 $B_{j_1}$ 分配给 $A_{i_1}$，然后丢弃第 $i_1$ 行和第 $j_1$ 列的所有元素。
2. 重复步骤 1，直到所有 $n_b$ 列都被丢弃。此时 $n_b$ 个锚框各自被分配了一个真实边界框。
3. 对剩余 $n_a - n_b$ 个锚框 $A_i$：在第 $i$ 行中找到与之 IoU 最大的 $B_j$，仅当该 IoU 超过预定阈值时才将 $B_j$ 分配给 $A_i$。

- `assign_anchor_to_bbox(ground_truth, anchors, device, iou_threshold)`：将最接近的真实边界框分配给锚框。


### 标记类别和偏移量

锚框 $A$ 的类别被标记为与分配到的 $B$ 相同。偏移量则根据 $B$ 和 $A$ 中心坐标的相对位置以及二者相对大小进行变换。给定中心坐标 $(x_a, y_a)$、$(x_b, y_b)$，宽高 $w_a, h_a, w_b, h_b$，可将 $A$ 的偏移量标记为：

$$\left( \frac{ \frac{x_b - x_a}{w_a} - \mu_x }{\sigma_x},
\frac{ \frac{y_b - y_a}{h_a} - \mu_y }{\sigma_y},
\frac{ \log \frac{w_b}{w_a} - \mu_w }{\sigma_w},
\frac{ \log \frac{h_b}{h_a} - \mu_h }{\sigma_h}\right)$$

常量默认值为 $\mu_x = \mu_y = \mu_w = \mu_h = 0$，$\sigma_x = \sigma_y = 0.1$，$\sigma_w = \sigma_h = 0.2$。

- `offset_boxes(anchors, assigned_bb, eps)`：对锚框偏移量做上述变换。
- `multibox_target(anchors, labels)`：使用真实边界框标记锚框的类别和偏移量。未分配到真实边界框的锚框被标记为*背景*（background），属于*负类*锚框，其余为*正类*锚框。返回值为三元组：偏移量、掩码（mask，用于过滤负类偏移）、类别标签。




## 使用非极大值抑制预测边界框

预测时为每张图像生成多个锚框，并预测每个锚框的类别和偏移量。

- `offset_inverse(anchors, offset_preds)`：根据带有预测偏移量的锚框来反推预测边界框坐标。

由于锚框众多，会产生许多围绕同一目标的相似预测边界框。使用 *non-maximum suppression*（NMS，非极大值抑制）来合并属于同一目标的相似预测框：

1. 将所有预测的非背景边界框按*置信度*（confidence，即最大类别预测概率）降序排序，生成列表 $L$。
2. 选取 $L$ 中置信度最高的预测边界框 $B_1$ 作为基准，删除所有与 $B_1$ 的 IoU 超过阈值 $\epsilon$ 的非基准边界框。
3. 选取下一个最高置信度的边界框 $B_2$ 重复操作，直到所有边界框都被遍历。

- `nms(boxes, scores, iou_threshold)`：按置信度降序排序，返回保留下来的边界框索引。
- `multibox_detection(cls_probs, offset_preds, anchors, nms_threshold, pos_threshold)`：将非极大值抑制应用于预测边界框。返回结果形状为 *(批量大小, 锚框数量, 6)*，最内层六个元素为 *(class_id, confidence, x1, y1, x2, y2)*，其中 `class_id = -1` 表示背景或被 NMS 移除。




## 小结

- 我们以图像的每个像素为中心生成不同形状的锚框。
- *IoU*（intersection over union）用于衡量两个边界框的相似性，是相交面积与相并面积的比率。
- 训练集中需要给每个锚框两种类型的标签：*class* 和相对于真实边界框的 *offset*。
- 预测期间可以使用 *NMS*（non-maximum suppression）来移除相似的预测边界框。






# Multiscale object detection(多尺度目标检测)

在 *anchor box* 一节中，我们以输入图像的每个像素为中心生成了多个锚框，但当输入图像较大时（如 $561 \times 728$），即使每个像素只生成 5 个锚框，全图也会得到超过 200 万个锚框，计算量过大。



## 多尺度锚框

减少锚框数量并不困难。可以在输入图像中均匀采样一小部分像素作为中心，并在不同尺度下生成不同数量和大小的锚框：

- 较小的目标在图像上出现的可能性更多样，使用较小的锚框检测时可以采样更多区域。
- 较大的目标可以采样较少区域。

通过定义*特征图*（feature map）的形状来确定锚框中心的均匀采样位置：

- 给定特征图宽高 `fmap_w`、`fmap_h`，将均匀采样 `fmap_h` 行和 `fmap_w` 列的像素作为锚框中心。
- 锚框坐标已经除以特征图宽高，因此值介于 0 和 1 之间，表示相对位置。

- `display_anchors(fmap_w, fmap_h, s)`：在指定形状的特征图上生成并显示锚框。

实验结论：

- 探测小目标：特征图大（如 $4 \times 4$），尺度小（如 0.15）。
- 探测中等目标：特征图缩小（如 $2 \times 2$），尺度增大（如 0.4）。
- 探测大目标：特征图最小（如 $1 \times 1$），尺度最大（如 0.8），锚框中心即图像中心。




## 多尺度检测

基于 CNN 实现多尺度目标检测的核心思想：

- 假设我们有 $c$ 张形状为 $h \times w$ 的特征图，生成 $hw$ 组锚框，每组包含 $a$ 个中心相同的锚框。
- 特征图在同一空间位置的 $c$ 个单元具有相同的*感受野*（receptive field），表征同一感受野内的输入图像信息。
- 因此可以将特征图同一空间位置的 $c$ 个单元变换为该位置生成的 $a$ 个锚框的类别和偏移量预测。
- 不同层的特征图在输入图像上具有不同大小的感受野，可用于检测不同大小的目标：靠近输出层的特征图单元具有更宽的感受野，可检测较大目标。




## 小结

- 在多个尺度下，可以生成不同尺寸的锚框来检测不同尺寸的目标。
- 通过定义特征图的形状，可以决定任何图像上均匀采样的锚框的中心。
- 使用输入图像在某个感受野区域内的信息，预测与该区域位置相近的锚框类别和偏移量。
- 利用深度神经网络在多个层次的图像分层表示，可以实现多尺度目标检测。






# Transposed convolution(转置卷积)

常规的卷积层和池化层通常会减少（下采样）输入图像的空间维度（高和宽）。然而在像素级分类的语义分割中，希望输入和输出的空间维度相同。*Transposed convolution*（转置卷积）正是用来逆转下采样导致的空间尺寸减小，即*上采样*（upsampling）。



## 基本操作

设输入张量形状 $n_h \times n_w$，卷积核 $k_h \times k_w$，步幅为 1 且无填充。基本运算过程：

1. 以步幅为 1 滑动卷积核窗口，每行 $n_w$ 次，每列 $n_h$ 次，共产生 $n_h n_w$ 个中间结果。
2. 每个中间结果都是 $(n_h + k_h - 1) \times (n_w + k_w - 1)$ 的张量，初始化为 0。
3. 输入张量每个元素乘以卷积核，所得 $k_h \times k_w$ 张量替换中间张量对应位置的部分。
4. 所有中间结果相加得到最终输出。

> 与常规卷积通过卷积核"减少"输入元素相反，**转置卷积通过卷积核"广播"输入元素**，产生大于输入的输出。

- `trans_conv(X, K)`：手动实现基本二维转置卷积。
- 高级 API：`nn.ConvTranspose2d(in_channels, out_channels, kernel_size, ...)`。



## 填充、步幅和多通道

与常规卷积不同：

- *padding*：被应用于**输出**（常规卷积应用于输入）。例如 padding=1 表示输出删除第一和最后的行与列。
- *stride*：被指定为**中间结果**（输出）的步幅。stride 越大，输出张量越大。
- *multi-channel*：与常规卷积运作方式相同。如果输入有 $c_i$ 通道，转置卷积为每个输入通道分配一个 $k_h \times k_w$ 的卷积核；指定多个输出通道时，每个输出通道有一个 $c_i \times k_h \times k_w$ 的卷积核。

特殊地：如果 $f$ 是卷积层，$g$ 是与 $f$ 具有相同超参数但输出通道数为输入通道数的转置卷积层，则 $g(f(\mathsf{X}))$ 的形状与 $\mathsf{X}$ 相同。



## 与矩阵变换的联系

可以用矩阵乘法实现卷积：将卷积核 $K$ 重写为含大量 0 的稀疏权重矩阵 $W$；将输入 $X$ 向量化为向量 $\mathbf{x}$；卷积输出 $\mathbf{y} = W\mathbf{x}$，重塑后得到 $Y$。

- `kernel2matrix(K)`：将卷积核重写为稀疏权重矩阵。

抽象来看：

- 卷积前向传播：$\mathbf{y} = W \mathbf{x}$；
- 卷积反向传播：依链式法则需乘以 $W^\top$。
- **转置卷积层交换了卷积层的正向传播函数和反向传播函数**：正向传播用 $W^\top$ 相乘，反向传播用 $W$ 相乘。

这便是"转置卷积"得名的原因。



## 小结

- 转置卷积通过卷积核"广播"输入元素，产生形状大于输入的输出。
- 如果将 $\mathsf{X}$ 输入卷积层 $f$ 获得输出 $\mathsf{Y}=f(\mathsf{X})$，并创建一个与 $f$ 有相同超参数但输出通道数为输入通道数的转置卷积层 $g$，则 $g(Y)$ 的形状与 $\mathsf{X}$ 相同。
- 可以使用矩阵乘法实现卷积。转置卷积层交换了卷积层的正向与反向传播函数。
