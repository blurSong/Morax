# csv文件说明：

## BERT、GNMT、ViT模型：

此类模型输入均为（a，b）型向量，a表示vocab size，b表示vector dimension

VS - Vocab size

VD - Vector dimension

TYP - type
(0表示linear层，12表示softmax，13表示layer norm，10表示矩阵相乘)

CD - common dimension(用于矩阵相乘时同维度)

IDX - index (表示层次顺序，默认为-1)

APD - appending (linear层时表示输出，矩阵相乘时表示第二个层的次序。)

## DLRM模型：

IF - input feature

OF - output feature

TYP - type
(0表示linear层，14表示Relu，15表示Sigmoid)

IDX- 表示层间顺序

