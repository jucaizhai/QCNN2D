# QCNN2D

## 简介
本仓库提供基于 PyTorch 的二维/三维数值计算辅助函数，包括张量初始化、有限差分滤波权重以及边界条件处理，便于在 PDE/流体相关计算中快速搭建计算框架。

## 代码结构
- `QCNN2D.py`：运行入口封装器，会尝试加载同目录下的 `QCNN2D.PY` 主程序（如果你有该主程序文件，请放置在同一目录）。
- `AI4PDEs_utils.py`：2D/3D 张量创建与线性滤波/差分权重生成。
- `AI4PDEs_bounds.py`：2D/3D 边界条件处理函数。

## 依赖库
- Python 3.x
- `numpy`
- `torch`（如需 GPU，请安装与 CUDA 匹配的版本）

## 安装依赖
```bash
pip install numpy torch
```

## 运行与启动
### 方式一：运行主程序
如果你有主程序文件 `QCNN2D.PY`（仓库当前未包含该文件），请将它放在同目录后运行：
```bash
python QCNN2D.PY
```
也可以通过封装入口启动（等价执行 `QCNN2D.PY`）：
```bash
python QCNN2D.py
```

### 方式二：作为库在你的脚本中调用
本仓库当前主要提供函数模块，请在你的驱动脚本中导入并调用相关函数（如张量创建、边界条件设置等）：
- `AI4PDEs_utils.py`
- `AI4PDEs_bounds.py`
