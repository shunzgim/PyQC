## PyQC
一种量子线路的经典模拟器。
### 安装

#### step1: 生成动态库
- `cd build`
- `cmake ..`
- `make`

#### step2: 虚拟环境
- `conda create -n pyqc_env python=3.8.3`
- `conda activate pyqc_env`

#### step3: pip 安装
- `pip install pyqc`

#### step4: 加载动态库
- 将 `pyqc/backends/simulator/libs` 中的 `.so` 文件放入安装环境的对应位置。








