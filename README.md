# 药物回归分析系统 - Drug Regression Analysis System
This project uses random forest to solve drug regression problems.
本项目使用随机森林回归算法解决药物回归预测问题。通过网格搜索优化超参数，对药物的生物活性进行精准预测，并提供全面的误差分析和可视化结果。

drug-regression-analysis/
├── main.py                 # 主程序文件
├── requirements.txt        # 项目依赖包
├── 回归预测.xlsx           # 数据文件（需自备）
├── README.md              # 项目说明文档
└── results/               # 结果输出目录（运行时创建）

pip install -r requirements.txt

python main.py
