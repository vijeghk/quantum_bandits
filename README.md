# quantum_bandits

# 安装依赖
pip install -r requirements.txt

# 预处理数据（根据配置从本地 MySQL 或 FASTA 文件加载）
python main.py --mode preprocess

# 运行所有算法（1200 轮，10 次重复）
python main.py --mode train --n_rounds 1200 --n_repeats 10

# 评估结果并生成图表
python main.py --mode evaluate
