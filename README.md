# quantum_bandits

'''python
# Install dependencies
pip install -r requirements.txt

# Preprocess data (loads from local MySQL or FASTA file as per configuration)
python main.py --mode preprocess

# Run all algorithms (1200 rounds, 10 repeats)
python main.py --mode train --n_rounds 1200 --n_repeats 10

# Evaluate results and generate figures
python main.py --mode evaluate
'''
