from evaluator import BanditEvaluator, Visualizer
from config import Config

config = Config()
evaluator = BanditEvaluator(config)
visualizer = Visualizer(config)
regret_curves = evaluator.compute_regret_curves()
visualizer.plot_regret_curves(regret_curves, save_path=f"{config.OUTPUT_DIR}/regret_curves.png")