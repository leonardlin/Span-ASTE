# Use pretrained SpanModel weights for prediction
import sys
sys.path.append("aste")
from wrapper import SpanModel

data_name = "14lap"

random_seed = 0
path_train = f"aste/data/triplet_data/{data_name}/train.txt"
path_dev = f"aste/data/triplet_data/{data_name}/dev.txt"
path_test = f"aste/data/triplet_data/{data_name}/test.txt"
save_dir = f"outputs/{data_name}/seed_{random_seed}"

model = SpanModel(save_dir=save_dir, random_seed=random_seed)
model.fit(path_train, path_dev)