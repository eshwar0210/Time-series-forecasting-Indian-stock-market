import os
import subprocess

# Ensure directories exist
if not os.path.exists("./logs"):
    os.makedirs("./logs")
if not os.path.exists("./logs/LongForecasting"):
    os.makedirs("./logs/LongForecasting")

seq_len = 30
model_name = "PathFormer"
root_path_name = "./dataset/weather"
data_path_name = "weather.csv"
model_id_name = "weather"
data_name = "custom"

for pred_len in [1]:
    cmd = [
        "python", "-u", "run.py",
        "--is_training", "1",
        "--root_path", root_path_name,
        "--data_path", data_path_name,
        "--model_id", f"{model_id_name}_{seq_len}_{pred_len}",
        "--model", model_name,
        "--data", data_name,
        "--features", "M",
        "--seq_len", str(seq_len),
        "--pred_len", str(pred_len),
        "--num_nodes", "21",
        "--layer_nums", "3",
        "--patch_size_list", "16", "12", "8", "4", "12", "8", "6", "4", "8", "6", "2", "12",
        "--residual_connection", "1",
        "--k", "2",
        "--d_model", "8",
        "--d_ff", "64",
        "--train_epochs", "30",
        "--patience", "10",
        "--lradj", "TST",
        "--itr", "1",
        "--batch_size", "256",
        "--learning_rate", "0.001"
    ]
    
    log_file = f"logs/LongForecasting/{model_name}_{model_id_name}_{seq_len}_{pred_len}.log"
    with open(log_file, "w") as log:
        subprocess.run(cmd, stdout=log)
