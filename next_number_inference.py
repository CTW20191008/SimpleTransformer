import torch
from model import SimpleTransformer
from dataset import generate_data_next


# 模型超参数
embed_size = 512
vocab_size = 10  # 假设数字范围是 0 到 9, 超过为0
input_length = 3
output_length = 1
fusion = 'max_pooling' # [sum, fc, max_pooling]

# Val params
val_samples = 100

model_path = 'mp_next_number.pth'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型
loaded_model = SimpleTransformer(embed_size, input_length, vocab_size, fusion).to(device)  # 创建模型实例
loaded_model.load_state_dict(torch.load(model_path))
loaded_model.eval()  # Set the model to evaluation mode

# Example sequence
sample_sequences, target_sequences = generate_data_next(val_samples, vocab_size, input_length, output_length)

right_total = 0
with torch.no_grad():  # Disable gradient computation for inference
    for i in range(0, len(sample_sequences)):
        sample_sequence = sample_sequences[i]
        target_sequence = target_sequences[i]
        try:
            # Add batch dimension and send to device
            sample_tensor = (
                torch.tensor(sample_sequence, dtype=torch.long).unsqueeze(0).to(device)
            )
            predictions = loaded_model(sample_tensor)
            predicted_index = predictions.argmax(
                -1
            )  # Get the index of the max log-probability for the last position
            # print(f"[TMP]: predicted_index shape is {predicted_index.shape}")

            # predicted_number = predicted_index[0, -1].item()  # Convert to Python number
            predicted_numbers = [predicted_index[0, i].item() for i in range(predicted_index.size(1))]
            print(f"Input Sequence: {sample_sequence}")
            print(f"Predicted Next Number: {predicted_numbers}, target: {target_sequence}")
            if predicted_numbers[0] == target_sequence[0]:
                right_total += 1
        except Exception as e:
            print(f"[ERROR]: exception is {e}")

print(f"[INFO]: {fusion} accuracy is {right_total/val_samples}")
