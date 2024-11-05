import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from model import SimpleTransformer
from dataset import generate_data_cycle
from dataset import generate_data_next


# 模型超参数
embed_size = 512
vocab_size = 10  # 假设数字范围是 0 到 9, 超过为0
input_length = 3
output_length = 1
fusion = 'max_pooling' # [sum, fc, max_pooling]

# 训练参数
num_epochs = 10
num_samples = 1000
learning_rate = 0.001
batch_size = 256

model_path = 'mp_next_number.pth'

# Val params
val_samples = 100

# Generating data
# inputs, targets = generate_data_cycle(
#     num_samples=num_samples,
#     sequence_length=input_length,
#     vocab_size=vocab_size
# )
inputs, targets = generate_data_next(
    num_samples=num_samples,
    vocab_size=vocab_size,
    input_length=input_length,
    output_length = output_length
)
inputs = torch.tensor(inputs)
targets = torch.tensor(targets)

plot_training_loss = True

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loss_fn = nn.CrossEntropyLoss()

model = SimpleTransformer(embed_size, input_length, vocab_size, fusion).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Create a TensorDataset to hold the inputs and targets
dataset = TensorDataset(inputs, targets)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

losses_per_epoch = []

# 训练模型
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for input_batch, target_batch in dataloader:
        input_batch, target_batch = input_batch.to(device), target_batch.to(device)
        optimizer.zero_grad()

        # 前向传播
        output = model(input_batch)

        # 计算损失
        # print(f"[TMP]: input_batch shape is {input_batch.shape}, target_batch shape is {target_batch.shape}, output_batch shape is {output.shape}")
        loss = loss_fn(output.view(-1, vocab_size), target_batch.view(-1))  # 只取最后一个时间步的预测
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}, Loss: {loss:.6f}")
    losses_per_epoch.append(loss)

if plot_training_loss:
    plt.plot(losses_per_epoch)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.savefig("training_loss.png")

torch.save(model.state_dict(), model_path)

model.eval()  # Set the model to evaluation mode

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
            predictions = model(sample_tensor)
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
