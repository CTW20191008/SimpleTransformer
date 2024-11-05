import torch


def generate_data_cycle(num_samples, sequence_length, vocab_size):
    inputs = torch.randint(0, vocab_size, (num_samples, sequence_length))
    targets = torch.roll(inputs, -1, dims=1)
    return inputs, targets


def generate_data_next(num_samples, vocab_size, input_length, output_length):
    inputs = []
    targets = []

    for _ in range(num_samples):
        sequences = []
        next_sequences = []
        number = next_number = 0
        for _ in range(input_length):
            number = torch.randint(0, vocab_size, (1,)).item()
            sequences.append(number)
        inputs.append(sequences)

        next_number = number
        for _ in range(output_length):
            next_number += 1
            if next_number == vocab_size:
                next_sequences.append(0)
            else:
                next_sequences.append(next_number)
        targets.append(next_sequences)  # 输出下一个数

    return inputs, targets


if __name__ == "__main__":
    # inputs, targets = generate_data(3, 3, 10)
    inputs, targets = generate_data_next(10, 3, 10)
    print(f"[INFO]: inputs are {inputs}")
    print(f"[INFO]: targets are {targets}")
