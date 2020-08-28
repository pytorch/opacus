#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import argparse
import glob
import math
import random
import string
import time
import unicodedata

import torch
import torch.nn as nn
from opacus import PrivacyEngine
from opacus.layers import DPLSTM
from sklearn.metrics import balanced_accuracy_score
from tqdm import tqdm


parser = argparse.ArgumentParser(
    description="PyTorch Name language classification DP Training"
)
parser.add_argument(
    "--training-path",
    type=str,
    help="Path to training set of names (ie. /home/[username]/names/*.txt)",
)
parser.add_argument(
    "--device",
    type=str,
    default="cuda",
    help="GPU ID for this process (default: 'cuda')",
)
parser.add_argument(
    "-b",
    "--batch-size",
    default=800,
    type=int,
    metavar="N",
    help="mini-batch size (default: 800)",
)
parser.add_argument(
    "--n-hidden", default=128, type=int, help="LSTM hidden state dimensions"
)
parser.add_argument(
    "--max-seq-length", default=15, type=int, help="Maximum sequence length"
)
parser.add_argument(
    "--learning-rate",
    default=2.0,
    type=float,
    metavar="LR",
    help="initial learning rate",
)
parser.add_argument(
    "--iterations", type=int, default=1000, help="Number of training iterations"
)
parser.add_argument(
    "--train-eval-split",
    type=float,
    default=0.8,
    help="Fraction of data to utilize for training (rest for evaluation)",
)
parser.add_argument(
    "--sigma",
    type=float,
    default=1.0,
    metavar="S",
    help="Noise multiplier (default 1.0)",
)
parser.add_argument(
    "-c",
    "--max-per-sample-grad-norm",
    type=float,
    default=1.5,
    metavar="C",
    help="Clip per-sample gradients to this norm (default 1.0)",
)
parser.add_argument(
    "--disable-dp",
    action="store_true",
    default=False,
    help="Disable privacy training and just train with vanilla SGD",
)
parser.add_argument(
    "--delta",
    type=float,
    default=8e-5,
    metavar="D",
    help="Target delta (default: 1e-5)",
)

# Print the evaluation accuracy every 'print_every' iterations
print_every = 5

"""
Dataset preparation : download the dataset and save it to the variable
'category_lines' which is a dict with key as language and value as list of
names belonging to that language. 'all_categories' is a list of supported
languages, and n_categories is the number of languages.
"""


# Turn a Unicode string to plain ASCII, thanks to http://stackoverflow.com/a/518232/2809427
def unicode_to_ascii(s, all_letters):
    return "".join(
        c
        for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn" and c in all_letters
    )


# Read a file and split into lines
def read_lines(filename, all_letters):
    with open(filename) as f_read:
        lines = f_read.read().strip().split("\n")
    return [unicode_to_ascii(line, all_letters) for line in lines]


def build_category_lines(all_filenames, all_letters):
    r"""
    Build the category_lines dictionary, a list of names per language
    Returns the dataset dict, list of languages, and number of classes
    """
    category_lines = {}
    all_categories = []

    for filename in all_filenames:
        category = filename.split("/")[-1].split(".")[0]
        all_categories.append(category)
        lines = read_lines(filename, all_letters)
        category_lines[category] = lines

    n_categories = len(all_categories)
    return category_lines, all_categories, n_categories


def split_data_train_eval(category_lines, frac):
    r"""
    Split the data into a training and a evaluation set with a specified split
    'frac' is the percentage of data tp retain in the training set
    Returns the training and the validation sets as dictionaries
    """
    category_lines_train = {}
    category_lines_eval = {}
    for key in category_lines.keys():
        category_lines_train[key] = []
        category_lines_eval[key] = []
    for key in category_lines.keys():
        for val in category_lines[key]:
            if random.uniform(0, 1) < frac:
                category_lines_train[key].append(val)
            else:
                category_lines_eval[key].append(val)
    return category_lines_train, category_lines_eval


def get_dataset_size(category_lines):
    return sum(len(category_lines[key]) for key in category_lines.keys())


def line_to_tensor(batch_size, max_seq_length, lines, all_letters, n_letters):
    r"""
    Turn a list of batch_size lines into a <line_length x batch_size> tensor
    where each element of tensor is index of corresponding letter in all_letters
    """
    tensor = torch.zeros(max_seq_length, batch_size).type(torch.LongTensor)
    for batch_idx, line in enumerate(lines):
        # Pad/truncate line to fit to max_seq_length
        padded_line = line[0:max_seq_length] + "#" * (max_seq_length - len(line))
        for li, letter in enumerate(padded_line):
            letter_index = all_letters.find(letter)
            tensor[li][batch_idx] = letter_index
    return tensor


"""
Definition of the model class. Model defined here is a character-level LSTM classifier
"""


class CharNNClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_letters, batch_size):
        super(CharNNClassifier, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.batch_size = batch_size

        self.embedding = nn.Embedding(n_letters, input_size)
        self.lstm = DPLSTM(input_size, hidden_size, batch_first=False)
        self.out_layer = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        input_emb = self.embedding(input)
        lstm_out, _ = self.lstm(input_emb, hidden)
        # batch dimension = 1 is needed throughout, so we add an additional
        # dimension and subsequently remove it before the softmax
        output = self.out_layer(lstm_out[-1].unsqueeze(0))
        return output[-1]

    def init_hidden(self):
        return (
            torch.zeros(1, self.batch_size, self.hidden_size),
            torch.zeros(1, self.batch_size, self.hidden_size),
        )


"""
Dataset iterator functions : for training the model, at every iteration a random
training batch is drawn, and the performance on the validation set
(in terms of balanced class accuracy) is reported every print_every iterations.
"""


def get_random_batch(
    category_lines, batch_size, all_categories, all_letters, n_letters, args, device
):
    categories = random.choices(
        all_categories, k=batch_size
    )  # Selects batch_size random languages
    lines = [random.choice(category_lines[category]) for category in categories]
    category_tensors = torch.LongTensor(
        [all_categories.index(category) for category in categories]
    )
    line_tensors = line_to_tensor(
        batch_size, args.max_seq_length, lines, all_letters, n_letters
    )
    return categories, lines, category_tensors.to(device), line_tensors.to(device)


def get_all_batches(
    category_lines,
    all_categories,
    all_letters,
    n_letters,
    batch_size,
    max_seq_length,
    device,
):
    all_lines = [(k, x) for k, l in category_lines.items() for x in l]
    num_samples = len(all_lines)
    batched_samples = [
        all_lines[i : i + batch_size] for i in range(0, num_samples, batch_size)
    ]
    for batch in batched_samples:
        categories, lines = map(list, zip(*batch))
        line_tensors = line_to_tensor(
            batch_size, max_seq_length, lines, all_letters, n_letters
        )
        category_tensors = torch.LongTensor(
            [all_categories.index(category) for category in categories]
        )
        yield categories, lines, category_tensors.to(device), line_tensors.to(device)


"""
Functions for model training and evaluation
"""


def train(rnn, criterion, optimizer, category_tensors, line_tensors, device):
    rnn.zero_grad()
    hidden = rnn.init_hidden()
    if isinstance(hidden, tuple):
        hidden = (hidden[0].to(device), hidden[1].to(device))
    else:
        hidden = hidden.to(device)
    output = rnn(line_tensors, hidden)
    loss = criterion(output, category_tensors)
    loss.backward()

    optimizer.step()

    return output, loss.data.item()


def evaluate(line_tensors, rnn, device):
    rnn.zero_grad()
    hidden = rnn.init_hidden()
    if isinstance(hidden, tuple):
        hidden = (hidden[0].to(device), hidden[1].to(device))
    else:
        hidden = hidden.to(device)
    output = rnn(line_tensors, hidden)
    return output


def category_from_output(output, all_categories):
    top_n, top_i = output.data.topk(1)  # Tensor out of Variable with .data
    category_i = top_i.flatten()
    return [all_categories[category] for category in category_i], category_i


def get_eval_metrics(
    rnn,
    category_lines,
    all_categories,
    all_letters,
    n_letters,
    batch_size,
    max_seq_length,
    device,
):
    pred = []
    truth = []
    for categories, _, _, line_tensors in get_all_batches(
        category_lines,
        all_categories,
        all_letters,
        n_letters,
        batch_size,
        max_seq_length,
        device,
    ):
        eval_output = evaluate(line_tensors, rnn, device)
        guess, _ = category_from_output(eval_output, all_categories)
        pred.extend(guess)
        truth.extend(categories)
    pred = pred[: min(len(pred), len(truth))]
    truth = truth[: min(len(pred), len(truth))]
    return balanced_accuracy_score(truth, pred)


def main():
    args = parser.parse_args()
    device = torch.device(args.device)

    all_filenames = glob.glob(args.training_path)
    all_letters = string.ascii_letters + " .,;'#"
    n_letters = len(all_letters)

    category_lines, all_categories, n_categories = build_category_lines(
        all_filenames, all_letters
    )
    category_lines_train, category_lines_val = split_data_train_eval(
        category_lines, args.train_eval_split
    )
    rnn = CharNNClassifier(
        n_letters, args.n_hidden, n_categories, n_letters, args.batch_size
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(rnn.parameters(), lr=args.learning_rate)

    if not args.disable_dp:
        privacy_engine = PrivacyEngine(
            rnn,
            batch_size=args.batch_size,
            sample_size=get_dataset_size(category_lines_train),
            alphas=[1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)),
            noise_multiplier=args.sigma,
            max_grad_norm=args.max_per_sample_grad_norm,
            batch_first=False,
        )
        privacy_engine.attach(optimizer)

    # Measure time elapsed for profiling training
    def time_since(since):
        now = time.time()
        s = now - since
        m = math.floor(s / 60)
        s -= m * 60
        return "%dm %ds" % (m, s)

    # Keep track of losses for tracking
    current_loss = 0

    start_time = time.time()
    for iteration in tqdm(range(1, args.iterations + 1)):
        # Get a random training input and target batch
        _, _, category_tensors, line_tensors = get_random_batch(
            category_lines_train,
            args.batch_size,
            all_categories,
            all_letters,
            n_letters,
            args,
            device,
        )
        output, loss = train(
            rnn, criterion, optimizer, category_tensors, line_tensors, device
        )
        current_loss += loss

        # Print iteration number, loss, name and guess
        if iteration % print_every == 0:
            acc = get_eval_metrics(
                rnn,
                category_lines_val,
                all_categories,
                all_letters,
                n_letters,
                args.batch_size,
                args.max_seq_length,
                device,
            )
            time_elapsed = time_since(start_time)

            if not args.disable_dp:
                epsilon, best_alpha = optimizer.privacy_engine.get_privacy_spent(
                    args.delta
                )
                print(
                    f"Iteration={iteration} / Time elapsed: {time_elapsed} / Loss={loss:.4f} / "
                    f"Eval Accuracy:{acc*100:.2f} / "
                    f"Ɛ = {epsilon:.2f}, 𝛿 = {args.delta:.2f}) for α = {best_alpha:.2f}"
                )
            else:
                print(
                    f"Iteration={iteration} / Time elapsed: {time_elapsed} / Loss={loss:.4f} / "
                    f"Eval Accuracy:{acc*100:.2f}"
                )


if __name__ == "__main__":
    main()
