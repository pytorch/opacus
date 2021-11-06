import numpy as np
import pandas as pd
import syft as sy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from opacus import PrivacyEngine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


COLNAMES = [
    "duration",
    "protocol_type",
    "service",
    "flag",
    "src_bytes",
    "dst_bytes",
    "land",
    "wrong_fragment",
    "urgent",
    "hot",
    "num_failed_logins",
    "logged_in",
    "num_compromised",
    "root_shell",
    "su_attempted",
    "num_root",
    "num_file_creations",
    "num_shells",
    "num_access_files",
    "num_outbound_cmds",
    "is_host_login",
    "is_guest_login",
    "count",
    "srv_count",
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "srv_rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "srv_diff_host_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
]

# 34 numerical columns are considered for training
NUMERICAL_COLNAMES = [
    "duration",
    "src_bytes",
    "dst_bytes",
    "wrong_fragment",
    "urgent",
    "hot",
    "num_failed_logins",
    "num_compromised",
    "root_shell",
    "su_attempted",
    "num_root",
    "num_file_creations",
    "num_shells",
    "num_access_files",
    "num_outbound_cmds",
    "count",
    "srv_count",
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "srv_rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "srv_diff_host_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
]


class Net(nn.Module):
    def __init__(self, input_dim, output_dim):
        """
        input_dim: number of input features.
        output_dim: number of labels.
        """
        super(Net, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        outputs = self.linear(x)
        return outputs


def train(model, device, federated_train_loader, optimizer, epoch):
    model.train()
    # Iterate through each gateway's dataset
    for idx, (data, target) in enumerate(federated_train_loader):
        batch_idx = idx + 1
        model.send(data.location)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        model.get()
        if batch_idx == len(federated_train_loader) or (
            batch_idx != 0 and batch_idx % LOG_INTERVAL == 0
        ):
            loss = loss.get()
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * BATCH_SIZE,
                    len(federated_train_loader) * BATCH_SIZE,
                    100.0 * batch_idx / len(federated_train_loader),
                    loss.item(),
                )
            )


def get_data():
    df = pd.read_csv(
        "http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data_10_percent.gz",
        names=COLNAMES + ["threat_type"],
    )[:100000]

    numerical_df = df[NUMERICAL_COLNAMES].copy()
    numerical_df = numerical_df.loc[:, (numerical_df != numerical_df.iloc[0]).any()]
    final_df = numerical_df / numerical_df.max()
    X = final_df.values
    print("Shape of feature matrix : ", X.shape)

    threat_types = df["threat_type"].values
    encoder = LabelEncoder()
    y = encoder.fit_transform(threat_types)
    print("Shape of target vector : ", y.shape)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y
    )
    print("Number of records in training data : ", X_train.shape[0])
    print("Number of records in test data : ", X_test.shape[0])
    print(
        "Total distinct number of threat types in training data : ", len(set(y_train))
    )
    print("Total distinct number of threat types in test data : ", len(set(y_test)))

    n_feature = X_train.shape[1]
    n_class = np.unique(y_train).shape[0]

    print("Number of training features : ", n_feature)
    print("Number of training classes : ", n_class)

    # Create pytorch tensor from X_train,y_train,X_test,y_test
    train_inputs = torch.tensor(X_train, dtype=torch.float).tag(
        "#iot", "#network", "#data", "#train"
    )
    train_labels = torch.tensor(y_train).tag("#iot", "#network", "#target", "#train")
    test_inputs = torch.tensor(X_test, dtype=torch.float).tag(
        "#iot", "#network", "#data", "#test"
    )
    test_labels = torch.tensor(y_test).tag("#iot", "#network", "#target", "#test")

    return train_inputs, train_labels, test_inputs, test_labels, n_feature, n_class


if __name__ == "__main__":
    hook = sy.TorchHook(torch)
    torch.manual_seed(1)
    device = torch.device("cpu")
    gatway1 = sy.VirtualWorker(hook, id="gatway1")
    gatway2 = sy.VirtualWorker(hook, id="gatway2")

    # Number of times we want to iterate over whole training data
    BATCH_SIZE = 1000
    EPOCHS = 2
    LOG_INTERVAL = 5
    lr = 0.01

    (
        train_inputs,
        train_labels,
        test_inputs,
        test_labels,
        n_feature,
        n_class,
    ) = get_data()

    # Send the training and test data to the gatways in equal proportion.
    train_idx = int(len(train_labels) / 2)
    test_idx = int(len(test_labels) / 2)

    gatway1_train_dataset = sy.BaseDataset(
        train_inputs[:train_idx], train_labels[:train_idx]
    ).send(gatway1)
    gatway2_train_dataset = sy.BaseDataset(
        train_inputs[train_idx:], train_labels[train_idx:]
    ).send(gatway2)
    gatway1_test_dataset = sy.BaseDataset(
        test_inputs[:test_idx], test_labels[:test_idx]
    ).send(gatway1)
    gatway2_test_dataset = sy.BaseDataset(
        test_inputs[test_idx:], test_labels[test_idx:]
    ).send(gatway2)

    # Create federated datasets, an extension of Pytorch TensorDataset class
    federated_train_dataset = sy.FederatedDataset(
        [gatway1_train_dataset, gatway2_train_dataset]
    )
    federated_test_dataset = sy.FederatedDataset(
        [gatway1_test_dataset, gatway2_test_dataset]
    )

    # Create federated dataloaders, an extension of Pytorch DataLoader class
    federated_train_loader = sy.FederatedDataLoader(
        federated_train_dataset, shuffle=True, batch_size=BATCH_SIZE
    )
    federated_test_loader = sy.FederatedDataLoader(
        federated_test_dataset, shuffle=False, batch_size=BATCH_SIZE
    )

    # Initialize the model
    model = Net(n_feature, n_class)

    # Initialize the SGD optimizer
    optimizer = optim.SGD(model.parameters(), lr=lr)
    privacy_engine = PrivacyEngine(
        model,
        sample_rate=0.01,
        alphas=[10, 100],
        noise_multiplier=1.3,
        max_grad_norm=1.0,
    )
    privacy_engine.attach(optimizer)

    for epoch in range(1, EPOCHS + 1):
        train(model, device, federated_train_loader, optimizer, epoch)
