import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
import util
from torchsummary import summary
import time
# train a naive linear model to memorize the space

# experiment-0.0
# the model can be used to memorize the mapping from vector to its label
# small amount(10000) of base vectors
# direct classification
# train = test
# linear model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Model_0_0(nn.Module):
    def __init__(self):
        super(Model_0_0, self).__init__()
        self.fc1 = nn.Linear(128, 1024)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 1024)
        self.fc4 = nn.Linear(1024, 1024)
        self.out = nn.Linear(1024, 10)
        
        # Random weight initialization
        # init.normal_(self.fc2.weight, mean=0, std=0.01)
        # init.constant_(self.fc2.bias, 0)  # initializing the bias to zero
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        x = self.relu(x)
        x = self.out(x)
        return x

def train_0(model, X, Y, epochs=10, batch_size=32):
    assert X.shape[0] == Y.shape[0]
    assert X.shape[1] == 128
    assert len(Y.shape) == 1
    X = torch.from_numpy(X).to(device)
    Y = torch.from_numpy(Y).to(device)
    model = model.to(device)


    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        test_acc, test_lat, test_loss = test_0_batched(model, test_X, test_y)
        print("acc: {}, latency: {}, test loss: {}".format(test_acc, test_lat, test_loss), end='\t')
        epoch_loss = 0
        for i in range(0, len(X), batch_size):
            optimizer.zero_grad()
            x = X[i:i+batch_size]
            y = Y[i:i+batch_size]
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * x.shape[0]
        epoch_loss /= X.shape[0]
        print("Epoch: {}, Loss: {}".format(epoch, loss.item()))

def test_0_batched(model, X, Y, batch_size=1000, device='cuda'):
    """
    Batched testing function.

    Parameters:
    - model: PyTorch model to test
    - X: Input data (numpy array of shape [n_samples, 128])
    - Y: Labels (numpy array with shape [n_samples])
    - batch_size: Size of each batch
    - device: Device on which to run the computations (e.g., 'cuda' or 'cpu')

    Returns:
    - accuracy: Proportion of correctly predicted samples
    - average_latency: Average time taken per sample for prediction
    """
    assert X.shape[0] == Y.shape[0]
    assert X.shape[1] == 128
    assert len(Y.shape) == 1

    model.eval()

    # Convert to PyTorch tensors and move to device
    X = torch.from_numpy(X).to(device)
    Y = torch.from_numpy(Y).to(device)

    correct = 0
    total_latency = 0
    avg_loss = 0
    loss_fn = nn.CrossEntropyLoss()

    with torch.no_grad():  # Disable gradient computation
        num_batches = len(X) // batch_size
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size

            x_batch = X[start_idx:end_idx]
            y_batch = Y[start_idx:end_idx]

            start_time = time.time()
            
            y_pred = model(x_batch)
            loss = loss_fn(y_pred, y_batch)
            avg_loss += loss.item() * x_batch.shape[0]
            # print(torch.nn.functional.softmax(y_pred, dim=1))
            _, predicted = torch.max(y_pred, 1)
            correct += (predicted == y_batch).sum().item()

            end_time = time.time()
            total_latency += (end_time - start_time)  # Measure time taken for prediction

        # Handle any remaining samples if the dataset size is not a multiple of batch_size
        # if len(X) % batch_size != 0:
        #     x_batch = X[num_batches * batch_size:]
        #     y_batch = Y[num_batches * batch_size:]

        #     start_time = time.time()
            
        #     y_pred = model(x_batch)
        #     loss = loss_fn(y_pred, y_batch)
        #     print(loss)
        #     avg_loss += loss.item() * x_batch.shape[0]
        #     _, predicted = torch.max(y_pred, 1)
        #     correct += (predicted == y_batch).sum().item()

        #     end_time = time.time()
        #     total_latency += (end_time - start_time)  # Measure time taken for prediction

    correct /= batch_size * num_batches
    total_latency /= batch_size * num_batches
    avg_loss /= batch_size * num_batches

    return correct, total_latency, avg_loss

# def test_0(model, X, Y):
#     assert X.shape[0] == Y.shape[0]
#     assert X.shape[1] == 128
#     assert len(Y.shape) == 1
#     model.eval()

#     X = torch.from_numpy(X).to(device)
#     Y = torch.from_numpy(Y).to(device)

    

#     correct = 0
#     total_latency = 0

#     with torch.no_grad():  # disable gradient computation
#         for i in range(len(X)):
#             start_time = time.time()
            
#             x = X[i].unsqueeze(0)  # add a batch dimension
#             y = Y[i].unsqueeze(0)
            
#             y_pred = model(x)
#             _, predicted = torch.max(y_pred, 1)
#             correct += (predicted == y).sum().item()

#             end_time = time.time()
#             total_latency += (end_time - start_time)  # measure time taken for prediction

#     accuracy = correct / len(X)
#     average_latency = total_latency / len(X)

#     # print('Accuracy: {}%'.format(accuracy))
#     # print('Average latency: {} ms'.format(average_latency))
#     return accuracy, average_latency

    

if __name__ == "__main__":
    # set random seeds
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    vecs = util.fvecs_read("data/siftsmall/siftsmall_learn.fvecs")
    vecs = vecs[:10000] # vecs.shape= [1000, 128]

    # normalize the vectors by min and max value in each dimension
    vecs = (vecs - vecs.min(axis=0)) / (vecs.max(axis=0) - vecs.min(axis=0))

    labels = np.arange(1000) # labels.shape = [1000]

    model = Model_0_0()

    # Experiment 0.1
    # X, Y_k = util.generate_query_data(vecs, k=10, n_query=2000000, random_seed=0)
    # print(vecs.shape, labels.shape, X.shape, Y_k.shape)
    # train_X, train_y = X[:1997000], Y_k[:1997000]
    # test_X, test_y = X[1997000:], Y_k[1997000:]

    # Experiment 0.2
    # X, Y_q_idx, Y_nn_idx = util.generate_query_data_v1(vecs, n_clusters=10, n_query=1000000, random_seed=0)
    # print(X.shape, Y_q_idx.shape, Y_nn_idx.shape)
    # train_X, train_y = X[:997000], Y_q_idx[:997000]
    # test_X, test_y = X[997000:], Y_q_idx[997000:]

    # Experiment 1.0
    X, Y_q_idx, Y_nn_idx = util.generate_query_data_v1(vecs, n_clusters=10, n_query=10000000, random_seed=0)
    print(X.shape, Y_q_idx.shape, Y_nn_idx.shape)
    train_X, train_y = X[:9990000], Y_nn_idx[:9990000]
    test_X, test_y = X[9990000:], Y_nn_idx[9990000:]

    train_0(model, train_X, train_y, epochs=1000, batch_size=5000)
    test_0_batched(model, test_X, test_y)
    summary(model, (32, 128))
    # test_0(model, vecs, labels)
