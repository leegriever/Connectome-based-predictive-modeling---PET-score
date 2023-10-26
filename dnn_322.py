import numpy as np
import pandas as pd
import copy

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
from scipy.stats import pearsonr

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


def main():
    seed = 0
    torch.manual_seed(seed)

    test_size = 0.1
    use_pca = True
    conn_path = 'data/Glasser_conn_mat_322_subj.npy'
    scores_path = 'data/general_scores_322_subj.csv'

    X, y = read_data(conn_path, scores_path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=test_size, random_state=seed)

    # Normalization of features and behavioral scores
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_validation = scaler.transform(X_validation)
    X_test = scaler.transform(X_test)

    y_train = scaler.fit_transform(y_train).ravel()
    y_validation = scaler.transform(y_validation).ravel()
    y_test = scaler.transform(y_test).ravel()

    if use_pca:
        # Apply PCA for feature selection
        pca = PCA(n_components=255, svd_solver="full", random_state=seed)
        X_train = pca.fit_transform(X_train)
        # Apply PCA on the testing data
        X_validation = pca.transform(X_validation)
        X_test = pca.transform(X_test)

        model = nn.Sequential(
        nn.Linear(255, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 1),)

    else:
        model = nn.Sequential(
            nn.Linear(64620, 512),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),)

    # train 
    model = run_model_dl(X_train, X_validation, y_train, y_validation, model)

    # test results
    X_test = torch.from_numpy(X_test).float()
    y_pred_for_test = model(X_test)
    y_pred_for_test = y_pred_for_test.detach().numpy()

    mse = mean_squared_error(y_test, y_pred_for_test)
    r, p_val = pearsonr(y_test, y_pred_for_test)
    print(f'mse: {mse}, r: {r}, "p_value: {p_val}')


def run_model_dl(X_train, X_test, y_train, y_test, model):
    loss_fn = nn.MSELoss(reduction='mean')
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)
    n_epochs = 300
    batch_size = 322
    history = []
    best_mse = np.inf
    
    train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(X_train).float(),
                                                   torch.from_numpy(y_train).float())
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    X_test = torch.from_numpy(X_test).float()
    y_test = torch.from_numpy(y_test).float()

    for epoch in range(n_epochs):
        model.train()
        for batch, (X, y) in enumerate(train_loader):
            X, y = X.to("cpu"), y.to("cpu")

            # forward pass
            y_pred = model(X)
            y = y.reshape(-1, 1)
            loss = loss_fn(y_pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()
        print(f'epoch {epoch + 1}', "   lr  {:.6f}".format(scheduler.get_last_lr()[0]))

        # evaluate accuracy at end of each epoch
        with torch.no_grad():
            model.eval()
            y_pred = model(X_test)
            y_test = y_test.reshape(-1, 1)
            mse = loss_fn(y_pred, y_test)
            mse = float(mse)
            print(f'epoch: {epoch}, mse: {mse}')
            history.append(mse)
            if mse < best_mse:
                best_mse = mse
                best_weights = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_weights)
    return model


def read_data(conn_mat_filename: str, general_scores_filename: str):
    Glasser_conn_mat = np.load(conn_mat_filename)
    indices = np.triu_indices(360, k=1)

    # create X: rows for subjects, and flattened upper triangle mat for each subject
    X = []
    for i in range(Glasser_conn_mat.shape[2]):
        X.append(Glasser_conn_mat[:, :, i][indices])
    # convert X to data frame for pipeline parameters
    X = pd.DataFrame(X)

    # read scores
    y = pd.read_csv(general_scores_filename, header=None).to_numpy()
    return X, y


if __name__ == '__main__':
    main()


