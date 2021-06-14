import numpy as np
from sklearn.datasets import make_blobs, make_circles
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import DBSCAN

import matplotlib.pyplot as plt

if __name__ == '__main__':
    '''make dataset one, gaussian mixture'''
    centers = np.array([[0.5, 2], [-1, -1], [1.5, -1]])
    d_x, d_y = make_blobs(n_samples=500, centers=centers, cluster_std=0.5, random_state=0)
    d_x = MinMaxScaler().fit_transform(d_x)

    '''make dataset two, smiley face'''
    # first making the circle
    circle_x, circle_y = make_circles(500, noise=0.05, factor=0.2, )
    circle_x = circle_x[circle_y == 0]
    circle_y = circle_y[circle_y == 0]
    # second making the eyes
    eye_1_x, eye_1_y = np.random.randn(25, 2), np.full((25,), fill_value=2)
    eye_1_x[:, 0] = (eye_1_x[:, 0] * 0.1) - 0.3
    eye_1_x[:, 1] = (eye_1_x[:, 1] * 0.1) + 0.35

    eye_2_x, eye_2_y = np.random.randn(25, 2), np.full((25,), fill_value=3)
    eye_2_x[:, 0] = (eye_2_x[:, 0] * 0.1) + 0.3
    eye_2_x[:, 1] = (eye_2_x[:, 1] * 0.1) + 0.35

    # third making the mouth
    mouth_x, mouth_y = make_circles(500, noise=0.05, factor=0.5, )
    mouth_x = mouth_x[mouth_y == 1]
    mouth_y = mouth_y[mouth_y == 1]

    mouth_y = mouth_y[mouth_x[:, 1] < -0.2]
    mouth_x = mouth_x[mouth_x[:, 1] < -0.2]

    d_x_2 = np.concatenate((circle_x, eye_1_x, eye_2_x, mouth_x))
    d_y_2 = np.concatenate((circle_y, eye_1_y, eye_2_y, mouth_y))
    d_x_2 = MinMaxScaler().fit_transform(d_x_2)

    fig, axes = plt.subplots(1, 2)
    fig.set_size_inches(12, 6)
    axes[0].scatter(d_x[:, 0], d_x[:, 1], c=d_y, cmap='Paired')
    axes[1].scatter(d_x_2[:, 0], d_x_2[:, 1], c=d_y_2, cmap='Paired')
    axes[0].grid(True)
    axes[1].grid(True)
    axes[0].set_title('Dataset 1')
    axes[1].set_title('Dataset 2')

    plt.show()

    min_samples = [5, 10, 15, 20, 25]
    epsilons = [0.1, 0.4, 0.7, 0.9]
    for min_sample in min_samples:
        for eps in epsilons:
            db = DBSCAN(eps=eps, min_samples=min_sample,)
            y_pred = db.fit_predict(d_x)
            db = DBSCAN(eps=eps, min_samples=min_sample,)
            y_pred_2 = db.fit_predict(d_x_2)
            fig, axes = plt.subplots(1, 2)
            fig.set_size_inches(12, 6)
            axes[0].scatter(d_x[:, 0], d_x[:, 1], c=y_pred, cmap='Paired')
            axes[1].scatter(d_x_2[:, 0], d_x_2[:, 1], c=y_pred_2, cmap='Paired')
            axes[0].grid(True)
            axes[1].grid(True)
            axes[0].set_title('Dataset 1: predicted label min_sample: {} eps:{}'.format(min_sample, eps))
            axes[1].set_title('Dataset 2: predicted label min_sample: {} eps:{}'.format(min_sample, eps))
            plt.show()

