import numpy as np
import plotly.graph_objects as plt
import time


def gen_data():
    # init seed:
    np.random.seed(239)

    # init sizes:
    features_cnt = 2
    objects_cnt = 300

    # generate true weight:
    true_weight = np.random.uniform(low=0.0, high=1.0, size=features_cnt)

    # generate features:
    X = np.random.uniform(low=-4, high=4, size=(objects_cnt, features_cnt))

    # mixing matrix:
    mixing = np.array([[2, 0], [0, 3]])

    # independent feature scaling:
    X = X.dot(mixing)

    # generate gaussian noise:
    noise = np.random.normal(loc=0, scale=1, size=objects_cnt)

    # calculate labels:
    Y = X.dot(true_weight) + noise

    return X, Y, true_weight


def gradient_descent(X, Y, batch_size, alpha=0.01, eps=0.1, w_0=None, true_w=None,
                     isMomentum=False, isNesterov=False, beta=0.1):
    # default weights:
    if w_0 is None:
        w_0 = np.random.uniform(-1, 1, len(X[0]))

    # calculate permutation:
    permutation = np.random.permutation(len(X))

    # prepare lists for storage:
    times = list()
    weights = list()

    # initial actions:
    start_pos = 0
    w = w_0.copy()
    weights.append(w_0.copy())
    moment = 0
    nesterov = np.zeros_like(w)

    while np.linalg.norm(w - true_w) >= eps:
        # remember time:
        current_time = time.time_ns()

        # calculate batch indices:
        if start_pos + batch_size >= len(X):
            start_pos = 0
        end_pos = start_pos + batch_size
        indices = permutation[start_pos:end_pos]

        # get batch based on indices from permutation:
        BX = X[indices]
        BY = Y[indices]

        # calculate gradient
        if isNesterov:
            G = 2 * np.dot(BX.T, np.dot(BX, w - alpha * beta * nesterov) - BY) / len(BX)
        else:
            G = 2 * np.dot(BX.T, np.dot(BX, w) - BY) / len(BX)

        # shift the weights in the direction of the anti-gradient of the loss function:
        if isMomentum:
            moment = beta * moment + (1 - beta) * G
            w = w - alpha * moment
        else:
            if isNesterov:
                nesterov = beta * nesterov + (1 - beta) * G
                w = w - alpha * nesterov
            else:
                w = w - alpha * G

        # remember new weights:
        weights.append(w.copy())

        # calculate time for this iteration:
        times.append(time.time_ns() - current_time)

        # shift batch:
        start_pos = end_pos

    return np.array(weights), np.array(times)


def my_plot(weights, true_weight, X, Y, name):
    dx = np.max(np.abs(weights[:, 0] - true_weight[0])) * 1.1
    dy = np.max(np.abs(weights[:, 1] - true_weight[1])) * 1.1
    xy_range = min(true_weight[0] - dx, true_weight[1] - dy), max(true_weight[0] + dx, true_weight[1] + dy)
    num = 100
    x, y = np.linspace(xy_range[0], xy_range[1], num), np.linspace(xy_range[0], xy_range[1], num)
    A, B = np.meshgrid(x, y)
    levels = np.empty_like(A)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            w_tmp = np.array([A[i, j], B[i, j]])
            levels[i, j] = np.mean((np.dot(X, w_tmp) - Y) ** 2, axis=0)
    contour = plt.Contour(x=x, y=y, z=levels, ncontours=30, name='loss levels')
    w0 = weights[:, 0]
    w1 = weights[:, 1]
    w_path = plt.Scatter(x=w0[:-1], y=w1[:-1], mode='lines+markers', name='weight', marker=dict(size=7, color='green'))
    w_final = plt.Scatter(x=[w0[-1]], y=[w1[-1]], mode='markers', name='weight_final',
                          marker=dict(size=10, color='blue'))
    w_true_point = plt.Scatter(x=[true_weight[0]], y=[true_weight[1]], mode='markers', name='weight_true',
                               marker=dict(size=10, color='red'))
    fig = plt.Figure(data=[contour, w_path, w_final, w_true_point])
    fig.update_xaxes(type='linear', range=xy_range)
    fig.update_yaxes(type='linear', range=xy_range)
    fig.update_layout(title=name)
    fig.update_layout(height=700, width=700, margin=dict(l=50, r=50, b=50, t=100, pad=4),
                      paper_bgcolor='orange')
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig.update_traces(showlegend=True)
    fig.show()


def my_print(times, name):
    print("! " + name + ": ")
    print("Total time: " + str(np.sum(times) * 1e-6) + " milliseconds")
    print("Total iter: " + str(len(times)) + " iterations")
    print("Average time per iteration: " + str(np.mean(times) * 1e-6) + " milliseconds / iter")


def simple_gradient(X, Y, batch_size, alpha, w_0, true_w, name):
    weights, times = gradient_descent(X, Y, batch_size=batch_size, alpha=alpha, w_0=w_0, true_w=true_w)
    my_print(times, name)
    my_plot(weights, true_w, X, Y, name)


def moment_gradient(X, Y, batch_size, alpha, w_0, true_w, name, beta):
    weights, times = gradient_descent(X, Y, batch_size=batch_size, alpha=alpha, w_0=w_0, true_w=true_w,
                                      isMomentum=True, beta=beta)
    my_print(times, name)
    my_plot(weights, true_w, X, Y, name)


def nesterov_gradient(X, Y, batch_size, alpha, w_0, true_w, name, beta):
    weights, times = gradient_descent(X, Y, batch_size=batch_size, alpha=alpha, w_0=w_0, true_w=true_w,
                                      isNesterov=True, beta=beta)
    my_print(times, name)
    my_plot(weights, true_w, X, Y, name)


def standardization(x):
    result = x.copy()
    result = result - np.mean(x, axis=0)
    result = result / np.std(x, axis=0)
    return result


if __name__ == '__main__':
    XX, YY, WW = gen_data()
    init_w = np.random.uniform(-5, 5, len(XX[0]))

    simple_gradient(X=XX, Y=YY, batch_size=len(XX), alpha=0.01, w_0=init_w, true_w=WW,
                    name="simple gradient, batch_size = all")
    simple_gradient(X=XX, Y=YY, batch_size=16, alpha=0.01, w_0=init_w, true_w=WW,
                    name="minibatch gradient, batch_size = 16")
    simple_gradient(X=XX, Y=YY, batch_size=1, alpha=0.01, w_0=init_w, true_w=WW,
                    name="stochastic gradient, batch_size = 1")

    XX_st = standardization(XX)
    WW_st = np.dot(np.linalg.pinv(XX_st), YY)

    simple_gradient(X=XX_st, Y=YY, batch_size=len(XX), alpha=0.1, w_0=init_w, true_w=WW_st,
                    name="norm simple gradient, batch_size = all")
    simple_gradient(X=XX_st, Y=YY, batch_size=16, alpha=0.1, w_0=init_w, true_w=WW_st,
                    name="norm minibatch gradient, batch_size = 16")
    simple_gradient(X=XX_st, Y=YY, batch_size=1, alpha=0.1, w_0=init_w, true_w=WW_st,
                    name="norm stochastic gradient, batch_size = 1")

    moment_gradient(X=XX, Y=YY, batch_size=1, alpha=0.01, w_0=init_w, true_w=WW,
                    name="moment stochastic gradient, beta = 0.3", beta=0.3)
    moment_gradient(X=XX, Y=YY, batch_size=1, alpha=0.01, w_0=init_w, true_w=WW,
                    name="moment stochastic gradient, beta = 0.5", beta=0.5)
    moment_gradient(X=XX, Y=YY, batch_size=1, alpha=0.01, w_0=init_w, true_w=WW,
                    name="moment stochastic gradient, beta = 0.8", beta=0.8)

    nesterov_gradient(X=XX, Y=YY, batch_size=1, alpha=0.01, w_0=init_w, true_w=WW,
                      name="nesterov stochastic gradient, beta = 0.3", beta=0.3)
    nesterov_gradient(X=XX, Y=YY, batch_size=1, alpha=0.01, w_0=init_w, true_w=WW,
                      name="nesterov stochastic gradient, beta = 0.5", beta=0.5)
    nesterov_gradient(X=XX, Y=YY, batch_size=1, alpha=0.01, w_0=init_w, true_w=WW,
                      name="nesterov stochastic gradient, beta = 0.8", beta=0.8)
