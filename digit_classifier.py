from argparse import ArgumentParser
import numpy
import pandas as pd


def sigmoid(x):
    return 1.0 / (1 + numpy.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x)*(1-sigmoid(x))
    # return numpy.exp(-x) / ((numpy.exp(-x) + 1) ** 2)


def softmax(x):
    exp_x = numpy.exp(x - x.max())
    return exp_x / exp_x.sum()


# def softmax_derivative(x):
#     s = softmax(x)
#     return s * (1 - s)


axis = 1


# def total_error(x, y):
#     return (x - y).sum()


class NeuralNet(object):

    def __init__(self, *layers):
        self.num_layers = len(layers)
        self.layer_sizes = layers

        # layers = [layer + 1 for layer in layers]
        # layers[-1] -= 1

        # self.weights = [numpy.random.randn(l1, l2) * numpy.sqrt(1.0 / l1)
        #                 for l1, l2 in zip(layers[:-1], layers[1:])]
        self.weights = [numpy.random.randn(l2, l1) * numpy.sqrt(1.0 / l2)
                        for l1, l2 in zip(layers[:-1], layers[1:])]

    def forward_pass(self, ip):
        X = ip[:]
        intermediate_values = []
        intermediate_activations = [X]
        for weight in self.weights[:-1]:
            X = X @ weight.T
            intermediate_values.append(X)
            X = sigmoid(X)
            # X = numpy.apply_along_axis(sigmoid, axis, X)
            intermediate_activations.append(X)

        X = X @ self.weights[-1].T
        intermediate_values.append(X)

        X = numpy.apply_along_axis(softmax, axis, X)
        intermediate_activations.append(X)

        return intermediate_activations, intermediate_values

    def predict(self, X):
        return self.forward_pass(X)[0][-1]

    def back_propagation(self, y, intermediate_activations,
                         intermediate_values):
        deltas = []

        # deltas.append(
        #     (intermediate_activations[-1] - y) *
        #     numpy.apply_along_axis(softmax_derivative, axis,
        #                            intermediate_values[-1]))

        deltas.append((intermediate_activations[-1] - y))

        for layer in range(self.num_layers - 2, 0, -1):
            delta = ((deltas[-1] @ self.weights[layer]) *
                 numpy.apply_along_axis(sigmoid_derivative, axis,
                                        intermediate_values[layer-1]))
            deltas.append(delta)

        deltas = deltas[::-1]
        return deltas

    def update_weights(self, deltas, intermediate_activations, learning_rate):
        for layer in range(len(self.weights)):
            self.weights[layer] = self.weights[layer] - learning_rate * (
                    deltas[layer].T @ intermediate_activations[layer])

    def train(self, X, y, learning_rate, epochs=300, batch_size=100,
              test_input=None, test_output=None):
        num_batches = len(X) // batch_size
        if len(X) % batch_size != 0:
            num_batches += 1

        last_correct = 0

        for epoch in range(epochs):
            # loss = 0
            for batch_num in range(num_batches):
                # if abs(max(self.weights[0].max(), self.weights[1].max(),
                #        self.weights[2].max())) > 2:
                #     import ipdb; ipdb.set_trace()
                batch_start = batch_num * batch_size
                batch = X[batch_start:batch_start + batch_size]
                batch_y = y[batch_start:batch_start + batch_size]
                intermediate_activations, intermediate_values = \
                    self.forward_pass(batch)

                deltas = self.back_propagation(batch_y,
                                               intermediate_activations,
                                               intermediate_values)

                self.update_weights(deltas, intermediate_activations,
                                    learning_rate)

                # print('Output: {}'.format(self.predict(batch)))
                # print('Truth : {}'.format(batch_y))
                #
                # input('Press enter...')
                # loss += -numpy.sum(numpy.log(intermediate_activations[-1]) * \
                #         batch_y)/batch_y.shape[0]

            # print("Loss:", loss)
            # if epoch % 10 == 0:
            #     print('Finished epoch {}'.format(epoch))
                # print('Error: {}'.format(total_error(batch_y,
                #                                      self.predict(batch))))

            # print(self.weights)


def read_input(f1, f2=None):
    pixels = pd.read_csv(f1, names=range(784))
    # pixels[785] = 255

    X = numpy.array(pixels)
    X = X / 255.

    if f2 is not None:
        labels = pd.read_csv(f2, names=['label'])['label']

        label_values = {
            0: numpy.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            1: numpy.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0]),
            2: numpy.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0]),
            3: numpy.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0]),
            4: numpy.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0]),
            5: numpy.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0]),
            6: numpy.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0]),
            7: numpy.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0]),
            8: numpy.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0]),
            9: numpy.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1]),
        }

        new_labels = numpy.array([label_values[label] for label in labels])

        return X, new_labels
    else:
        return X


def get_output(y):
    res = []
    for x in y:
        res.append(numpy.argmax(x))

    return numpy.array(res)


if __name__ == '__main__':

    a = ArgumentParser()

    a.add_argument('input1')
    a.add_argument('input2')
    a.add_argument('input3')

    args = a.parse_args()

    X, labels = read_input(args.input1, args.input2)
    t_x = read_input(args.input3)

    indices = numpy.arange(X.shape[0])
    numpy.random.shuffle(indices)

    X = X[indices]
    labels = labels[indices]

    # test_data = numpy.array([[0.1, 0.2, 0.3, 0.4, 1],
    #                          [0.5, 0.4, 0.4, 0.2, 1]])
    #
    # test_output = numpy.array([[0, 1],
    #                            [1, 0]])

    train_data, train_output = X[:10000], labels[:10000]

    # print(train_data.shape, train_output.shape)
    # input('Press Enter...')

    # net = NeuralNet(4, 3, 7, 2)
    net = NeuralNet(784, 128, 64, 10)

    # res = net.predict(t_x)
    # correct = 0
    # for x, yy in zip(res, t_labels):
    #     if numpy.argmax(x) == numpy.argmax(yy):
    #         correct += 1
    #
    # print('Correct: {}'.format(correct/len(res)))

    net.train(train_data, train_output, learning_rate=5e-3, epochs=350,
              batch_size=64)

    res = net.predict(t_x)
    # correct = 0
    # for x, yy in zip(res, t_labels):
    #     if numpy.argmax(x) == numpy.argmax(yy):
    #         correct += 1
    #
    # print('Correct: {}'.format(correct/len(res)))

    with open('test_predictions.csv', 'w') as f:
        for i in res[:-1]:
            f.write('{}\n'.format(numpy.argmax(i)))
        f.write('{}'.format(numpy.argmax(res[-1])))
