import gym
import random
import numpy as np
import tflearn
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from statistics import mean, median
from collections import Counter

if tflearn.tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tflearn.tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")

LR = 1e-3
env = gym.make('Acrobot-v1')
env.reset()
goal_steps = 500
score_requirement = -450
initial_games = 1000


def random_games():
    for episode in range(100):
        env.reset()
        for t in range(goal_steps):
            env.render()
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            if done:
                break


def runSim(inmodel=False, render=False, games=10000):
    training_data = []
    scores = []
    accepted_scores = []

    for z in range(games):
        if z % 100 == 0:
            print(z)
        score = 0
        game_memory = []
        prev_observation = []
        for _ in range(goal_steps):
            if render:
                env.render()  # PERFORMANCE REDUCING
            if not inmodel or len(prev_observation) == 0:
                action = random.randrange(-1, 2)
            else:
                action = np.argmax(model.predict(prev_observation.reshape(-1, len(prev_observation), 1))[0])
            observation, reward, done, info = env.step(action)

            # Action taken based on previous observation therefore save w/ prev
            if len(prev_observation) > 0:
                game_memory.append([prev_observation, action])

            prev_observation = observation
            score += reward  # How well ai did
            if done:
                break

        if score >= score_requirement:
            accepted_scores.append(score)
            for data in game_memory:
                # convert to 1 hot 0/1 -> [1,0] / [0,1]
                if data[1] == 1:
                    output = [0, 0, 1]
                elif data[1] == 0:
                    output = [0, 1, 0]
                elif data[1] == -1:
                    output = [1, 0, 0]
                else:
                    print(data[1])
                    output = [0, 0, 0]
                training_data.append([data[0], output])
        env.reset()
        #print(score)

    training_data_save = np.array(training_data)
    np.save('saved.npy', training_data_save)

    print('Average accepted score:', mean(accepted_scores))
    print('Median accepted score:', median(accepted_scores))
    print(Counter(accepted_scores))
    print(max(accepted_scores))

    return training_data


# Reloading saved model requires model of same size
def neural_network_model(input_size):
    f = open("size.txt", "w+")
    f.write(str(input_size))
    f.close()

    network = input_data(shape=[None, input_size, 1], name='input')
    network = fully_connected(network, 128, activation='relu')
    network = dropout(network, 0.8)
    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.8)
    network = fully_connected(network, 512, activation='relu')
    network = dropout(network, 0.8)
    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.8)
    network = fully_connected(network, 128, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 3, activation='softmax')
    network = regression(network, optimizer='adam', learning_rate=LR,
                         loss='categorical_crossentropy', name='targets')
    model = tflearn.DNN(network, tensorboard_dir='log')

    return model  # Untrained


def train_model(training_data, model=False):
    X = np.array([i[0] for i in training_data]).reshape(-1, len(training_data[0][0]), 1)
    print(training_data)
    y = [i[1] for i in training_data]

    if not model:
        model = neural_network_model(input_size=len(X[0]))

    model.fit({'input': X}, {'targets': y}, n_epoch=4, snapshot_step=500, show_metric=True,
              run_id='tensor')
    return model


if input("(L)oad or create").upper() != "L":
    print("Creating training data")
    training_data = runSim(False, False, initial_games)
    print("Training data done")
    model = train_model(training_data)
    model.save('model.model')
else:
    model = neural_network_model(6)
    # load our saved model
    model.load('./model.model')

print("Done training")
print("Rerunning")
runSim(model, True, 200)
# print("Retraining")
# model = train_model(training_data)
# model.save('model.model')
# print("Final run")
# runSim(model, True, 100)
