scores = []
choices = []
best_game_mem = []
best_score = 0
for each_game in range(10):
    score = 0
    game_memory = []
    best_game_mem_temp = []
    prev_obs = []
    env.reset()
    for _ in range(goal_steps):
        env.render()
        if len(prev_obs) == 0:
            action = random.randrange(0,2)
        else:
            # np.argmax converts 1 hot to 0/1
            action = np.argmax(model.predict(prev_obs.reshape(-1, len(prev_obs), 1))[0])
        choices.append(action)
        best_game_mem_temp.append(action)
        new_observation, reward, done, info = env.step(action)

        prev_obs = new_observation
        game_memory.append([new_observation, action])
        score += reward
        if done:
            break
    if score > best_score:
        best_score = score
        best_game_mem = best_game_mem_temp
    scores.append(score)

print('Average score:', mean(scores))
print(Counter(scores))
print(Counter(choices))
print(max(scores))
print(min(scores))