def shape_reward(x, b):
    offset = 20
    neg = +1.5 * x
    pos = +0.5 * x
    if b <= offset:
        res = pos if x >= 0 else neg
    else:
        res = x
    return res


reward = -10
b = 10
shaped_reward = shape_reward(reward, b)
print shaped_reward