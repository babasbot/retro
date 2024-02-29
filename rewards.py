_MIN_REWARD = -10
_MAX_REWARD = 10

_SCORE_REWARD = 3
_SCROLL_REWARD = 5 
_LIFE_LOSS_PENALIZATION = _MIN_REWARD

def calculate_reward(info, next_info):
    reward = 0
    
    # penalize loosing lifes
    if info.lives() > next_info.lives():
        reward += _LIFE_LOSS_PENALIZATION 
        
    # reward scoring 
    if info.score() < next_info.score():
        reward += _SCORE_REWARD 

    # reward scrolling left 
    if info.xscroll() < next_info.xscroll():
        reward += _SCROLL_REWARD 

    # reward level progress 
    if info.world() < next_info.world() or info.level() < next_info.level():
        reward += _MAX_REWARD

    return _normalize(max(min(reward, _MAX_REWARD), _MIN_REWARD))

def _normalize(reward):
    return (2 * (reward - _MIN_REWARD) / (_MAX_REWARD - _MIN_REWARD)) - 1

