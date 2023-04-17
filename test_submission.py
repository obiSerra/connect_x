from connectx.utils import get_win_percentages


agent = lambda obs, config: 0
agent_2 = lambda obs, config: 0


# f = open("./submission_2.py", "r")
# source = f.read()
# exec(source)

# agent_2 = agent

f = open("./submission.py", "r")
source = f.read()
exec(source)
# print(agent)

# print("submission.py (agent_1) vs submission_2.py (agent_2)")
# win = get_win_percentages(agent1=agent, agent2=agent_2)
# print(win)

win = get_win_percentages(agent1=agent, agent2="random")
print("vs random")
print(win)


win = get_win_percentages(agent1=agent, agent2="negamax")
print("vs negamax")
print(win)
