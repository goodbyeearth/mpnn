# 普通训练
python main.py --env-name  simple_spread --num-agents 3 --save-dir g_agent3 --net gated_mpnn

# 课程迁移学习
python automate.py --env-name simple_spread --save-dir curr_agent4 --net gated_mpnn
