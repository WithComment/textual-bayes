source ~/.bashrc
source venv/mad

cd /h/299/xiaowen/projects/textual-bayes

# python main.py \
#     +data=agreement \
#     +method=baselines \
#     method.perturber_select=simple_sample \
#     method.aggregator_select=FrequencyUQ

python main.py \
    +data=agreement \
    +method=bop \
    method.mcmc.likelihood_beta=1 \
    method.steps=50 \
    method.burn_in=5 \
    method.thinning=5 \
    method.num_repeats=1