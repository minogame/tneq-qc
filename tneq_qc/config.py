class Configuration:
    initialize_variance = 1e-1
    opt_einsum_optimize = 'greedy'
    contraction_engine = 'opt_einsum' # or 'QCTN'

class AgentBehavier:
    n_iter = 10
    estimation_iter = 100
    timeout = 1800
    # if we can finish % of the iterations before timeout, then we will continue
    allow_waiting_after_timeout = True
    allow_waiting_after_timeout_rate = 0.5
    max_abnormal_before_block = 10

class Experiment:
    evoluation_goal = 'a.npy'
    max_generation = 30
    random_init = True

class GenerationProperty:
    evaluate_repeat = 2
    # this allows job to be repeated even a hard timeout is reported
    still_allow_repeat_after_hard_timeout = True

    class SocietyProperty:
        n_societies = 2

        class Society:
            # define property of each society
            societies = [
                {
                    "n_individuals_span": 200,
                    "fitness_func": "FITNESS_FUNCS.defualt",
                    "evolution": [
                        {"ops": "EVOLVE_OPS.elimination", "n_individuals_survive": 100},
                        {"ops": "EVOLVE_OPS.mutation", "prob": 0.05},
                        {"ops": "EVOLVE_OPS.fillup", "adj_func": "Individual.naive_random_adj_matrix_with_sparsity_limitation"},
                    ],
                },
                {
                    "n_individuals_span": 200,
                    "fitness_func": "FITNESS_FUNCS.defualt",
                    "evolution": [
                        {"ops": "EVOLVE_OPS.elimination", "n_individuals_survive": 100},
                        {"ops": "EVOLVE_OPS.mutation", "prob": 0.05},
                        {"ops": "EVOLVE_OPS.fillup", "adj_func": "Individual.naive_random_adj_matrix_with_sparsity_limitation"},
                    ],
                },
            ]

class IndividualProperty:
    discard_hard_timeout_result = False

    class RandomInitializationProperty:
        tn_size = 4
        tn_rank = 2
        presented_shape = 2
        init_sparsity = -0.00001

class EvolutionProperty:
    elimiation_threshold = 0.8
    immigration_prob = 0
    immigration_number = 5
    crossover_alpha = 1
    mutation_prob = 0.05

class OverlordProperty:
    tik = 1

# print(AgentBehavier.__dict__)