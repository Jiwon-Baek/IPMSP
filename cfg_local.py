
class Configure:
    def __init__(self, n_job=100, n_machine=5 ):
        self.env = 'OE'
        self.use_vessl = 0
        self.load_model = False
        self.model_path = None

        self.n_episode = 1000
        self.eval_every = 100
        self.save_every = 1000
        self.num_job = n_job # 100
        self.num_machine = n_machine # 5
        self.weight_tard = 0.5
        self.weight_setup = 0.5

        self.lr = 1e-4
        self.gamma = 0.98
        self.lmbda = 0.95
        self.eps_clip = 0.2
        self.K_epoch = 1
        self.T_horizon = 1
        self.optim = "Adam"