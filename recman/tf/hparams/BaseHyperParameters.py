import itertools
from tensorboard.plugins.hparams import api as hp


class HParam:
    def __init__(self, name, default_value):
        assert name

        self._name = name
        self._default_value = default_value
        self._hparam = None

    def __call__(self, domain=None):
        if domain is None:
            domain = [self._default_value]

        try:
            self._hparam = hp.HParam(
                self._name,
                domain=domain if isinstance(domain, hp.Domain) else hp.Discrete(domain),
                display_name=self._name,
            )
            self._domain = [d for d in self._hparam.domain.values]
            self._advanced_dtype = False
            return self
        except ValueError:
            valid_domain = [str(d) for d in domain]
            self._domain = domain
            self._hparam = hp.HParam(
                self._name, domain=hp.Discrete(valid_domain), display_name=self.name
            )
            self._advanced_dtype = True
            return self
        except:
            raise RuntimeError

    @property
    def advanced_dtype(self):
        return self._advanced_dtype

    @property
    def name(self):
        return self._name

    @property
    def tf_hparam(self):
        return self._hparam

    @property
    def hp_domain(self):
        return self._domain

    @property
    def default_value(self):
        return self._default_value


class BaseHyperParameters(dict):

    LearningRate = "learning_rate"
    Optimizer = "optimizer"

    def __init__(self):
        dict.__init__(self)

        self.add_param(self.LearningRate, 0.001)
        self.add_param(self.Optimizer, "adam")

    def add_param(self, name, default_val):
        self[name] = HParam(name, default_val)()

    def _possible_values(self, hparam: HParam):
        if isinstance(hparam.tf_hparam.domain, hp.Discrete):
            return [(hparam.name, val) for val in hparam.hp_domain]

        raise NotImplementedError

    def grid_search(self, print_hp=False):
        for bags in list(
            itertools.product(
                *list([self._possible_values(param) for param in self.values()])
            )
        ):
            dict_bag = dict(bag for bag in bags)
            if print_hp:
                print(dict_bag)

            yield dict_bag
