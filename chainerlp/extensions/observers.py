""" Observe some values """

from chainer.training.extensions.value_observation import observe_value


def observe_nzu(optimizer_name='main', observation_key='nzu'):
    """ Count the total number of zero updates """

    def target_func(trainer):
        """ Iterate every parameter and sum their nzu.  """
        link = trainer.updater.get_optimizer(optimizer_name).target
        return sum([p.update_rule.state['nzu'] for p in link.params()])

    return observe_value(observation_key, target_func)