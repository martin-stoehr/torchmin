from collections import namedtuple


# scalar function result (value)
sf_value = namedtuple('sf_value', ['f', 'grad', 'hessp', 'hess'])

# directional evaluate result
de_value = namedtuple('de_value', ['f', 'grad'])

# vector function result (value)
vf_value = namedtuple('vf_value', ['f', 'jacp', 'jac'])

