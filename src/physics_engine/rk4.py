import numpy as np

def rk4_step(t, y, dydt, h, **kwargs) -> tuple:
    """
    Calculates the state vector at the next time step based on the current time, state vector, and kwargs
    
    :param t: Current time
    :param y: Current State Vector (or scalar)
    :param dydt: Function which calculates the time deriviative of the state vector
    :param h: Time Step
    :param kwargs: Optinoal arguments to pass to *dydt*
    :return: (time at next step, State vector at next time step)
    :rtype: tuple
    """

    k_1 = dydt(t, y, **kwargs)
    k_2 = dydt(t + h/2, y + k_1*h/2, **kwargs)
    k_3 = dydt(t + h/2, y + k_2*h/2, **kwargs)
    k_4 = dydt(t + h, y + k_3*h, **kwargs)

    y_new = y + h/6 * (k_1 + 2*k_2 + 2*k_3 + k_4)
    t_new = t + h
    return (t_new, y_new)