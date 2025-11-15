import numpy as np

def rk4_step(t, y, dydt, h, **kwargs):
    
    k_1 = dydt(t, y, **kwargs)
    k_2 = dydt(t + h/2, y + k_1*h/2, **kwargs)
    k_3 = dydt(t + h/2, y + k_2*h/2, **kwargs)
    k_4 = dydt(t + h, y + k_3*h, **kwargs)

    y_new = y + h/6 * (k_1 + 2*k_2 + 2*k_3 + k_4)
    t_new = t + h
    return t_new, y_new

def main():
    t = 3
    y = np.array([-1,-1,-1])
    dydt = lambda t, y: np.array([0, 1, 2*t])
    h = 2.4
    duration = 5
    num_steps = int(duration/h)

    print(f"Step: {0}, Time: {t:.3f}, Y: {y}")
    for step in range(num_steps):
        t, y = rk4_step(t, y, dydt, h)
        print(f"Step: {step+1}, Time: {t:.3f}, Y: {y}")

    


if __name__ == "__main__":
    main()
