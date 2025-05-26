# %matplotlib notebook
import matplotlib.pyplot as plt
import time
import numpy as np
from random import random, randrange
from random import sample

__version__ = "0.1.3"
print(f"Version: {__version__}")

class Parameters:
    d: int  # number of dimensions

    n: int  # number of sensors
    N: list  # indexes of sensors
    s: dict  # sensors coordinates
    s_norms: dict

    m: int  # nuber of targets
    M: list  # targets indexes
    r: dict  # targets coordinates
    theta: list  # searching for    !!!! maybe not needed

    meas: dict  # measurments of distance between target and sensor

    init_coord: list  # r estimations    !!!! maybe not needed

    beta_1: float
    beta_2: float
    beta: float
    alpha: float
    gamma: float
    weight: list


def rho(point_1: float, point_2: float):
    """Calculate distance between point_1 and point_2"""
    diff = point_1 - point_2
    return sum(diff * diff)


def cond_num(matrix):
    eig = np.linalg.eig(matrix)[0]
    eig = sorted([abs(n) for n in eig if abs(n) > 0.00001])
    return eig[-1] / eig[0]


class Result:
    def __init__(self, **kwargs):
        self.__dict__.update(**kwargs)


class SPSA:
    def __init__(self, **kwargs):
        self.__dict__.update(**kwargs)
        self.Delta_abs_value = 1 / np.sqrt(self.d)

    def init_plot(self):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot()
        plt.ion()

        self.fig.show()
        self.fig.canvas.draw()

        # draw targets
        x = [i for i, j in self.r.values()]
        y = [j for i, j in self.r.values()]
        self.ax.plot(x, y, 'bx', markersize=8)

        x = [i for i, j in self.s.values()]
        y = [j for i, j in self.s.values()]
        self.ax.plot(x, y, 'gx', markersize=3)

    def update_matrix(self, method):
        cond_start = cond_num(self.weight)

        if method == "cheb":
            weight = self.cheb_acceleration(self.weight)

        elif method == "inv":
            dig = np.diag(np.diag(self.weight))
            inv = np.linalg.inv(dig)
            weight = np.matmul(self.weight, inv)

        elif method == "main":
            weight = self.weight

        cond = cond_num(weight)
        print(f"Previous condition number: {cond_start}\n New condition number: {cond}")
        return cond, weight

    def run(self, method, num_steps=20, eps=0.001):
        self.method = method
        cond, weight = self.update_matrix(method)
        self.init_plot()

        errors = {}
        theta_hat = {
            target:
                {
                    sensor: self.s.get(sensor) + np.array([np.sqrt(self.meas.get(target).get(sensor)), 0]) for sensor in
                    self.N
                } for target in self.M
        }

        for k in range(1, num_steps):  # шаги
            theta_new = {}
            err = 0

            for l in self.M:
                theta_new[l] = {}

                neibors = self.get_random_neibors(weight, 2)

                colors = ["blue", "red", "green", "orange", "yellow", "purple", "black", "gray"]
                for ind, i in enumerate(self.N):

                    if k == 0:
                        self.ax.plot(theta_hat[l][i][0], theta_hat[l][i][1], 'o', markersize=3,
                                     color=colors[ind])

                    coef1 = 1 if random() < 0.5 else -1
                    coef2 = 1 if random() < 0.5 else -1
                    delta = np.array([coef1 * self.Delta_abs_value, coef2 * self.Delta_abs_value])

                    x1 = theta_hat[l][i] + self.beta_1 * delta
                    x2 = theta_hat[l][i] - self.beta_2 * delta

                    y1 = self.f_l_i(l, i, x1, neibors)
                    y2 = self.f_l_i(l, i, x2, neibors)

                    spsa = (y1 - y2) / self.beta * delta / 2

                    neibors_i = neibors.get(i, [])

                    b = weight[i - 1]
                    theta_diff = [abs(b[j - 1]) * (theta_hat[l][i] - theta_hat[l][j]) for j in neibors_i]

                    theta_new[l][i] = theta_hat[l][i] - (self.alpha * spsa + self.gamma * sum(theta_diff))
                    if i == 8:
                        self.ax.plot([theta_hat[l][i][0], theta_new[l][i][0]], [theta_hat[l][i][1], theta_new[l][i][1]], markersize=2, color=colors[ind])
                    self.ax.plot(theta_new[l][i][0], theta_new[l][i][1], 'o', markersize=3, color=colors[ind])

                    err += self.compute_error(theta_new[l][i], self.r[l])

                self.fig.canvas.draw()
                print(f"Error - {err:.2f} on {k} step")
                time.sleep(1)

            theta_hat = theta_new.copy()
            errors[k] = err

            if err < eps or err > 1e+9:
                break
        self.errors = errors

        # Compute error for each target and sensor separately
        target_err = {}
        for target in self.M:
            target_err[target] = {sensor: self.compute_error(theta_hat[target][sensor], self.r[target]) for sensor in
                                  self.N}

        return Result(errors=errors, cond=cond, theta_hat=theta_hat, target_err=target_err)

    def f_l_i(self, l, i, r_hat_l, neibors):
        """ Calculate function f for target l and for sensor i
        :param l: target index
        :param i: i sensor index
        :param r_hat_1: x point at witch calculate
        :return: matrixe D for i sensor and l target
        """
        C = self.C_i(i, neibors)
        D = self.D_l_i(l, i, neibors)

        try:
            C_i_inv = np.linalg.inv(C)
        except Exception:
            C_i_inv = np.linalg.pinv(C)

        diff = r_hat_l - np.matmul(C_i_inv, D)
        return sum(diff * diff)

    def C_i(self, i, neibors):
        """ Calculate matrix C for sensor i
        :param i: index of i sensor
        :return: matrixe C for i sensor
        """
        C_i = [self.s.get(j) - self.s.get(i) for j in neibors.get(i)]
        return 2 * np.array(C_i)

    def D_l_i(self, l, i, neibors):
        """ Calculate matrix D for target l and for sensor i
        :param l: target index
        :param i: i sensor index
        :return: matrixe D for i sensor and l target
        """
        Dli = [self.calc_D_l_i_j(self.meas.get(l), i, j) for j in neibors.get(i)]
        return Dli

    def calc_D_l_i_j(self, meas_l: dict, i, j):
        """Calculate value of D_l_i_j
        :param meas_l: distances between l target and each sensor
        :param i: index of 1st sensor
        :param j: index of 2nd sensor
        :return: D_l_i[j] for vector D_l_i
        """
        return self.rho_overline(meas_l.get(i), meas_l.get(j)) + self.s_norms.get(j) - self.s_norms.get(i)

    def rho_overline(self, meas_1: float, meas_2: float):
        """Calculate difference between meas_1 and meas_2"""
        return meas_1 - meas_2

    def gen_new_coordinates(self, coords: np.array, R: float = 1):
        """ Add shift to coordinats within specified radius
        """
        phi = 2 * np.pi * random()
        rad = R * random()

        shift = rad * np.array([np.sin(phi), np.cos(phi)])
        return coords + shift

    def compute_error(self, vector_1, vector_2):
        return pow(sum(vector_1 - vector_2), 2)

    def get_random_neibors(self, weight, max=2):
        neibors_mat = (weight != 0).astype(int)
        np.fill_diagonal(neibors_mat, 0)

        # choose num random neibors from all neibors
        neibors = {}
        for sensor in self.N:
            neib = [ind + 1 for ind, sens in enumerate(neibors_mat[sensor - 1]) if sens == 1]
            if len(neib) > max:
                neib = sample(neib, max)
            neibors[sensor] = neib

        return neibors

    #  Optimization functions
    def cheb_polyn_mat(self, n, x, c2):
        cheb = [0] * 3
        cheb[0] = np.identity(len(x))
        cheb[1] = c2 * x

        for i in range(2, self.n + 1):
            next_cheb = 2 * c2 * np.matmul(x, cheb[1]) - cheb[0]
            cheb[0] = cheb[1]
            cheb[1] = next_cheb
        return cheb[min([n, 1])]

    def cheb_acceleration(self, mat):
        eigens = sorted(np.linalg.eig(mat)[0])
        cond = abs(eigens[-1]) / abs(eigens[1])

        c2 = (cond + 1) / (cond - 1)
        c3 = 2 / (eigens[-1] + eigens[1])

        eye_mat = np.identity(len(mat))
        print(cond)
        k = int(np.floor(np.sqrt(cond)))

        mat_k = self.cheb_polyn_mat(k, eye_mat - c3 * mat, c2)
        a_k = self.cheb_polyn_mat(k, np.array([1]), c2)
        cheb_pol = eye_mat - mat_k / a_k

        return cheb_pol


if __name__ == "__main__":
    par = Parameters()

    par.n = 5  # number of sensors
    par.N = {i for i in range(1, par.n + 1)}  # indexes of sensors
    par.s = {1: np.array([1, 2]),  # sensors coordinates
             2: np.array([3, 20]),
             3: np.array([10, 3]),
             4: np.array([20, 3]),
             5: np.array([3, 10])}
    par.s_norms = {i: sum(val * val) for i, val in par.s.items()}

    par.m = 1  # nuber of targets
    par.M = {i for i in range(1, par.m + 1)}  # targets indexes
    par.r = {1: np.array([40, 40])}  # targets coordinates
    par.theta = np.array([val for key, val in par.r.items()])  # searching for

    par.meas = {1: {
        1: rho(par.r.get(1), par.s.get(1)),  # measurments  from target to sensor
        2: rho(par.r.get(1), par.s.get(2)),
        3: rho(par.r.get(1), par.s.get(3)),
        4: rho(par.r.get(1), par.s.get(4)),
        5: rho(par.r.get(1), par.s.get(5))
    }
    }

    par.init_coord = np.array([5, 5])  # r estimations

    par.d = 2  # number of dimensions

    par.beta_1 = 0.5
    par.beta_2 = 0.5
    par.beta = par.beta_1 + par.beta_2

    par.alpha = 1 / 4
    par.gamma = 1 / 4

    par.weight = np.array([[2., 0., -1., -1., 0.],
                           [0., 2., 0., -1., -1.],
                           [-1., 0., 1., 0., 0.],
                           [-1., -1., 0., 3., -1.],
                           [0., -1., 0., -1., 3.]])

    par.b = 1


    def K(u):
        coef1 = 1 if u[0] > 0 else -1
        coef2 = 1 if u[1] > 0 else -1
        return np.array([1 / 4, 1 / 4])


    spsa = SPSA(**par.__dict__)
    spsa.run("cheb")
