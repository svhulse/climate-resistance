import numpy as np

from scipy.integrate import solve_ivp

class TempModel:

    def __init__(self, **kwargs):
        self.K = 2000
        self.fec_max = 2 / 365
        self.beta_max = 1 / 365
        self.mu = 0.2 * self.fec_max

        self.res1 = 1
        self.res2 = 0.25

        self.cost1 = 1
        self.cost2 = 0.95

        self.fec_threshold = 15
        self.trans_threshold = 15
        self.cure_threshold = 27.5

        self.cure_rate = 1      #Emme's rate: 0.8/365
        self.uncure_rate = 1    #Emme's rate: 0.8/365

        self.temp_min = 5
        self.temp_max = 30

        self.S1 = self.K*0.9
        self.S2 = 0
        self.I1 = self.K*0.1
        self.I2 = 0
        self.C1 = 0
        self.C2 = 0

        self.mode='default'
        self.n_years = 150
        self.N = 0
        self.t = 0

        for key, value in kwargs.items():
            setattr(self, key, value)

        self.temp_scale = self.temp_max - self.temp_min

    def temp(self, t):
        t = t * (2*np.pi / 365) - np.pi

        temp = (np.cos(t) + 1) / 2
        temp = temp * self.temp_scale
        temp = temp + self.temp_min

        return temp
    
    def births(self, temp, steepness = 10):
        b = self.fec_max / (1 + np.exp(-steepness*(temp - self.fec_threshold)))

        return b
    
    def beta(self, temp, steepness = 10):
        beta = self.beta_max / (1 + np.exp(-steepness*(temp - self.trans_threshold)))

        return beta
    
    def cure(self, temp, steepness = 10):
        rate = self.cure_rate / (1 + np.exp(-steepness*(temp - self.cure_threshold)))

        return rate

    def uncure(self, temp, steepness = 10):
        rate = self.uncure_rate / (1 + np.exp(-steepness*(temp - self.cure_threshold)))

        return 1 - rate
    
    def solve(self):
        X_0 = (self.S1, self.S2, self.I1, self.I2, self.C1, self.C2)

        def df(t, X):
            S1 = X[0]
            S2 = X[1]
            I1 = X[2]
            I2 = X[3]
            C1 = X[4]
            C2 = X[5]

            N = np.sum(X)

            temp = self.temp(t)

            dS1 = (S1+C1)*(self.births(temp)*self.cost1)*(1 - N/self.K) - S1*(self.res1*self.beta(temp)*(I1+I2)/N + self.mu)
            dS2 = (S2+C2)*(self.births(temp)*self.cost2)*(1 - N/self.K) - S2*(self.res2*self.beta(temp)*(I1+I2)/N + self.mu)

            dI1 = S1*self.res1*self.beta(temp)*(I1+I2)/N + C1*self.uncure(temp) - I1*(self.cure(temp) + self.mu)
            dI2 = S2*self.res2*self.beta(temp)*(I1+I2)/N + C2*self.uncure(temp) - I2*(self.cure(temp) + self.mu)

            dC1 = I1*self.cure(temp) - C1*(self.uncure(temp) + self.mu)
            dC2 = I2*self.cure(temp) - C2*(self.uncure(temp) + self.mu)

            return np.array((dS1, dS2, dI1, dI2, dC1, dC2))

        def df_trans(t, X):
            S1 = X[0]
            S2 = X[1]
            I1 = X[2]
            I2 = X[3]
            C1 = X[4]
            C2 = X[5]

            N = np.sum(X)

            temp = self.temp(t)

            dS1 = (S1+C1)*(self.births(temp)*self.cost1)*(1 - N/self.K) - S1*(self.res1*self.beta(temp)*(I1+I2)/N + self.mu)
            dS2 = (S2+C2)*(self.births(temp)*self.cost2)*(1 - N/self.K) - S2*(self.res2*self.beta(temp)*(I1+I2)/N + self.mu)

            dI1 = S1*self.res1*self.beta(temp)*(I1+I2+C1+C2)/N + C1*self.uncure(temp) - I1*(self.cure(temp) + self.mu)
            dI2 = S2*self.res2*self.beta(temp)*(I1+I2+C1+C2)/N + C2*self.uncure(temp) - I2*(self.cure(temp) + self.mu)

            dC1 = I1*self.cure(temp) - C1*(self.res1*self.beta(temp)*(I1+I2+C1+C2)/N + self.uncure(temp) + self.mu)
            dC2 = I2*self.cure(temp) - C2*(self.res2*self.beta(temp)*(I1+I2+C1+C2)/N + self.uncure(temp) + self.mu)

            return np.array((dS1, dS2, dI1, dI2, dC1, dC2))

        def df_vir(t, X):
            S1 = X[0]
            S2 = X[1]
            I1 = X[2]
            I2 = X[3]
            C1 = X[4]
            C2 = X[5]

            N = np.sum(X)

            temp = self.temp(t)

            dS1 = S1*(self.births(temp)*self.cost1)*(1 - N/self.K) - S1*(self.res1*self.beta(temp)*(I1+I2)/N + self.mu)
            dS2 = S2*(self.births(temp)*self.cost2)*(1 - N/self.K) - S2*(self.res2*self.beta(temp)*(I1+I2)/N + self.mu)

            dI1 = S1*self.res1*self.beta(temp)*(I1+I2)/N + C1*self.uncure(temp) - I1*(self.cure(temp) + self.mu)
            dI2 = S2*self.res2*self.beta(temp)*(I1+I2)/N + C2*self.uncure(temp) - I2*(self.cure(temp) + self.mu)

            dC1 = I1*self.cure(temp) - C1*(self.uncure(temp) + self.mu)
            dC2 = I2*self.cure(temp) - C2*(self.uncure(temp) + self.mu)

            return np.array((dS1, dS2, dI1, dI2, dC1, dC2))

        t_range = (0, self.n_years*365)
        t_eval = np.linspace(0, t_range[1], t_range[1])

        if self.mode == 'default':
            sol = solve_ivp(df, t_range, X_0, t_eval = t_eval, max_step=1)
        elif self.mode == 'vir':
            sol = solve_ivp(df_vir, t_range, X_0, t_eval = t_eval, max_step=1)
        elif self.mode == 'trans':
            sol = solve_ivp(df_trans, t_range, X_0, t_eval = t_eval, max_step=1)

        self.S1 = sol.y[0, :]
        self.S2 = sol.y[1, :]
        self.I1 = sol.y[2, :]
        self.I2 = sol.y[3, :]
        self.C1 = sol.y[4, :]
        self.C2 = sol.y[5, :]
        
        self.N = np.sum(sol.y, axis=0)

        self.t = sol.t

    def annual_mean(self,  val):
        data = getattr(self, val)
        data_annual = np.zeros(self.n_years)

        for i in range(self.n_years):
            inds = np.logical_and(self.t >= 365 * i, self.t < 365 * (i+1))
            data_annual[i] = np.mean(data[inds])

        return data_annual