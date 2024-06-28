from Base import NLS_FQ_solutions

class NLS_FQ_soliton(NLS_FQ_solutions):
    
    def __init__(self, L, N, dt, tf, pars, ti=0):
        super().__init__(L, N, dt, tf, pars, ti)
        self.soliton = "soliton"
        
        self.w = self.theo

    def theo(self, x, t):
        i = 1j

        # Get parameters
        b = self.pars["b"]
        V = self.pars["V"]
        g = self.pars["g"]
        x_0 = self.pars["x_0"]
        
        def Q_w(x):
            # Do calculations
            a_w = np.sqrt(1 + 16/3 * b * 0.04)
            output = np.sqrt(4 / (1 + a_w * np.cosh(2*x)))
            
            return output
        
        def psi_w(x):
            output = np.sqrt(b) * Q_w(np.sqrt(b) * x)
            return output

        return np.exp(i * (V/2 * (x - x_0) - V**2/4 * t + b*t + g)) * psi_w(x - x_0 - V*t)
    
    def plot(self, t):
        plt.plot(self.x, self.norm(self.w(self.x, t)))

    def plot_real(self, t):
        plt.plot(self.x, np.real(self.w(self.x, t)))

    def plot_imag(self, t):
        plt.plot(self.x, np.imag(self.w(self.x, t)))

    def animate_solution(self, path, N_frames):
        # Create figure
        fig, ax = plt.subplots()
        
        # For empty plot
        empty_line = np.full(len(self.x), fill_value=None)

        # Plot init instant
        line1, = ax.plot(self.x, self.norm(self.w(self.x, self.ti)))
    
        ax.set_title("NLS-FQ solution "+self.soliton)
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$|w|^2$")

        # Animate function
        def animate(j):

            # Update plot
            line1.set_data(self.x, self.norm(self.w(self.x, self.ti+j*self.dt)))
            return line1, 

        # Animación
        ani = animation.FuncAnimation(fig, animate, frames=np.arange(0, self.Nt, N_frames), interval=150)

        ani.save(path)
        return ani
