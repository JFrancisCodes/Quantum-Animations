from tkinter import *
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
import matplotlib.figure
from mpl_toolkits.mplot3d import Axes3D
import tkinter.messagebox as tmsg
import numpy as np

###################################################################################################
#the mathematical engine

#the heat equation
x = np.linspace(0,1,100)
x0 = 0.25

t_heat = np.linspace(0,0.08,100)
u = lambda x,t,n,m: np.exp(-t*(n*np.pi)**2.0)*np.sin(n*np.pi*x) +  np.exp(-t*(m*np.pi)**2.0)*np.sin(m*np.pi*x)
u_t = lambda x,t,n,m: -(n*np.pi**2.0)*u(x,t,n,m) + -(m*np.pi**2.0)*u(x,t,n,m)

#the wave equation

t_wave = np.linspace(0,4,400)
f = lambda x,t,n,m: np.cos(n*np.pi*t)*np.sin(n*np.pi*x) + np.cos(m*np.pi*t)*np.sin(m*np.pi*x)
f_tt = lambda x,t,n,m: -(n*np.pi)**2.0*f(x,t,n,m) +  -(m*np.pi)**2.0*f(x,t,n,m)

#the schrodinger equation
t_schrodinger = np.linspace(0, 4, 800)
psi = lambda x,t,n,m: np.exp(-1j*t*0.5*(n*np.pi)**2.0)*np.sin(n*np.pi*x)  +  np.exp(-1j*t*0.5*(m*np.pi)**2.0)*np.sin(m*np.pi*x)

################################################################################################
class GUI(Tk):
    def __init__(self):
        super().__init__()
        #this fixes the size
        self.geometry("1600x850")
        self.maxsize(1600,850)
        self.minsize(1600,850)
        #title of our gui
        self.title("PDEs and Boundary Conditions")

        self.choice = StringVar()
        self.choice.set("Radio")
        self.choice.set("schrodinger")
 #########################################################################################################   
    def show_init_profile(self):
        self.ax.cla()
        self.ax2.cla()

        if self.choice.get() == "heat":
            #the superposition
            self.ax.plot(x, np.zeros(len(x)), u(x,0.0,self.first_harmonic.get(), self.second_harmonic.get()), c='red' )
            self.ax.set_title("Heat Equation, Superposition")

            self.ax.set_axis_off()
            self.ax.view_init(10,270)
            
            #the x axis and z axis
            self.ax.plot(x, np.zeros(len(x)), np.zeros(len(x)), c='white', linestyle='dashed')
            self.ax.plot(np.zeros(len(x)), np.zeros(len(x)), np.linspace(-2,2,100), c='white', linestyle='dashed')

            #the individual harmonics
            self.ax2.plot(x , np.zeros(len(x)), np.sin(self.first_harmonic.get()*np.pi*x), c='orange')
            self.ax2.plot(x , np.zeros(len(x)), np.sin(self.second_harmonic.get()*np.pi*x), c='yellow')
            self.ax2.set_title("Heat Equation, Individual Harmonics")

            self.ax2.set_axis_off()
            self.ax2.view_init(10,270)

            #the x axis and z axis
            self.ax2.plot(x, np.zeros(len(x)), np.zeros(len(x)), c='white', linestyle='dashed')
            self.ax2.plot(np.zeros(len(x)), np.zeros(len(x)), np.linspace(-2,2,100), c='white', linestyle='dashed')

            self.canvas.draw()
        elif self.choice.get() =="wave":
            #the superposition
            self.ax.plot(x, np.zeros(len(x)), f(x,0.0,self.first_harmonic.get(), self.second_harmonic.get()), c='red' )
            self.ax.set_title("Wave Equation, Superposition")

            self.ax.set_axis_off()
            self.ax.view_init(10,270)
            
            #the x axis and z axis
            self.ax.plot(x, np.zeros(len(x)), np.zeros(len(x)), c='white', linestyle='dashed')
            self.ax.plot(np.zeros(len(x)), np.zeros(len(x)), np.linspace(-2,2,100), c='white', linestyle='dashed')

            #the individual harmonics
            self.ax2.plot(x , np.zeros(len(x)), np.sin(self.first_harmonic.get()*np.pi*x), c='orange')
            self.ax2.plot(x , np.zeros(len(x)), np.sin(self.second_harmonic.get()*np.pi*x), c='yellow')
            self.ax2.set_title("Wave Equation, Individual Harmonics")
            
            self.ax2.set_axis_off()
            self.ax2.view_init(10,270)

            #the x axis and z axis
            self.ax2.plot(x, np.zeros(len(x)), np.zeros(len(x)), c='white', linestyle='dashed')
            self.ax2.plot(np.zeros(len(x)), np.zeros(len(x)), np.linspace(-2,2,100), c='white', linestyle='dashed')

            self.canvas.draw()

        elif self.choice.get() =="schrodinger":
            self.ax.plot(x, np.zeros(len(x)), np.real(psi(x,0.0,self.first_harmonic.get(), self.second_harmonic.get())), c='red' )
            self.ax.set_title("Schrodinger Equation, Superposition")
           
            self.ax2.plot(x , np.zeros(len(x)), np.sin(self.first_harmonic.get()*np.pi*x), c='orange')
            self.ax2.plot(x , np.zeros(len(x)), np.sin(self.second_harmonic.get()*np.pi*x), c='yellow')
            self.ax2.set_title("Schrodinger Equation, Individual Harmonics")
           
            self.ax.set_axis_off()
            self.ax2.set_axis_off()

            self.ax.view_init(10,270)
            self.ax2.view_init(10,270)

            #the x axis and z axis
            self.ax.plot(x, np.zeros(len(x)), np.zeros(len(x)), c='white', linestyle='dashed')
            self.ax.plot(np.zeros(len(x)), np.zeros(len(x)), np.linspace(-2,2,100), c='white', linestyle='dashed')

            self.ax2.plot(x, np.zeros(len(x)), np.zeros(len(x)), c='white', linestyle='dashed')
            self.ax2.plot(np.zeros(len(x)), np.zeros(len(x)), np.linspace(-2,2,100), c='white', linestyle='dashed')

            self.canvas.draw()
            
#########################################################################################################   
    def show_time_evolution(self):
        self.ax.cla()
        self.ax2.cla()
        n = self.first_harmonic.get()
        m = self.second_harmonic.get()
        z = np.real(psi(x0,t_schrodinger, n,m))
        y = np.imag(psi(x0,t_schrodinger,n,m))
        
        if self.choice.get() =="heat":
            for i in range(0,len(t_heat)):
                #first the superposition
                self.ax.plot(x, np.zeros(len(x)), u(x, t_heat[i],n,m), c='red')        
                self.ax.set_title('Heat Equation, Superposition')
                self.ax.set_axis_off()
                self.ax.view_init(10,270)
                #the axes
                self.ax.plot(x, np.zeros(len(x)), np.zeros(len(x)), c='white', linestyle='dashed')
                self.ax.plot(np.zeros(len(x)), np.zeros(len(x)), np.linspace(-2,2,100), c='white', linestyle='dashed')
                self.ax.text2D(0.5, 0.10, "t:"+str("%.3f"%(t_heat[i])), size=30, c='yellow',transform=self.ax.transAxes)

                #then the individual harmonics
                self.ax2.plot(x, np.zeros(len(x)), np.exp(-t_heat[i]*(n*np.pi)**2.0)*np.sin(n*np.pi*x),  c='orange')
                self.ax2.plot(x, np.zeros(len(x)), np.exp(-t_heat[i]*(m*np.pi)**2.0)*np.sin(m*np.pi*x),  c='yellow')
                self.ax2.set_title('Heat Equation, Individual Harmonics')
                self.ax2.set_axis_off()
                self.ax2.view_init(10,270)
                #the axes
                self.ax2.plot(x, np.zeros(len(x)), np.zeros(len(x)), c='white', linestyle='dashed')
                self.ax2.plot(np.zeros(len(x)), np.zeros(len(x)), np.linspace(-2,2,100), c='white', linestyle='dashed')
                self.ax2.text2D(0.5, 0.10, "t:"+str("%.3f"%(t_heat[i])), size=30, c='yellow',transform=self.ax2.transAxes)

                self.canvas.draw()
                self.canvas.start_event_loop(0.1)
                self.ax.cla()
                self.ax2.cla()



        if self.choice.get()=="wave":
            for i in range(0,len(t_wave)):
                #first the superposition
                self.ax.plot(x, np.zeros(len(x)), f(x, t_wave[i],n,m), c='red')        
                self.ax.set_title('Wave Equation, Superposition')
                self.ax.set_axis_off()
                self.ax.view_init(10,270)
                #the axes
                self.ax.plot(x, np.zeros(len(x)), np.zeros(len(x)), c='white', linestyle='dashed')
                self.ax.plot(np.zeros(len(x)), np.zeros(len(x)), np.linspace(-2,2,100), c='white', linestyle='dashed')
                self.ax.text2D(0.5, 0.10, "t:"+str("%.3f"%(t_wave[i])), size=30, c='yellow',transform=self.ax.transAxes)

                #then the individual harmonics
                self.ax2.plot(x, np.zeros(len(x)), np.cos(n*np.pi*t_wave[i])*np.sin(n*np.pi*x),  c='orange')
                self.ax2.plot(x, np.zeros(len(x)), np.cos(m*np.pi*t_wave[i])*np.sin(m*np.pi*x),  c='yellow')
                self.ax2.set_title('Wave Equation, Individual Harmonics')
                self.ax2.set_axis_off()
                self.ax2.view_init(10,270)
                #the axes
                self.ax2.plot(x, np.zeros(len(x)), np.zeros(len(x)), c='white', linestyle='dashed')
                self.ax2.plot(np.zeros(len(x)), np.zeros(len(x)), np.linspace(-2,2,100), c='white', linestyle='dashed')
                self.ax2.text2D(0.5, 0.10, "t:"+str("%.3f"%(t_wave[i])), size=30, c='yellow',transform=self.ax2.transAxes)

                self.canvas.draw()
                self.canvas.start_event_loop(0.1)
                self.ax.cla()
                self.ax2.cla()

        elif self.choice.get()=="schrodinger":
            #first the rotation of the axes to show the real and imaginary parts
            for i in range(0,76, 5):
                #for the superposition
                self.ax.set_title('Schrodinger Equation, Superposition')
                self.ax.plot(x,np.zeros(len(x)), np.real(psi(x,0.0,n,m)), c ='red')
                self.ax.scatter(x0,0,np.real(psi(x0,0.0,n,m)), c='orange', s= 50)
                #the x axis
                self.ax.plot(x, np.zeros(len(x)), np.zeros(len(x)), c='white', linestyle='dashed')
                #the y axis
                self.ax.plot(np.zeros(len(x)), x, np.zeros(len(x)), c='white', linestyle='dashed')
                #the z axis
                self.ax.plot(np.zeros(len(x)), np.zeros(len(x)), np.linspace(-2,2,100), c='white', linestyle='dashed')
                self.ax.set_axis_off()
                self.ax.view_init(10, 270+ i )
                self.ax.text2D(0.5, 0.10, "t: 0", size=30, c='yellow',transform=self.ax.transAxes)
                
                #for the individual harmonics
                self.ax2.set_title('Schrodinger Equation, Individual Harmonics')
                self.ax2.plot(x,np.zeros(len(x)), np.sin(n*np.pi*x), c ='orange')
                self.ax2.plot(x,np.zeros(len(x)), np.sin(m*np.pi*x), c ='yellow')

                #the x axis
                self.ax2.plot(x, np.zeros(len(x)), np.zeros(len(x)), c='white', linestyle='dashed')
                #the y axis
                self.ax2.plot(np.zeros(len(x)), x, np.zeros(len(x)), c='white', linestyle='dashed')
                #the z axis
                self.ax2.plot(np.zeros(len(x)), np.zeros(len(x)), np.linspace(-2,2,100), c='white', linestyle='dashed')
                self.ax2.set_axis_off()
                self.ax2.view_init(10, 270+ i )
                self.ax2.text2D(0.5, 0.10, "t: 0", size=30, c='yellow',transform=self.ax2.transAxes)


                self.canvas.draw()
                self.canvas.start_event_loop(0.1)
                self.ax.cla()
                self.ax2.cla()

            #then the rotation to see the beautiful pattern
            for i in range(0, len(t_schrodinger)):
                #first, for the superposition

                # the evolution of the point
                self.ax.set_title('Schrodinger Equation, Superposition')
                self.ax.scatter(0, np.imag(psi(x0,t_schrodinger[i], n,m)), np.real(psi(x0,t_schrodinger[i],n,m)), c='orange', s= 50)
                # the circular trajectory
                self.ax.plot(np.zeros(i+1), y[0:i+1], z[0:i+1], linestyle='dashed', c='orange')
                #the evolution of the sine wave
                self.ax.plot(x,np.imag(psi(x,t_schrodinger[i],n,m)), np.real(psi(x,t_schrodinger[i],n,m)), c ='red')
                # the evolution of the point being analyzed in real and imaginary space
                self.ax.scatter(x0, np.imag(psi(x0,t_schrodinger[i],n,m)), np.real(psi(x0,t_schrodinger[i],n,m)), c='orange', s= 50)
                
                self.ax.set_axis_off()
                #the x axis
                self.ax.plot(x, np.zeros(len(x)), np.zeros(len(x)), c='white', linestyle='dashed')
                #the y axis
                self.ax.plot(np.zeros(len(x)), np.linspace(-2,2,100), np.zeros(len(x)), c='white', linestyle='dashed')
                #the z axis
                self.ax.plot(np.zeros(len(x)), np.zeros(len(x)), np.linspace(-2,2,100), c='white', linestyle='dashed')
                self.ax.view_init(10, 345)
                self.ax.text2D(0.5, 0.10, "t:"+str("%.3f"%(t_schrodinger[i])), size=30, c='yellow',transform=self.ax.transAxes)
                
                #then, for the individual harmonics
                self.ax2.set_title('Schrodinger Equation, Individual Harmonics')
               
                #the evolution of the sine waves
                self.ax2.plot(x,np.imag(np.exp(-1j*(n*np.pi)**2.0*t_schrodinger[i]*0.5)*np.sin(n*np.pi*x)), np.real(np.exp(-1j*(n*np.pi)**2.0*t_schrodinger[i]*0.5)*np.sin(n*np.pi*x)), c ='orange')
                self.ax2.plot(x,np.imag(np.exp(-1j*(m*np.pi)**2.0*t_schrodinger[i]*0.5)*np.sin(m*np.pi*x)), np.real(np.exp(-1j*(m*np.pi)**2.0*t_schrodinger[i]*0.5)*np.sin(m*np.pi*x)), c ='yellow')
                
                self.ax2.set_axis_off()
                #the x axis
                self.ax2.plot(x, np.zeros(len(x)), np.zeros(len(x)), c='white', linestyle='dashed')
                #the y axis
                self.ax2.plot(np.zeros(len(x)), np.linspace(-2,2,100), np.zeros(len(x)), c='white', linestyle='dashed')
                #the z axis
                self.ax2.plot(np.zeros(len(x)), np.zeros(len(x)), np.linspace(-2,2,100), c='white', linestyle='dashed')
                self.ax2.view_init(10, 345)
                self.ax2.text2D(0.5, 0.10, "t:"+str("%.3f"%(t_schrodinger[i])), size=30, c='yellow',transform=self.ax2.transAxes)


                self.canvas.draw()
                self.canvas.start_event_loop(0.1)
                self.ax.cla()
                self.ax2.cla()
#########################################################################################################           
    def show_help(self):
        tmsg.showinfo("Tips", "This applet shows the evolution of the same initial data for different PDEs. You can toggle between the harmonics of two superimposed waves, and then look at how the initial data evolves over time.")
#############################################################################################################    
    def make_title(self):
        f1 = Frame(self, bg='red', borderwidth=10, relief=GROOVE)
        f1.pack(side=TOP, fill=X, pady=10)
        Label(f1, bg = "red", fg = "black", font = ("calibiri", 20, "bold"),  text = "PDEs and Boundary Conditions").pack()
##########################################################################################################################    
    def make_initial_conditions_frame(self):
        f2 = Frame(self, bg = "lightblue", borderwidth=10 , relief = RAISED)
        f2.pack(fill=X)
        Label(f2, bg = "lightblue", fg = "black", font = ("calibiri", 15, "bold"), text = "Initial Configuration").pack()

        heat = Radiobutton(f2, bg = "lightblue", fg='black',text="HEAT",  variable=self.choice, value="heat")
        heat.pack(side=LEFT, padx=10)

        wave = Radiobutton(f2, bg = "lightblue", fg='black',text="WAVE",  variable=self.choice, value="wave")
        wave.pack(side=LEFT,padx=10)

        schrodinger = Radiobutton(f2, bg = "lightblue", fg='black',text="SCHRODINGER",  variable=self.choice, value="schrodinger")
        schrodinger.pack(side=LEFT,padx=10)

        Label(f2, bg = "lightblue", fg ="black", font = ("calibiri", 10, "bold"), text="First Harmonic").pack(side = LEFT, padx=10)
        self.first_harmonic = Scale(f2, bg ='lightblue', fg='black',from_=1, to=10, orient=HORIZONTAL)
        self.first_harmonic.set(2)
        self.first_harmonic.pack(side = LEFT, padx=10)

        Label(f2, bg = "lightblue", fg ="black", font = ("calibiri", 10, "bold"), text="Second Harmonic").pack(side = LEFT, padx=10)
        self.second_harmonic = Scale(f2, bg ='lightblue', fg='black',from_=1, to=10, orient=HORIZONTAL)
        self.second_harmonic.set(3)
        self.second_harmonic.pack(side = LEFT, padx=10)

        b1 = Button(f2, fg = 'black', bg='red',text = "Show Initial Probability Distribution", font = ("calibiri", 12, "bold"), command = self.show_init_profile)
        b1.pack(side=LEFT, padx=20)
#########################################################################################################################################    
    def make_time_evolution_frame(self):
        f3 = Frame(self,bg = "lightblue", borderwidth = 10, relief = RAISED)
        f3.pack(fill=X)
        Label(f3, bg = "lightblue", fg = "black", font = ("calibiri", 15, "bold"),  text = "Time Evolution of Configuration").pack()
        
        b2 = Button(f3, fg='black',bg='red', text="Show Time Evolution", font = ("calibiri", 12, "bold"), command = self.show_time_evolution)
        b2.pack()
    
##############################################################################################################################################
    def make_plot(self):
        f4 = Frame(self, bg = "lightblue", borderwidth = 10, relief = RAISED)
        f4.pack(fill=X)
        self.fig = matplotlib.figure.Figure(edgecolor='black', facecolor='lightblue',linewidth=7)
        self.fig.set_size_inches(16,4.5, forward=True)
        self.ax = self.fig.add_subplot(1,2,1, projection='3d')
        self.ax2 = self.fig.add_subplot(1,2,2, projection='3d')

        self.ax.set_position([0.05, 0.1, 0.50, 0.8])  # Adjust these values as needed
        self.ax2.set_position([0.45, 0.1, 0.50, 0.8])  # Adjust these values as needed

        self.ax.set_axis_off()
        self.ax2.set_axis_off()

        self.ax.set_facecolor('black')
        self.ax2.set_facecolor('black')
        self.canvas = FigureCanvasTkAgg(self.fig, master=f4)  
        self.canvas.get_tk_widget().pack(pady=5)

        # Create Toolbar
        toolbar = NavigationToolbar2Tk(self.canvas, f4, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(fill=X)


        help = Button(f4, fg='black', bg='red', text="Help", font = ("calibiri", 12, "bold"), command = self.show_help)
        help.pack()
#############################################################################################################################################

#The main body of the program, which just calls all the functions from before

#instance of out GUI class
pde = GUI()

#methods that give life to the GUI
pde.make_title()
pde.make_initial_conditions_frame()
pde.make_time_evolution_frame()
pde.make_plot()
pde.show_init_profile()

#the infinite GUI loop
pde.mainloop()
