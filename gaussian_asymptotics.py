## theorem of asymptotics, mqt

from tkinter import *
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
import matplotlib.figure
import numpy as np
import tkinter.messagebox as tmsg
from PIL import Image, ImageTk

#these are defined here and show up later for gui purposes
key = 1
movie_mode = False

#the math stuff
##################################################################################################################################

#x = np.linspace(-100,100,1000)
#dx = x[1] - x[0]

#k = np.linspace(-100,100,1000)
#dk = k[1]-k[0]


#L2 norm, to check normalization
def L2_norm(psi, dx):
    norm = np.sqrt(np.trapz(np.abs(psi) ** 2, dx=dx))
    return norm

#analytic gaussian wave psi(x,t). This follows the convention of the latex file.
def gaussian_position_single(x, t, x0, p, sigma):

    coeff  = (1.0/(np.sqrt(2.0*np.pi)))*(((2.0*sigma**2.0)/np.pi)**(1.0/4.0))*np.sqrt(np.pi/(sigma**2.0 + 1j*t/2.0))
    a = (2.0*p*sigma**2.0 + 1j*(x-x0))/(2.0*sigma**2.0 + 1j*t)
    exponent = -(p*sigma)**2.0 + (sigma**2.0 + 1j*t/2.0)*a**2.0
    phase = 1j*x0*p
    return coeff*np.exp(exponent)*np.exp(phase)
   
# analytic gaussian wave for two wave functions in (x,t). This follows the convention of the latex file.
def gaussian_position_double(x01,p1,sigma1,w1,w2,x,t):
    psi_1 = w1*gaussian_position_single(x,t,x01,p1,sigma1)
    psi_2 = w2*gaussian_position_single(x,t,-x01,-p1,sigma1)

    return psi_1 + psi_2

#analytic gaussian in momentum space. This follows the convention of the latex file.
def gaussian_momentum_single(k, x0, p, sigma): 
    coeff = ((2.0*sigma**2.0)/np.pi)**(1.0/4.0)
    exponent = 1j*x0*p - 1j*x0*k -((k-p)*sigma)**2.0

    return coeff*np.exp(exponent) 

# analytic gaussian wave for two wave functions in (x,t). This follows the convention of the latex file.
def gaussian_momentum_double(x01,p1,sigma1,w1,w2,k):
    psi_1 = w1*gaussian_momentum_single(k,x01,p1,sigma1)
    psi_2 = w2*gaussian_momentum_single(k,-x01,-p1,sigma1)

    return psi_1 + psi_2

#this little gem here is used for dynamic time setting, based on the average and the mean
def f(x,y):
    b = 499.0/5.0
    c = 1.0 - 0.1*(499.0/5.0)
    e = (4900.0 - 5.0*b)/125.0
    a = (100.0 - 0.1*b - c - 2.5*e)/25.0

    return a*x + b*y + c + e*x*y

#this shows the initial position and momentum distribution.
def show_init_profile():
    
    global movie_mode
    global key
    if movie_mode==False: #should only work if the movie functions aren't running
        key = 1
        ax.cla()
        ax2.cla()
        if varprofile.get() == "Single Gaussian":

            x_new = np.linspace( gauss_avg.get() - 3.5*gauss_sigma.get(),gauss_avg.get() + 3.5*gauss_sigma.get()  ,100)
            ax.plot(x_new,np.abs(gaussian_position_single(x_new,0, gauss_avg.get(),gauss_mom.get(), gauss_sigma.get()))**2.0, c='darkblue')
            ax.set_title("Initial Gaussian Position Profile")
            ax.set_ylabel(r"$|\psi(x,0)|^2$")
            ax.set_xlabel("Position")
            k_new = np.linspace(gauss_mom.get() - 3.5*0.5/gauss_sigma.get(),gauss_mom.get() + 3.5*0.5/gauss_sigma.get(),100)
            ax2.plot(k_new,np.abs(gaussian_momentum_single(k_new,gauss_avg.get(),gauss_mom.get(), gauss_sigma.get()))**2.0, c='darkblue')
            ax2.set_title("Gaussian Momentum Profile")
            ax2.set_ylabel(r"$|\widehat{\psi(p,0)}|^2$")
            ax2.set_xlabel("Momentum")

            canvas.draw()
        elif varprofile.get() =="Double Gaussian":
            x_new = np.linspace(min([gauss_avg.get() - 3.5*gauss_sigma.get(), -gauss_avg.get() - 3.5*gauss_sigma.get()]), max([gauss_avg.get() + 3.5*gauss_sigma.get(),-gauss_avg.get() + 3.5*gauss_sigma.get()]),1000)
            ax.plot(x_new,np.abs(gaussian_position_double(gauss_avg.get(),gauss_mom.get(), gauss_sigma.get(), weight_1.get(), weight_2.get(),x_new,0)**2.0), c='darkblue')
            ax.set_title("Initial Gaussian Position Profiles")
            ax.set_ylabel(r"$|\psi(x,0)|^2$")
            ax.set_xlabel("Position")
            k_new = np.linspace(min([-gauss_mom.get() - 3.5*0.5/gauss_sigma.get(), gauss_mom.get() - 3.5*0.5/gauss_sigma.get()]),max([-gauss_mom.get() + 3.5*0.5/gauss_sigma.get(), gauss_mom.get() + 3.5*0.5/gauss_sigma.get()]),1000)
            ax2.plot(k_new,np.abs(gaussian_momentum_double(gauss_avg.get(), gauss_mom.get(), gauss_sigma.get(),weight_1.get(),weight_2.get(),k_new))**2.0, c='darkblue')
            ax2.set_title("Gaussian Momentum Profiles")
            ax2.set_ylabel(r"$|\widehat{\psi(p,0)}|^2$")
            ax2.set_xlabel("Momentum")

            canvas.draw()
        else:
            tmsg.showinfo("Nothing selected!", "You must choose an initial distribution!")
       
#this shows the time evolution in position space and checks the theorem
def show_time_evolution():
    global key
    global movie_mode #let the app know that now we are only outputting movie frames
    movie_mode = True
    key =0
    ax.cla()
    ax2.cla()
    #fix the shape parameters for the Gaussians
    fix_avg = gauss_avg.get()
    fix_sigma = gauss_sigma.get()
    fix_mom = gauss_mom.get()
    fix_w1 = weight_1.get()
    fix_w2 = weight_2.get()
   
    t = np.linspace(0, f(abs(fix_avg),fix_sigma), 50)
   
    if varprofile.get() =="Single Gaussian":
        for i in range(0,len(t)):
            #change the limits of plotting based on where the gaussian is
            new_avg = fix_avg + fix_mom*t[i]
            new_sigma = np.sqrt(fix_sigma**2.0 + (t[i]/(2.0*fix_sigma))**2.0)
            x_new = np.linspace( new_avg - 3.5*new_sigma,new_avg + 3.5*new_sigma  ,100)
            x_new_init = np.linspace(fix_avg -3.5*fix_sigma, fix_avg + 3.5*fix_sigma, 100)
            ax.plot(x_new_init,np.abs(gaussian_position_single(x_new_init, 0, fix_avg, fix_mom, fix_sigma))**2.0, label=r'$|\psi(x,0)|^2$', linestyle='dashed', c='darkblue')
            ax.plot(x_new, np.abs(gaussian_position_single(x_new, t[i], fix_avg, fix_mom, fix_sigma))**2.0, label=r"$|\psi(x,t)|^2$", c='white')
           
            ax.legend(loc=1)
            ax.set_title("Gaussian Time Evolved State, t: " +"%.2f"%(t[i]))
            ax.set_xlabel("Position")
            #ax.annotate("time: "+"%.2f"%(t_new[i]),(x_new[75], np.max(np.abs(gaussian_position_single(x_new, 0, fix_avg, fix_mom, fix_sigma))**2.0)*0.5))
    
        
            #ax2.plot(x_new,np.abs(gaussian_momentum_single(x_new/t_new[i],fix_avg,fix_mom, fix_sigma))**2.0, label=r'$|\widehat{\psi(x/t)}|^2$', linestyle='dashed')
            #ax2.plot(x_new, t_new[i]*np.abs(gaussian_position_single(x_new, t_new[i], fix_avg, fix_mom, fix_sigma))**2.0, label=r'$t|\psi(x,t)|^2$')
            k_new = np.linspace(fix_mom - 3.5*0.5/fix_sigma,fix_mom + 3.5*0.5/fix_sigma,100)
            ax2.plot(k_new,np.abs(gaussian_momentum_single(k_new,fix_avg,fix_mom, fix_sigma))**2.0, label=r'$|\widehat{\psi(p)}|^2$', linestyle='dashed', c='darkblue')
            ax2.plot(k_new, t[i]*np.abs(gaussian_position_single(k_new*t[i], t[i], fix_avg, fix_mom, fix_sigma))**2.0, label=r'$t|\psi(pt,t)|^2$', c='white')
           
            ax2.set_ylim([0, np.max(np.abs(gaussian_momentum_single(k_new,fix_avg,fix_mom, fix_sigma))**2.0) + 0.1*np.max(np.abs(gaussian_momentum_single(k_new,fix_avg,fix_mom, fix_sigma))**2.0)])
            ax2.legend(loc=1)
            ax2.set_title("Theorem of Asymptotics, t: "+"%.2f"%(t[i]))
            ax2.set_xlabel("Momentum")
            #ax2.annotate("time: "+"%.2f"%(t_new[i]),(x_new[75], np.max(t_new[i]*np.abs(gaussian_position_single(x_new, t_new[i], fix_avg, fix_mom, fix_sigma))**2.0)/2))
            
            canvas.draw()
            canvas.start_event_loop(0.1)
            ax.cla()
            ax2.cla()
    elif varprofile.get() =="Double Gaussian":
        
        for i in range(0,len(t)):
            #change the limits of plotting based on where the gaussians are
            new_avg_1 = fix_avg + fix_mom*t[i]
            new_avg_2 = -fix_avg - fix_mom*t[i]
            new_sigma = np.sqrt(fix_sigma**2.0 + (t[i]/(2.0*fix_sigma))**2.0)
            x_new = np.linspace(min([new_avg_1 - 3.5*new_sigma, new_avg_2 - 3.5*new_sigma]), max([new_avg_1 + 3.5*new_sigma,new_avg_2 + 3.5*new_sigma]),1000)
            x_new_init = np.linspace(min([fix_avg - 3.5*fix_sigma, -fix_avg - 3.5*fix_sigma]), max([fix_avg + 3.5*fix_sigma,-fix_avg + 3.5*fix_sigma]),1000)
            ax.plot(x_new_init,np.abs(gaussian_position_double(fix_avg, fix_mom, fix_sigma, fix_w1, fix_w2, x_new_init, 0))**2.0, label=r'$|\psi(x,0)|^2$', linestyle='dashed', c='darkblue')
            ax.plot(x_new, np.abs(gaussian_position_double(fix_avg, fix_mom, fix_sigma, fix_w1, fix_w2, x_new, t[i]))**2.0, label=r"$|\psi(x,t)|^2$", c='white')
            ax.legend(loc=1)
            ax.set_title("Gaussian Time Evolved States, t: " +"%.2f"%(t[i]))
            ax.set_xlabel("Position")
            #ax.annotate("time: "+"%.2f"%(t[i]),(x_new[750], np.max(np.abs(gaussian_position_double(fix_avg, fix_mom, fix_sigma, fix_w1, fix_w2, x, 0))**2.0)*0.5))
            
            #ax2.plot(x_new,np.abs(gaussian_momentum_double(fix_avg, fix_mom, fix_sigma,fix_w1,fix_w2,x_new/t[i]))**2.0, label=r'$|\widehat{\psi(x/t)}|^2$', linestyle='dashed')
            #ax2.plot(x_new,t[i]*np.abs(gaussian_position_double(fix_avg, fix_mom, fix_sigma, fix_w1,fix_w2, x_new, t[i]))**2.0, label=r'$t|\psi(x,t)|^2$')
            
            k_new = np.linspace(min([-fix_mom - 3.5*0.5/fix_sigma, fix_mom - 3.5*0.5/fix_sigma]),max([-fix_mom + 3.5*0.5/fix_sigma, fix_mom + 3.5*0.5/fix_sigma]),1000)
            ax2.plot(k_new,np.abs(gaussian_momentum_double(fix_avg,fix_mom, fix_sigma, fix_w1, fix_w2, k_new))**2.0, label=r'$|\widehat{\psi(p)}|^2$', linestyle='dashed', c='darkblue')
            ax2.plot(k_new, t[i]*np.abs(gaussian_position_double(fix_avg, fix_mom, fix_sigma, fix_w1, fix_w2, k_new*t[i], t[i]))**2.0, label=r'$t|\psi(pt,t)|^2$', c='white')
            ax2.set_ylim([0, np.max(np.abs(gaussian_momentum_double(fix_avg,fix_mom, fix_sigma, fix_w1, fix_w2, k_new))**2.0) + 0.1*np.max(np.abs(gaussian_momentum_double(fix_avg,fix_mom, fix_sigma, fix_w1, fix_w2, k_new))**2.0)])
            ax2.legend(loc=1)
            ax2.set_title("Theorem of Asymptotics, t: " +"%.2f"%(t[i]))
            ax2.set_xlabel("Momentum")
           
            #ax2.annotate("time: "+"%.2f"%(t[i]),(x_new[750],np.max(t[i]*np.abs(gaussian_position_double(fix_avg, fix_mom, fix_sigma, fix_w1, fix_w2, x_new, t[i]))**2.0)/2))
            
            canvas.draw()
            canvas.start_event_loop(0.1)
            ax.cla()
            ax2.cla()
    movie_mode = False #once frames are shown, set it to non movie mode.


def show_help(): #the help button
    tmsg.showinfo("Tips", "This applet has two parts. Firstly, you can pick one or two Gaussians to simulate, sliding the mean, variance and momentum. The second Gaussian will have the same variance, but its position and momentum are inverted. You can select how to weight each Gaussian as well. Once you hit 'show time evolution', the app will go into movie mode and will output 50 frames.")

def weight_warning(*args):
    #gives error message if weights are selected in single gaussian mode
    if varprofile.get()=='Single Gaussian':
        tmsg.showinfo("Change Selection", "Must select Double Gaussian for changing weights.")
    else:
        choose_function(*args)

def choose_function(*args): #only if init profile is selected, should the graph change (when key=1). In movie mode (when key=0), it should not
    global key
    if key ==1:
        show_init_profile()
    else:
        pass

###################################################################################################################
        
#asym is the instance of our gui/its an object
asym = Tk()
#this fixes the size
asym.geometry("1600x900")
asym.maxsize(1600,900)
asym.minsize(1600,900)
#title of our gui
asym.title("Theorem of Asymptotics")


#******* Title frame********
f1 = Frame(asym, bg = "red", borderwidth = 10, relief = GROOVE)
f1.pack(side = TOP,  fill=X, pady =10)
title = Label(f1, bg = "red", fg = "black", font = ("calibiri", 20, "bold"),  text = "Theorem of Asymptotics for Gaussian Distributions")
title.pack()
#******* *********

#******** this frame handles choosing initial distribution ********
f2 = Frame(asym, bg = "lightblue", borderwidth=10 , relief = RAISED)
f2.pack(fill = X)
Label(f2, bg = "lightblue", fg = "black", font = ("calibiri", 15, "bold"), text = "Initial Data in Position and Momentum Space").pack()

varprofile = StringVar()
varprofile.set("Radio")

single_gaussian = Radiobutton(f2, bg = "lightblue", fg='black',text="SINGLE GAUSSIAN",  variable=varprofile, value="Single Gaussian")
single_gaussian.pack()
double_gaussian = Radiobutton(f2, bg = "lightblue", fg='black',text="DOUBLE GAUSSIAN",  variable = varprofile, value="Double Gaussian")
double_gaussian.pack()
varprofile.set("Single Gaussian")


Label(f2, bg = "lightblue", fg ="black", font = ("calibiri", 10, "bold"), text="Gaussian Mean").pack(side = LEFT, padx=10)
gauss_avg = Scale(f2, bg ='lightblue', fg='black',from_=-25, to=25, orient=HORIZONTAL)
gauss_avg.set(15)
gauss_avg.pack(side = LEFT, padx=10)
Label(f2, bg = "lightblue", fg ="black", font = ("calibiri", 10, "bold"), text="Gaussian Sigma").pack(side = LEFT, padx=10)
gauss_sigma = Scale(f2, bg ='lightblue', fg='black',from_=0.1, to=5.0, resolution= 0.5,orient=HORIZONTAL)
gauss_sigma.pack(side = LEFT, padx=10)
Label(f2, bg = "lightblue", fg ="black", font = ("calibiri", 10, "bold"), text="Gaussian Mom").pack(side = LEFT, padx=10)
gauss_mom = Scale(f2, bg ='lightblue', fg='black',from_=-10, to=10, orient=HORIZONTAL)
gauss_mom.set(-8)
gauss_mom.pack(side = LEFT, padx=10)

Label(f2, bg = "lightblue", fg ="black", font = ("calibiri", 10, "bold"), text="Weight 1").pack(side = LEFT, padx=10)
weight_1 = Scale(f2, bg ='lightblue', fg='black',from_=1, to=5, orient=HORIZONTAL)
weight_1.pack(side = LEFT, padx=10)
Label(f2, bg = "lightblue", fg ="black", font = ("calibiri", 10, "bold"), text="Weight 2").pack(side = LEFT, padx=10)
weight_2 = Scale(f2, bg ='lightblue', fg='black',from_=1, to=5, orient=HORIZONTAL)
weight_2.pack(side = LEFT, padx=10)


b_initial = Button(f2, fg = 'black', bg='red',text = "Show Initial Position and Momentum Profiles", font = ("calibiri", 12, "bold"), command = show_init_profile)
b_initial.pack(side=LEFT, padx=20)
#********* **************************************

#************ this frame handles dynamics/time evolution ************

f3 = Frame(asym, bg = "lightblue", borderwidth = 10, relief = RAISED)
f3.pack(fill = X)

Label(f3, bg = "lightblue", fg = "black", font = ("calibiri", 15, "bold"),  text = "Dynamics").pack()

b_t = Button(f3, fg='black',bg='red', text="Show Time Evolution and Check Theorem", font = ("calibiri", 12, "bold"), command = show_time_evolution)
b_t.pack()

#************ ***********************************


#********** this frame holds the graph **************
f4 = Frame(asym, bg = "lightblue", borderwidth = 10, relief = RAISED)
f4.pack(fill = X)
fig = matplotlib.figure.Figure(edgecolor='black', facecolor='lightblue',linewidth=7)
fig.set_size_inches(13,4.5)
ax = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)
#hexadecimal code for colour
#ax.set_facecolor('#e1ddbf')
#ax2.set_facecolor('#e1ddbf')
ax.set_facecolor('#FF8D33')
ax2.set_facecolor('#FF8D33')
canvas = FigureCanvasTkAgg(fig, master=f4)  
canvas.get_tk_widget().pack(pady=5)

#this gets the initial profile displaying right away!
show_init_profile()


#configuring the selection
single_gaussian.config(command=show_init_profile)
double_gaussian.config(command=show_init_profile)

#configuring the scales
gauss_avg.config(command=choose_function)
gauss_sigma.config(command=choose_function)
gauss_mom.config(command=choose_function)
weight_1.config(command = weight_warning)
weight_2.config(command = weight_warning)


# Create Toolbar
toolbar = NavigationToolbar2Tk(canvas, f4, pack_toolbar=False)
toolbar.update()
toolbar.pack(fill=X)


help = Button(f4, fg='black', bg='red', text="Help", font = ("calibiri", 12, "bold"), command = show_help)
help.pack()
#************************************************************

#this keeps the gui running
asym.mainloop()

