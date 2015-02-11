import matplotlib.pyplot as plt
import numpy as np

# -----------------------------------------------------------------------------
# function of airfoil surface
def airfoil():
    # define x
    x1 = np.linspace(1, p, 25, endpoint=False)
    x2 = np.linspace(p, 0, 15, endpoint=False)
    x3 = np.linspace(0, p, 15, endpoint=False)
    x4 = np.linspace(p, 1, 26)
    x  = np.concatenate((x1, x2, x3, x4))

    yt = 5*t * (0.2969*x**.5 - 0.1260*x - 0.3516*x**2 + \
            0.2843*x**3 - 0.1015*x**4)

    yc  = np.zeros(xn)
    dyc = np.zeros(xn)
    for i in range(0, xn):
        if x[i] < p:
            yc[i]  = m/p**2*(2*p*x[i] - x[i]**2)
            dyc[i] = m/p**2*(2*p - 2*x[i])
        else:
            yc[i]  = m/(1-p)**2*((1-2*p) + 2*p*x[i] - x[i]**2) 
            dyc[i] = m/(1-p)**2*(2*p - 2*x[i])
    th = np.arctan(dyc)

    xa = np.zeros(xn)
    ya = np.zeros(xn)
    for i in range(0, xn/2):
        xa[i] = x[i]  - yt[i]*np.sin(th[i])
        ya[i] = yc[i] + yt[i]*np.cos(th[i])
    for i in range(xn/2, xn):
        xa[i] = x[i]  + yt[i]*np.sin(th[i])
        ya[i] = yc[i] - yt[i]*np.cos(th[i])
    return xa, ya

# -----------------------------------------------------------------------------
# initialize grid
def init_grid():
    x = np.zeros((xn,yn))
    y = np.zeros((xn,yn))

    # airfoil surface
    x[:,0] = xa
    y[:,0] = ya

    # domain boundary
    theta  = np.linspace(0, 2*np.pi, xn, endpoint=False)
    theta += .5*(2*np.pi-theta[-1])
    x[:,-1] = p + np.cos(theta)
    y[:,-1] = np.sin(theta)

    for i in range(0, xn):
        x[i,:] = np.linspace(x[i,0], x[i,-1], yn)
        y[i,:] = np.linspace(y[i,0], y[i,-1], yn)
    
    return x, y

# -----------------------------------------------------------------------------
# show grid domain
def show_grid(x, y, t):
    for j in range(0, yn):
        xlines[j].set_xdata(np.append(x[:,j], x[0,j]))
        xlines[j].set_ydata(np.append(y[:,j], y[0,j]))
    for i in range(0, xn):
        ylines[i].set_xdata(x[i,:])
        ylines[i].set_ydata(y[i,:])

    ax.set_title(method.upper() + ' (t = ' + str(t) + ')')
    fig.canvas.draw()
    return

# -----------------------------------------------------------------------------
# Thomas algorithm
def thomas(A, B, C, D):
    a = np.insert(A, 0, 0)
    b = B
    c = C.copy()
    d = D.copy()

    n = len(b)
    
    # forward
    c[0] /= b[0]
    for i in range(1, n-1):
        c[i] /= (b[i] - a[i]*c[i-1])
    d[0] /= b[0]
    for i in range(1, n):
        d[i] = (d[i] - a[i]*d[i-1]) / (b[i] - a[i]*c[i-1])

    # backward substitution
    x = np.zeros(n)
    x[n-1] = d[n-1]
    for i in range(n-2, -1, -1):
        x[i] = d[i] - c[i]*x[i+1]
    return x

# -----------------------------------------------------------------------------
# Alternating-Direction Implicit
def adi(x, y):
    for t in range(0, tmax):
        show_grid(x, y, t)

        # first half-step (xi-direction implicit)
        for j in range(1, yn-1):
            # initialization
            A  = np.zeros(xn-1)
            B  = np.zeros(xn)
            C  = np.zeros(xn-1)
            Dx = np.zeros(xn)
            Dy = np.zeros(xn)
            for i in range(0, xn):
                if (i == xn-1):
                    # duplicate the frist line in the last
                    x = np.vstack([x, x[0,:]])
                    y = np.vstack([y, y[0,:]])

                deta  = np.sqrt((x[i,j]-x[i,j-1])**2 + (y[i,j]-y[i,j-1])**2)
                dxi   = np.sqrt((x[i,j]-x[i-1,j])**2 + (y[i,j]-y[i-1,j])**2)
                xeta  = (x[i,j+1] - x[i,j-1]) / (2*deta)
                yeta  = (y[i,j+1] - y[i,j-1]) / (2*deta)
                xxi   = (x[i+1,j] - x[i-1,j]) / (2*dxi)
                yxi   = (y[i+1,j] - y[i-1,j]) / (2*dxi)
                x2    = (x[i+1,j+1]-x[i+1,j-1]-x[i-1,j+1]+x[i-1,j-1]) \
                        / (4*deta*dxi)
                y2    = (y[i+1,j+1]-y[i+1,j-1]-y[i-1,j+1]+y[i-1,j-1]) \
                        / (4*deta*dxi)
                a     = xeta**2 + yeta**2
                b     = xxi*xeta + yxi*yeta
                c     = xxi**2 + yxi**2

                if (i != 0   ): A[i-1] = a * deta**2
                if (i != xn-1): C[i]   = a * deta**2
                B[i]  = -2*(a * deta**2 + c * dxi**2)
                Dx[i] = 2*b*deta**2*dxi**2*x2 - c*dxi**2*(x[i,j+1]+x[i,j-1]) 
                Dy[i] = 2*b*deta**2*dxi**2*y2 - c*dxi**2*(y[i,j+1]+y[i,j-1]) 
                if (i == 0):
                    Dx[i] -= a * deta**2 * x[i-1,j]
                    Dy[i] -= a * deta**2 * y[i-1,j]
                if (i == xn-1):
                    Dx[i] -= a * deta**2 * x[i+1,j]
                    Dy[i] -= a * deta**2 * y[i+1,j]
                
                if (i == xn-1):
                    # delete the duplicate line
                    x = x[:-1,:]
                    y = y[:-1,:]
            # use Thomas algorithm to solve the matrix
            x[:, j] = thomas(A, B, C, Dx)
            y[:, j] = thomas(A, B, C, Dy)

        # second half-step (eta-direction implicit)
        for i in range(0, xn):
            if (i == xn-1):
                # duplicate the frist line in the last
                x = np.vstack([x, x[0,:]])
                y = np.vstack([y, y[0,:]])
            # initialization
            A  = np.zeros(yn-1)
            B  = np.zeros(yn)
            C  = np.zeros(yn-1)
            Dx = np.zeros(yn)
            Dy = np.zeros(yn)
            # j = 0 (airfoil boundary)
            B[0]  = 1
            C[0]  = 0
            Dx[0] = x[i,0]
            Dy[0] = y[i,0]
            # j = yn-1 (domain boundary)
            A[yn-2]  = 0
            B[yn-1]  = 1
            Dx[yn-1] = x[i,yn-1]
            Dy[yn-1] = y[i,yn-1]
            for j in range(1, yn-1):
                deta  = np.sqrt((x[i,j]-x[i,j-1])**2 + (y[i,j]-y[i,j-1])**2)
                dxi   = np.sqrt((x[i,j]-x[i-1,j])**2 + (y[i,j]-y[i-1,j])**2)
                xeta  = (x[i,j+1] - x[i,j-1]) / (2*deta)
                yeta  = (y[i,j+1] - y[i,j-1]) / (2*deta)
                xxi   = (x[i+1,j] - x[i-1,j]) / (2*dxi)
                yxi   = (y[i+1,j] - y[i-1,j]) / (2*dxi)
                x2    = (x[i+1,j+1]-x[i+1,j-1]-x[i-1,j+1]+x[i-1,j-1]) \
                        / (4*deta*dxi)
                y2    = (y[i+1,j+1]-y[i+1,j-1]-y[i-1,j+1]+y[i-1,j-1]) \
                        / (4*deta*dxi)
                a     = xeta**2 + yeta**2
                b     = xxi*xeta + yxi*yeta
                c     = xxi**2 + yxi**2

                A[j-1] = c * dxi**2
                C[j]   = c * dxi**2
                B[j]  = -2*(a * deta**2 + c * dxi**2)
                Dx[j] = 2*b*deta**2*dxi**2*x2 - a*deta**2*(x[i+1,j]+x[i-1,j]) 
                Dy[j] = 2*b*deta**2*dxi**2*y2 - a*deta**2*(y[i+1,j]+y[i-1,j]) 
                
            if (i == xn-1):
                # delete the duplicate line
                x = x[:-1,:]
                y = y[:-1,:]

            # use Thomas algorithm to solve the matrix
            x[i, :] = thomas(A, B, C, Dx)
            y[i, :] = thomas(A, B, C, Dy)
    return x, y


# -----------------------------------------------------------------------------
# SOR
def sor(x, y):
    for t in range(0, tmax):
        show_grid(x, y, t)
        for i in range(0, xn):
            if (i == xn-1):
                # duplicate the frist line in the last
                x = np.vstack([x, x[0,:]])
                y = np.vstack([y, y[0,:]])
            for j in range(1, yn-1):
                deta  = np.sqrt((x[i,j]-x[i,j-1])**2 + (y[i,j]-y[i,j-1])**2)
                dxi   = np.sqrt((x[i,j]-x[i-1,j])**2 + (y[i,j]-y[i-1,j])**2)
                xeta  = (x[i,j+1] - x[i,j-1]) / (2*deta)
                yeta  = (y[i,j+1] - y[i,j-1]) / (2*deta)
                xxi   = (x[i+1,j] - x[i-1,j]) / (2*dxi)
                yxi   = (y[i+1,j] - y[i-1,j]) / (2*dxi)
                x2    = (x[i+1,j+1]-x[i+1,j-1]-x[i-1,j+1]+x[i-1,j-1]) \
                        / (4*deta*dxi)
                y2    = (y[i+1,j+1]-y[i+1,j-1]-y[i-1,j+1]+y[i-1,j-1]) \
                        / (4*deta*dxi)
                a     = xeta**2 + yeta**2
                b     = xxi*xeta + yxi*yeta
                c     = xxi**2 + yxi**2

                # update
                err_x = (2*b*x2 - a*(x[i+1,j]+x[i-1,j])/dxi**2 - \
                        c*(x[i,j+1]+x[i,j-1])/deta**2) / \
                        (-2*a/dxi**2 - 2*c/deta**2) - x[i,j]
                err_y = (2*b*y2 - a*(y[i+1,j]+y[i-1,j])/dxi**2 - \
                        c*(y[i,j+1]+y[i,j-1])/deta**2) / \
                        (-2*a/dxi**2 - 2*c/deta**2) - y[i,j]
                x[i,j] += err_x * omega
                y[i,j] += err_y * omega
            if (i == xn-1):
                # delete the duplicate line
                x = x[:-1,:]
                y = y[:-1,:]
                
    return x, y

# -----------------------------------------------------------------------------
# main program

method = "adi"

# parameters
m  = 0.04
p  = 0.4
t  = 0.12
xn = 81
yn = 21

omega = 1.5 # relaxation parameter
tmax = 100

xa, ya = airfoil()
x, y = init_grid()

fig = plt.figure()
ax = fig.add_subplot(111)
ax.axis('equal')

xlines = []
ylines = []
for j in range(0, yn):
    line, = ax.plot(np.append(x[:,j],x[0,j]), \
            np.append(y[:,j], y[0,j]), color='k')
    xlines.append(line)
for i in range(0, xn):
    line, = ax.plot(x[i,:], y[i,:], color='k')
    ylines.append(line)

fig.show()

# iteration
if (method == "sor"):
    x, y = sor(x, y)
elif (method == "adi"):
    x, y = adi(x, y)
