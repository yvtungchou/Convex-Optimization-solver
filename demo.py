import matplotlib.pyplot as plt
import numpy as np
import cvxpy as cvx

# in phase1, we do not care about the equality constraint
# for those constraints that are already satisfied, we use the log barrier to prevent the point from violating them
# for those constraints that are not satisfied, we use the gradient of the hyperplane to push the point into the boundry
def DrawHyperplanes(hyperplanes):
    linescale = 2
    plt.figure(1)
    numplanes = hyperplanes.shape[0]
    x0 = np.asmatrix(np.zeros((2,numplanes)))
    for i in range(0,numplanes):
        a = hyperplanes[i][:,0:2].T
        b = hyperplanes[i,2]
        x0[:,i] = (np.linalg.pinv(a)*b).T #basically the same as x0[:,i] = inv(a'*a)*a'*b
        plt.plot([x0[0,i]],[x0[1,i]],'bo')
        #plt.plot([x0[0,i], (x0[0,i]+a[0])],[x0[1,i], (x0[1,i]+a[1])],'b') #arrow without a tip
        plt.arrow(x0[0,i], x0[1,i], a[0,0], a[1,0], head_width=0.05, head_length=0.1, fc='b', ec='b') #arrow with tip
        plt.plot([x0[0,i], (x0[0,i]-linescale*a[1,0])],[x0[1,i],(x0[1,i]+linescale*a[0,0])],'r')
        plt.plot([x0[0,i], (x0[0,i]+linescale*a[1,0])],[x0[1,i],(x0[1,i]-linescale*a[0,0])],'r')
        
    plt.axis('equal')

def DrawEqualityConstraints(hyperplanes):
    linescale = 2
    numplanes = hyperplanes.shape[0]
    x0 = np.asmatrix(np.zeros((2,numplanes)))
    for i in range(0,numplanes):
        a = hyperplanes[i][:,0:2].T
        b = hyperplanes[i,2]
        x0[:,i] = (np.linalg.pinv(a)*b).T #basically the same as x0[:,i] = inv(a'*a)*a'*b
        #plt.plot([x0[0,i], (x0[0,i]+a[0])],[x0[1,i], (x0[1,i]+a[1])],'b') #arrow without a tip
        plt.plot([x0[0,i], (x0[0,i]-linescale*a[1,0])],[x0[1,i],(x0[1,i]+linescale*a[0,0])],'grey', alpha=0.5)
        plt.plot([x0[0,i], (x0[0,i]+linescale*a[1,0])],[x0[1,i],(x0[1,i]-linescale*a[0,0])],'grey', alpha=0.5)


def phase1(inequality_constraints, x, dim=2):
    #define parameters to be used in the solver
    t = 0.1 #this is the "force" with which the optimization function "pulls" on the current point
    mu = 1.5 #this is how much to scale t at each outerloop iteration
    epsilon = 0.01 #this is the desired precision of the outer loop
    # alpha = 0.01 #for back-tracking line search
    beta = 0.6 #for back-tracking line search

    #number of inequality constraints in this problem
    numplanes = inequality_constraints.shape[0]
    
    if dim == 2:
        DrawHyperplanes(inequality_constraints)
        plt.plot(x[0,0],x[1,0], 'bx')
            


    # break down the data into variables we are familiar with
    a = inequality_constraints[:][:,0:dim].T # each column is the "a" part of a hyperplane
    b = inequality_constraints[:,dim] # each row is the "b" part of a hyperplane (only one element in each row)

    # p = equality_constraints[:][:,0:2].T # each column is the "a" part of a hyperplane
    # q = equality_constraints[:,2] # each row is the "b" part of a hyperplane (only one element in each row)

    # initialze the flag list, for the ith element in the list, 0 means the ith constraint is not violated, 1 otherwise
    flag = []
    checker = True
    for coeff, offset in zip(a.T, b):
        value = coeff @ x - offset
        if value >= 0:
            checker = False
            flag.append(1)
        else:
            flag.append(0)

    if checker:
        print("Initial guess is feasible, phase1 passed.")
        return checker, x
    

    num_outer_iterations = 0
    num_inner_iterations = 0

    ###############Start outer loop (Barrier Method)#####################
    while 1:
        num_outer_iterations = num_outer_iterations + 1
        ###############Start inner loop (Newton's Method)#####################
        num_inner_iterations = 0
        while 1:
            
            # we need this dynamic flag, example:
            '''

            inequality_constraints = np.asmatrix([[0.7071,    0.7071, 1.5], 
                    
                    [0.7071,    -0.7071, 1],
                    
                    [-1, 0, -1]
                    ])

            x = np.asmatrix([0, 2]).T

            global min by wolframe:
            plt.plot(-0.2322, 0.3535, 'bx')
            '''

            flag = []
            for coeff, offset in zip(a.T, b):
                value = coeff @ x - offset
                if value >= 0:
                    flag.append(1)
                else:
                    flag.append(0)
            
            
            num_inner_iterations = num_inner_iterations + 1

            #now start computing f' (fprime), which is the sum of the optimization force 
            #and the forces pushing away from the barriers

            #compute fprime for just the optimization force first
            # fprime = t * (2 * k_cons * (p.T @ x + q).item() * p)
            fprime = 0
            
            #compute fprimeprime for just the optimization force first
            # fprimeprime = t * 2 * k_cons * p @ p.T
            fprimeprime = np.zeros((dim, dim))

            # breakpoint()

            #compute the first and second derivatives from each hyperplane and aggregate
            for j in range(0,numplanes):
                if flag[j] == 0:
                    f_for_plane_j = a[:, j].T @ x - b[j]
                    fprime_for_plane_j =  - 1 / f_for_plane_j.item() * a[:, j]
                    fprimeprime_for_plane_j = 1 / f_for_plane_j.item() ** 2 * a[:, j] @ a[:, j].T
                else:
                    fprime_for_plane_j = a[:, j]
                    fprimeprime_for_plane_j = np.zeros((dim, dim))
                fprime = fprime + fprime_for_plane_j # put in the contribution of hyperplane j to fprime
                fprimeprime = fprimeprime + fprimeprime_for_plane_j # put in the contribution of hyperplane j to fprimeprime

            
            if np.linalg.det(fprimeprime) == 0:
                step = - fprime
            else:
                step = - np.linalg.inv(fprimeprime) @ fprime
            #plt.arrow(x[0].item(), x[1].item(), step[0].item(), step[1].item(), head_width=0.05, head_length=0.1, fc='b', ec='g')
            #compute the Newton decrement squared (in terms of step and fprime)
            lambda2 = - fprime.T @ step
            #check if we've reached the Newton's method stopping condition
            #if so, break out of Newton's method
            if(lambda2 / 2 <= epsilon):
                break
    
            #now we have a direction to move the point x (i.e. the Newton step) but we don't 
            #know how far to move in that direction
            #so we look along the direction for the biggest step we can take which doesn't jump 
            #over a barrier or move to a higher-cost location
            #the method we use here is called back-tracking line search

            #back-tracking line search
            
            k = 1 #this is how much to scale the step, start with the original magnitude
            # f = t * ((p.T @ x + q).item()) ** 2

            f = 0
            for j in range(0,numplanes):
                if flag[j] == 0:
                    f = f - np.log(-a[:,j].T*x + b[j])
                else:
                    f += a[:,j].T * x - b[j]

            while 1:
                xnew = x + k*step
                fnew = 0
                pastboundary = 0
                # breakpoint()
                #check if we've jumped over a boundary
                for j in range(0,numplanes):
                    if flag[j] == 0:
                        dist = -a[:,j].T*xnew + b[j]
                        if (dist < 0):
                            pastboundary = 1
                            break
                        fnew = fnew - np.log(dist)
                    else:
                        fnew = fnew + a[:,j].T*xnew - b[j]
                
                
                #use alpha and beta to generate new guess for how much to move along the step direction
                if(pastboundary or fnew > f):  # here i deleted the fnew > f check
                # if(pastboundary):  #put in the check for terminating backtracking line search
                    #if we're not done
                    k = beta * k
                else:
                    break
            
            #now we have k, the amount to scale the step
            x = x + k*step
            if dim == 2:
                plt.plot(x[0,0],x[1,0], 'bx')
            # breakpoint()

        ###############End inner loop (Newton's Method)#####################

        # print('OUTER loop iteration %d: Number of INNER loop iterations: %d\n'%(num_outer_iterations, num_inner_iterations))
        if dim == 2:
            plt.plot(x[0,0],x[1,0], 'bx')

        #compute the duality gap (in terms of numplanes and t)
        duality_gap = numplanes / t

        #If the duality gap is below our error tolerance (epsilon), we're done!
        if duality_gap < epsilon:
            break
    
        #now that we've figured out the optimal point for this amount of optimization "force," increase the optimization force to a larger value
        #compute the new optimization force magnitude
        t = mu * t

    ###############End outer loop (Barrier Method)#####################

    print(f"\nPhase1: the computed feasible point:\n{x}\n")
    # print('Total number of outer loop iterations: %d\n'%(num_outer_iterations))

    flag = True
    for coeff, offset in zip(a.T, b):
        value = coeff @ x - offset
        if value >= 0:
            flag = False
            break
    print(f"Phase1: feasiblity check: {flag}")
    return flag, x



def phase2(c, inequality_constraints, equality_constraints, x, dim=2):
    #define parameters to be used in the solver
    t = 0.1 #this is the "force" with which the optimization function "pulls" on the current point
    mu = 1.5 #this is how much to scale t at each outerloop iteration
    epsilon = 0.01 #this is the desired precision of the outer loop
    alpha = 0.1 #for back-tracking line search
    beta = 0.6 #for back-tracking line search
    k_cons = 1000 # k_cons needs to be propotional to t, because t will get really large

    if dim == 2:
        DrawEqualityConstraints(equality_constraints)
    
    #number of inequality constraints in this problem
    numplanes = inequality_constraints.shape[0]
    num_eq = equality_constraints.shape[0]

    #let's break down the data into variables we are familiar with
    a = inequality_constraints[:][:,0:dim].T # each column is the "a" part of a hyperplane
    b = inequality_constraints[:,dim] # each row is the "b" part of a hyperplane (only one element in each row)

    p = equality_constraints[:][:,0:dim].T # each column is the "a" part of a hyperplane
    q = equality_constraints[:,dim] # each row is the "b" part of a hyperplane (only one element in each row)

    num_outer_iterations = 0
    num_inner_iterations = 0

    ###############Start outer loop (Barrier Method)#####################
    while 1:
        num_outer_iterations = num_outer_iterations + 1
        ###############Start inner loop (Newton's Method)#####################
        num_inner_iterations = 0
        while 1:
            num_inner_iterations = num_inner_iterations + 1

            #now start computing f' (fprime), which is the sum of the optimization force 
            #and the forces pushing away from the barriers


            #compute fprime for just the optimization force first
            fprime = t * c
            for j in range(0, num_eq):
                fprime += t * 2 * k_cons * (p[:, j].T @ x - q[j]).item() * p[:, j]
            #compute fprimeprime for just the optimization force first
            fprimeprime = np.zeros((dim, dim))
            for j in range(0, num_eq):
                fprimeprime += t * 2 * k_cons * p[:, j] @ p[:, j].T

            # breakpoint()

            #compute the first and second derivatives from each hyperplane and aggregate
            for j in range(0,numplanes):
                f_for_plane_j = a[:, j].T @ x - b[j]
                fprime_for_plane_j =  - 1 / f_for_plane_j.item() * a[:, j]

                fprimeprime_for_plane_j = 1 / f_for_plane_j.item() ** 2 * a[:, j] @ a[:, j].T

                fprime = fprime + fprime_for_plane_j # put in the contribution of hyperplane j to fprime
                fprimeprime = fprimeprime + fprimeprime_for_plane_j # put in the contribution of hyperplane j to fprimeprime


            #the step according to Newton's method (in terms of fprime and fprimeprime)
            if np.linalg.det(fprimeprime) == 0:
                step = - fprime
            else:
                step = - np.linalg.inv(fprimeprime) @ fprime
            # plt.arrow(x[0].item(), x[1].item(), step[0].item(), step[1].item(), head_width=0.05, head_length=0.1, fc='b', ec='g')
            # compute the Newton decrement squared (in terms of step and fprime)
            lambda2 = - fprime.T @ step

            #check if we've reached the Newton's method stopping condition
            #if so, break out of Newton's method
            if(lambda2 / 2 <= epsilon):
                break
    
            #now we have a direction to move the point x (i.e. the Newton step) but we don't 
            #know how far to move in that direction
            #so we look along the direction for the biggest step we can take which doesn't jump 
            #over a barrier or move to a higher-cost location
            #the method we use here is called back-tracking line search

            #back-tracking line search
            
            k = 1 #this is how much to scale the step, start with the original magnitude
            f = t * c.T * x
            for j in range(0, num_eq):
                f += t * k_cons * ((p[:, j].T @ x - q[j]).item()) ** 2

            for j in range(0,numplanes):
                f = f - np.log(-a[:,j].T*x + b[j])

            while 1:
                xnew = x + k*step
                fnew = t * c.T * xnew
                for j in range(0, num_eq):
                    fnew += t * k_cons * ((p[:, j].T @ xnew - q[j]).item()) ** 2
                pastboundary = 0
                #check if we've jumped over a boundary
                for j in range(0,numplanes):
                    dist = -a[:,j].T*xnew + b[j]
                    if (dist < 0):
                        pastboundary = 1
                        break
                    fnew = fnew - np.log(dist)

                #use alpha and beta to generate new guess for how much to move along the step direction
                if(pastboundary or fnew > f + alpha * k * fprime.T @ step):  #put in the check for terminating backtracking line search
                    #if we're not done
                    k = beta * k
                else:
                    break
            
            #now we have k, the amount to scale the step
            x = x + k*step
            #plot the new point for this Newton iteration in green
            if dim == 2:
                plt.plot(x[0,0],x[1,0], 'rx')

        ###############End inner loop (Newton's Method)#####################
        #plot the new point for this outer loop iteration in red
        if dim == 2:
            plt.plot(x[0,0],x[1,0], 'rx')
        # print('OUTER loop iteration %d: Number of INNER loop iterations: %d\n'%(num_outer_iterations, num_inner_iterations))


        #compute the duality gap (in terms of numplanes and t)
        duality_gap = numplanes / t

        #If the duality gap is below our error tolerance (epsilon), we're done!
        if duality_gap < epsilon:
            break
    
        #now that we've figured out the optimal point for this amount of optimization "force," increase the optimization force to a larger value
        #compute the new optimization force magnitude
        t = mu * t

    ###############End outer loop (Barrier Method)#####################

    print(f"\nPhase2: the optimal point:\n{x}\n")
    # print('Total number of outer loop iterations: %d\n'%(num_outer_iterations))

    return x

def cvx_solver(c, inequality_constraints, equality_constraints, dim):
    A = inequality_constraints[:][:,0:dim] # each column is the "a" part of a hyperplane
    b = inequality_constraints[:,dim] # each row is the "b" part of a hyperplane (only one element in each row)

    M = equality_constraints[:][:,0:dim]
    n = equality_constraints[:,dim]

    x = cvx.Variable((dim, 1))

    prob = cvx.Problem(cvx.Minimize(c.T @ x),
                [ A @ x <= b,
                M @ x == n])

    prob.solve()

    return x.value

def nice_print_problem(c, inequality_constraints, equality_constraints, x_init, dim):
    print("\n###Problem setup:\n")
    print(f"dimension of the problem: dim = {dim}")
    print(f"coefficient of the objective function: c = \n{c}")
    print(f"augmented matrix of the inequality constraints: G = \n{inequality_constraints}")
    print(f"augmented matrix of the equality constraints: A = \n{equality_constraints}")
    print(f"initial guess: x_init = \n{x_init}")

def run_example(i, c, inequality_constraints, equality_constraints, x_init, dim):
    print(f"\n###################### Example {i} #######################")
    nice_print_problem(c, inequality_constraints, equality_constraints, x_init, dim)
    print("\n###Solving:")

    flag, x_feasible = phase1(inequality_constraints, x_init, dim)

    if flag == False:
        print("Phase1 failed.")
        if dim == 2:
            plt.show()
    else:
        x_op = phase2(c, inequality_constraints, equality_constraints, x_feasible, dim)
    
    print("###Solving finished, checking result with CVX solver")
    x_cvx = cvx_solver(c, inequality_constraints, equality_constraints, dim)
    print(f"\nCVX: the optimal point:\n{x_cvx}\n")
    if dim == 2:
        plt.show()
    print(f"################### End of Example {i} ####################\n")


if __name__ == "__main__":
    print("The expected runtime for this demo should be instant.\nThere are three examples in this demo. Close the figure to proceed to the next example.\nThe third example is six dimensional, which does not have a visualization.")
    #################### Example 1 ####################
    dim = 2

    c = np.asmatrix([1, 2]).T

    inequality_constraints = np.asmatrix([[0.7071,    0.7071, 1.5], 
                    [-0.7071,    0.7071, 1.5],
                    [0.7071,    -0.7071, 1],
                    [-0.7071,    -0.7071, 1]
                    ])
    
    equality_constraints = np.asmatrix([[1, -2, 0.7]
                                        ])
    
    x_init = np.asmatrix([-2, 2]).T
    run_example(1, c, inequality_constraints, equality_constraints, x_init, dim)
    ################### end of example1 ##############

    #################### Example 2 ####################
    dim = 2
    c = np.asmatrix([1, 2]).T

    inequality_constraints = np.asmatrix([[0.7071,    0.7071, 1.5], 
                    
                    [0.7071,    -0.7071, 1],
                    
                    [-1, 0, -1]
                    ])
    
    equality_constraints = np.asmatrix([[2, 1, 2]
                                        ])
    
    x_init = np.asmatrix([0, 2]).T
    run_example(2, c, inequality_constraints, equality_constraints, x_init, dim)
    ################### end of example2 ##############


    #################### Example 3 ####################
    dim = 6
    #the optimization function c:
    c = np.asmatrix([2, 1, 3, 0, 1, -2]).T

    inequality_constraints = np.asmatrix([  [1, 0, 0, 0, 0, 0, 2], 
                                            [0, 1, 0, 0, 0, 0, 2],
                                            [0, 0, 1, 0, 0, 0, 2],
                                            [0, 0, 0, 1, 0, 0, 2],
                                            [0, 0, 0, 0, 1, 0, 2],
                                            [0, 0, 0, 0, 0, 1, 2],
                                            [-1, 0, 0, 0, 0, 0, 2], 
                                            [0, -1, 0, 0, 0, 0, 2],
                                            [0, 0, -1, 0, 0, 0, 2],
                                            [0, 0, 0, -1, 0, 0, 2],
                                            [0, 0, 0, 0, -1, 0, 2],
                                            [0, 0, 0, 0, 0, -1, 2]
                                        ])
    
    equality_constraints = np.asmatrix([[1, 1, 1, 1, 1, 1, 0],
                                        [1, -1, 1, 1, 1, 1, 0.2]
                                        ])
    
    x_init = np.asmatrix([4, 4, 4, 4, 4, 4]).T
    run_example(3, c, inequality_constraints, equality_constraints, x_init, dim)
    ################### end of example3 ##############
