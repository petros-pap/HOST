############################ Model Generation #################################
import gurobipy as gp
import numpy as np
import pandas as pd
from time import time


def PBOS(Din, data, verbose = False):
    
    I = data[0]
    N_t = len(I)
    T = data[1]
    D = data[2]
    
    scalars = data[3]
    Kappa = scalars[0]      #($/PM task) Cost of each PM task
    Fi = scalars[1]        #($/CM task) Cost of each CM task
    Psi = scalars[2]         #($/hour) Maintenance crew hourly cost
    Omega = scalars[3]     #($/day) Vessel daily rental cost
    Q = scalars[4]           #($/hour) Maintenance crew overtime cost
    R = scalars[5]            #(MW) Wind turbine rated power output
    W = scalars[6]             #(hours) Number of workhours with standard payment
    B = scalars[7]             #(-) Number of maintenance crews
    C = scalars[8]
    
    rle = data[4].copy(True)
    tau = data[5]
    Pi = data[6]
    Pi_daily = data[7]
    fw = data[8]
    fwL = data[9]
    a_STH = data[10]
    a_LTH = data[11]
    
    
    pm_hourly_df = pd.DataFrame(columns = I)
    cm_hourly_df = pd.DataFrame(columns = I)
    x_hourly = pd.DataFrame(columns = I)
    avail_hourly = pd.DataFrame(columns = I)
    power_hourly = pd.DataFrame(columns = I)
    
    Omega = 0.1
    
    total_profit = 0
    total_cost = 0
    v1 = [] # vessel rental for day Din+1
    vessel_rentals = 0
    rentals_utilized = 0
    total_downtime = 0
    total_prod_loss = 0
    
    
    theta = pd.Series(np.zeros(len(I)), index = I)
    theta[rle <= 30] = 1

    owf = gp.Model('Offshore Windfarm Maintenance Scheduling')
    
    opt_hor = 60
    
    t1 = time()
    
    run = -1
    
    while any(np.sum(x_hourly,0) < 1):
        
        run += 1
        
        if verbose:
            print('*********************')
            print(f'Run number = {run}')
            print(f'theta = {theta}')
            print(f'rle = {rle}')
            print('*********************')
        
        i = I[:N_t]
        t = T[(Din+run)*24:(Din+run+1)*24] # Hours in a day (STH)
        if (run + 10 < opt_hor) | (N_t >= 50):
            d = D[Din+run+1:Din+opt_hor] # Days considered in optimization (LTH)
            
        else:
            d = D[Din+run+1:Din+run+11] #to completely avoid infeasible solutions 
            #we should look at which turbines still need repair, find the one that 
            #has the most inaccessible days and use this number of days as an extension of the LTH
        #print(t)
        
        owf.update()
        
        # Continuous variables
        P = owf.addVars(t, i, lb=0, name="P") # Hourly power generation for the STH
        PL = owf.addVars(d, i, lb=0, name="PL") # Daily power generation for the LTH
        s = owf.addVar(name = "s") #Profit obtained in the STH
        l = owf.addVars(d, name = "l") #Profit obtained in day d of the LTH

        # Integer variable
        q = owf.addVar(lb = 0, vtype = gp.GRB.INTEGER, name = "q") # Overtime hours

        # Binary Variables
        m = owf.addVars(t, i, vtype = gp.GRB.BINARY, name = "m") #Preventive maintenance is scheduled at hour t, wt i in the STH
        mL = owf.addVars(d, i, vtype = gp.GRB.BINARY, name = "mL") #Preventive maintenance is scheduled at day d, wt i in the LTH
        n = owf.addVars(t, i, vtype = gp.GRB.BINARY, name = "n") #Corrective maintenance is scheduled at hour t, wt i in the STH
        nL = owf.addVars(d, i, vtype = gp.GRB.BINARY, name = "nL") #Corrective maintenance is scheduled at day d, wt i in the LTH
        y = owf.addVars(t, i, vtype = gp.GRB.BINARY, name = "y") #Turbine availability at hour t, wt i in the STH
        yL = owf.addVars(d, i, vtype = gp.GRB.BINARY, name = "yL") #Turbine availability at day d, wt i in the LTH
        v = owf.addVar(vtype = gp.GRB.BINARY, name = "v") #Vessel is rented in the STH
        vL = owf.addVars(d, vtype = gp.GRB.BINARY, name = "vL") #Vessel is rented at day d in the LTH
        x = owf.addVars(t, i, vtype = gp.GRB.BINARY, name = "x") #Turbine is under maintenance at hour t, wt i in the STH

        owf.update()
        
        availability_1st_term_STH = np.ones((24,N_t))
        availability_1st_term_STH[24*np.array(rle)/np.arange(1,25).reshape((-1,1)) <= 0] = 0.
        availability_1st_term_STH = pd.DataFrame(availability_1st_term_STH, columns=i, index=t)
        
        availability_1st_term_LTH = np.ones((len(d),N_t))
        availability_1st_term_LTH[np.array(rle)-1-np.arange(len(d)).reshape((-1,1)) <= 0] = 0.
        availability_1st_term_LTH = pd.DataFrame(availability_1st_term_LTH, columns=i, index=d)

        # Constraints
        obj_fun = s + gp.quicksum(l[day] for day in d) #objective function
        con1 = owf.addConstr(s == gp.quicksum(Pi[hour]*P[hour,wt]-Kappa*m[hour,wt]-Fi*n[hour,wt]
                                         -Psi*x[hour,wt] for hour in t for wt in i)-Omega*v-Q*q
                            , name = "STH profit")
        con2 = owf.addConstrs((l[day] == gp.quicksum(Pi_daily[day]*PL[day,wt]-Kappa*mL[day,wt]-Fi*nL[day,wt]
                                                     -Psi*tau[wt]*(mL[day,wt]+nL[day,wt]) for wt in i) 
                               -Omega*vL[day] for day in d), name = "LTH profit")
        con3 = owf.addConstrs((gp.quicksum(m[hour,wt]+n[hour,wt] for hour in t)+
                              gp.quicksum(mL[day,wt]+nL[day,wt] for day in d) >= 
                              theta[wt] for wt in i), name = "Force maintenance")
        con4 = owf.addConstrs((gp.quicksum(x[t[t.index(hour)+t_hat],wt] for t_hat in range(tau[wt]))
                              >= tau[wt]*(m[hour,wt]+n[hour,wt]) for hour in t for wt in i 
                              if t.index(hour) <= len(t)-tau[wt]), name = "Consequtive")
        con5 = owf.addConstrs((gp.quicksum(x[hour,wt] for wt in i) <= B for hour in t), 
                             name = 'Max tasks per hour')
        con6 = owf.addConstrs((m[hour,wt]<=24*rle[wt]/(t.index(hour)+1) for hour in t for wt in i),
                              name = 'STH PM RLE')
        con7 = owf.addConstrs((mL[day,wt]<=rle[wt]/(d.index(day)+2) for day in d for wt in i),
                              name = 'LTH PM RLE')
        #con8 = owf.addConstrs((n[hour,wt]<=(t.index(hour)+1)/(24*rle[wt]+0.1) for hour in t for wt in i),
        #                      name = 'STH CM RLE')
        #con9 = owf.addConstrs((nL[day,wt]<=(d.index(d[-1])+1-rle[wt])/
        #                       (d.index(d[-1])-d.index(day)+0.1) for day in d for wt in i),
        #                      name = 'LTH CM RLE')
        con10 = owf.addConstrs((y[hour,wt] <= availability_1st_term_STH[wt][hour]
                               + gp.quicksum((23-t.index(tt))*n[tt,wt] for tt in t)/
                                 (23-t.index(hour)+0.1) + 1.0-theta[wt]
                                 for hour in t for wt in i), name = 'Availability STH')
        con11 = owf.addConstrs((yL[day,wt] <= availability_1st_term_LTH[wt][day]
                                +(d.index(d[-1])-gp.quicksum(d.index(dd)*nL[dd,wt] for dd in d))/
                                (d.index(d[-1])-d.index(day)+0.1) for day in d for wt in i),
                                name = "Availability LTH")
        con12 = owf.addConstrs((y[hour,wt] <= 1 - x[hour,wt] for hour in t for wt in i),
                               name = "Unavailability from maintenance")
        con13 = owf.addConstr((v>=1/N_t*gp.quicksum(m[hour,wt]+n[hour,wt] for hour in t for wt in i))
                             , name = "STH vessel")
        con14 = owf.addConstrs((vL[day] >= 1/N_t*gp.quicksum(mL[day,wt]+nL[day,wt] for wt in i)
                             for day in d), name = "LTH vessel")
        con15 = owf.addConstr((gp.quicksum(x[hour,wt] for hour in t for wt in i)<=B*W+q),
                             name = "Overtime")
        con16 = owf.addConstrs((gp.quicksum((mL[day,wt]+nL[day,wt])*tau[wt] for wt in i) <= B*W 
                               for day in d), name = 'Max daily repairs')
        con17 = owf.addConstrs((P[hour,wt]<=R*(fw[wt][hour]+1e-4)*y[hour,wt] for hour in t for wt in i),
                              name = "STH power")
        con18 = owf.addConstrs((PL[day,wt]<=24*R*(fwL[wt][day]+1e-4)*(yL[day,wt]-mL[day,wt]*tau[wt]/24) 
                                for day in d for wt in i),name = "LTH power")
        con19 = owf.addConstrs((m[hour,wt]+n[hour,wt]<=a_STH[wt][hour] for hour in t for wt in i),
                              name = "STH access")
        #con20 = owf.addConstrs((mL[day,wt]+nL[day,wt]<=a_LTH[wt][day] for day in d for wt in i),
        #                      name = "LTH access")
        con21 = owf.addConstrs((m[hour,wt]+n[hour,wt]<=0 for wt in i for hour in t 
                                if t.index(hour) < 5 or t.index(hour) > 20-tau[wt]))
        curtail = owf.addConstrs(gp.quicksum(P[hour,wt] for wt in i) <= 
                                 gp.quicksum(fw[wt][hour] for wt in i)*R*C for hour in t)
        
        
        # Set objective
        owf.setObjective(obj_fun, gp.GRB.MAXIMIZE)

        owf.update()
        
        # Solve model
        owf.optimize()

        
        if owf.SolCount == 0:
            empty_df = pd.DataFrame(np.zeros((1,len(i))), columns = i, index = [t[0]])
            pm_hourly_df = pd.concat([pm_hourly_df,empty_df])
            cm_hourly_df = pd.concat([cm_hourly_df,empty_df])
            x_hourly = pd.concat([x_hourly,empty_df])
            avail_hourly = pd.concat([avail_hourly,empty_df])
            power_hourly = pd.concat([power_hourly,empty_df])
            break
        
        mm = pd.DataFrame(np.array(owf.getAttr('X',m).values()).reshape(-1,len(i)), columns = i, index = t)
        nn = pd.DataFrame(np.array(owf.getAttr('X',n).values()).reshape(-1,len(i)), columns = i, index = t)
        xx = pd.DataFrame(np.array(owf.getAttr('X',x).values()).reshape(-1,len(i)), columns = i, index = t)
        yy = pd.DataFrame(np.array(owf.getAttr('X',y).values()).reshape(-1,len(i)), columns = i, index = t)
        PP = pd.DataFrame(np.array(owf.getAttr('X',P).values()).reshape(-1,len(i)), columns = i, index = t)

        pm_hourly_df = pd.concat([pm_hourly_df,mm])
        cm_hourly_df = pd.concat([cm_hourly_df,nn])
        x_hourly = pd.concat([x_hourly,xx])
        avail_hourly = pd.concat([avail_hourly,yy])
        power_hourly = pd.concat([power_hourly,PP])
        
        rle -= 1
        rle[rle<0] = 0
        
        for wt in i:
            if rle[wt] == 30: theta[wt] = 1
            if (np.any(mm[wt]+nn[wt] >= 0.5)): theta[wt] = 0
    
        
        total_profit += s.X
        total_cost += np.sum(np.array(fw[(Din+run)*24:(Din+run+1)*24][i])*np.array(
            Pi[(Din+run)*24:(Din+run+1)*24]).reshape(-1,1)*R*C) - s.X
        
        
        if run == 0: 
            v1.append(round(owf.getAttr("X",vL).values()[0]))
            vessel_rentals += v.X + owf.getAttr("X",vL).values()[0]
            rentals_utilized += v.X
        elif run < opt_hor-1:
            v1.append(round(owf.getAttr("X",vL).values()[0]))
            rentals_utilized += v.X
            vessel_rentals += owf.getAttr("X",vL).values()[0] + (1-v1[run-1])*v.X # the second term is saying: if
                                                                                  # the maintenance was not scheduled in 
                                                                                  # LTH of the previous horizon, include 
                                                                                  # current vessel 
        
        total_downtime += 24*N_t - np.sum(owf.getAttr("X",y).values())
        total_prod_loss += np.sum(np.array(fw[(Din+run)*24:(Din+run+1)*24])*R*C) - np.sum(np.array(PP))
    
    opt_time = round(time()-t1,2)
    print(f'Optimization finished. Total time: {opt_time} sec')
    
    # Add the cost of the unutilized vessels:
    total_cost += vessel_rentals*2500
    
    hourly_schedule = (pm_hourly_df, cm_hourly_df, x_hourly, avail_hourly, power_hourly)
    total_PM = round(np.sum(np.array(pm_hourly_df)))
    total_CM = round(np.sum(np.array(cm_hourly_df)))
    metrics = pd.Series([total_profit, total_cost, vessel_rentals, rentals_utilized, total_downtime, 
               total_prod_loss, total_PM, total_CM, opt_time], index=['total_profit_$', 'total_cost_$', 'vessel_rentals', 
                                                                      'rentals_utilized', 'total_downtime_hours', 
                                                                      'total_prod_loss_MWh', 'total_PM', 'total_CM', 
                                                                      'opt_time_sec'])
        
    return hourly_schedule, metrics


############################ Model Generation #################################
def BESN(Din, data, verbose = False):
    
    I = data[0]
    N_t = len(I)
    T = data[1]
    D = data[2]
    
    scalars = data[3]
    Kappa = scalars[0]      #($/PM task) Cost of each PM task
    Fi = scalars[1]        #($/CM task) Cost of each CM task
    Psi = scalars[2]         #($/hour) Maintenance crew hourly cost
    Omega = scalars[3]     #($/day) Vessel daily rental cost
    Q = scalars[4]           #($/hour) Maintenance crew overtime cost
    R = scalars[5]            #(MW) Wind turbine rated power output
    W = scalars[6]             #(hours) Number of workhours with standard payment
    B = scalars[7]             #(-) Number of maintenance crews
    C = scalars[8]
    
    rle = data[4].copy(True)
    tau = data[5]
    Pi = data[6]
    Pi_daily = data[7]
    fw = data[8]
    fwL = data[9]
    a_STH = data[10]
    a_LTH = data[11]
    
    
    pm_hourly_df = pd.DataFrame(columns = I)
    cm_hourly_df = pd.DataFrame(columns = I)
    x_hourly = pd.DataFrame(columns = I)
    avail_hourly = pd.DataFrame(columns = I)
    power_hourly = pd.DataFrame(columns = I)
    
    
    total_profit = 0
    total_cost = 0
    v1 = [] # vessel rental for day Din+1
    vessel_rentals = 0
    rentals_utilized = 0
    total_downtime = 0
    total_prod_loss = 0
    
    
    theta = pd.Series(np.zeros(len(I)), index = I)
    theta[rle <= 30] = 1

    owf = gp.Model('Offshore Windfarm Maintenance Scheduling')
    
    opt_hor = 60
    
    t1 = time()
    
    run = -1
    
    while any(np.sum(x_hourly,0) < 1):
        
        run += 1
        
        if verbose:
            print('*********************')
            print(f'Run number = {run}')
            print(f'theta = {theta}')
            print(f'rle = {rle}')
            print('*********************')
        
        i = I[:N_t]
        t = T[(Din+run)*24:(Din+run+1)*24] # Hours in a day (STH)
        if (run + 10 < opt_hor) | (N_t >= 50):
            d = D[Din+run+1:Din+opt_hor] # Days considered in optimization (LTH)
            
        else:
            d = D[Din+run+1:Din+run+11] #to completely avoid infeasible solutions 
            #we should look at which turbines still need repair, find the one that 
            #has the most inaccessible days and use this number of days as an extension of the LTH
        #print(t)
        
        owf.update()
        
        # Continuous variables
        P = owf.addVars(t, i, lb=0, name="P") # Hourly power generation for the STH
        PL = owf.addVars(d, i, lb=0, name="PL") # Daily power generation for the LTH
        s = owf.addVar(name = "s") #Profit obtained in the STH
        l = owf.addVars(d, name = "l") #Profit obtained in day d of the LTH

        # Integer variable
        q = owf.addVar(lb = 0, vtype = gp.GRB.INTEGER, name = "q") # Overtime hours

        # Binary Variables
        m = owf.addVars(t, i, vtype = gp.GRB.BINARY, name = "m") #Preventive maintenance is scheduled at hour t, wt i in the STH
        mL = owf.addVars(d, i, vtype = gp.GRB.BINARY, name = "mL") #Preventive maintenance is scheduled at day d, wt i in the LTH
        n = owf.addVars(t, i, vtype = gp.GRB.BINARY, name = "n") #Corrective maintenance is scheduled at hour t, wt i in the STH
        nL = owf.addVars(d, i, vtype = gp.GRB.BINARY, name = "nL") #Corrective maintenance is scheduled at day d, wt i in the LTH
        y = owf.addVars(t, i, vtype = gp.GRB.BINARY, name = "y") #Turbine availability at hour t, wt i in the STH
        yL = owf.addVars(d, i, vtype = gp.GRB.BINARY, name = "yL") #Turbine availability at day d, wt i in the LTH
        v = owf.addVar(vtype = gp.GRB.BINARY, name = "v") #Vessel is rented in the STH
        vL = owf.addVars(d, vtype = gp.GRB.BINARY, name = "vL") #Vessel is rented at day d in the LTH
        x = owf.addVars(t, i, vtype = gp.GRB.BINARY, name = "x") #Turbine is under maintenance at hour t, wt i in the STH

        owf.update()
        
        availability_1st_term_STH = np.ones((24,N_t))
        availability_1st_term_STH[24*np.array(rle)/np.arange(1,25).reshape((-1,1)) <= 0] = 0.
        availability_1st_term_STH = pd.DataFrame(availability_1st_term_STH, columns=i, index=t)
        
        availability_1st_term_LTH = np.ones((len(d),N_t))
        availability_1st_term_LTH[np.array(rle)-1-np.arange(len(d)).reshape((-1,1)) <= 0] = 0.
        availability_1st_term_LTH = pd.DataFrame(availability_1st_term_LTH, columns=i, index=d)

        # Constraints
        obj_fun = s + gp.quicksum(l[day] for day in d) #objective function
        con1 = owf.addConstr(s == gp.quicksum(Pi[hour]*P[hour,wt]-Kappa*m[hour,wt]-Fi*n[hour,wt]
                                         -Psi*x[hour,wt] for hour in t for wt in i)-Omega*v-Q*q
                            , name = "STH profit")
        con2 = owf.addConstrs((l[day] == gp.quicksum(Pi_daily[day]*PL[day,wt]-Kappa*mL[day,wt]-Fi*nL[day,wt]
                                                     -Psi*tau[wt]*(mL[day,wt]+nL[day,wt]) for wt in i) 
                               -Omega*vL[day] for day in d), name = "LTH profit")
        con3 = owf.addConstrs((gp.quicksum(m[hour,wt]+n[hour,wt] for hour in t)+
                              gp.quicksum(mL[day,wt]+nL[day,wt] for day in d) >= 
                              theta[wt] for wt in i), name = "Force maintenance")
        con4 = owf.addConstrs((gp.quicksum(x[t[t.index(hour)+t_hat],wt] for t_hat in range(tau[wt]))
                              >= tau[wt]*(m[hour,wt]+n[hour,wt]) for hour in t for wt in i 
                              if t.index(hour) <= len(t)-tau[wt]), name = "Consequtive")
        con5 = owf.addConstrs((gp.quicksum(x[hour,wt] for wt in i) <= B for hour in t), 
                             name = 'Max tasks per hour')
        con6 = owf.addConstrs((m[hour,wt]<=24*rle[wt]/(t.index(hour)+1) for hour in t for wt in i),
                              name = 'STH PM RLE')
        con7 = owf.addConstrs((mL[day,wt]<=rle[wt]/(d.index(day)+2) for day in d for wt in i),
                              name = 'LTH PM RLE')
        #con8 = owf.addConstrs((n[hour,wt]<=(t.index(hour)+1)/(24*rle[wt]+0.1) for hour in t for wt in i),
        #                      name = 'STH CM RLE')
        #con9 = owf.addConstrs((nL[day,wt]<=(d.index(d[-1])+1-rle[wt])/
        #                       (d.index(d[-1])-d.index(day)+0.1) for day in d for wt in i),
        #                      name = 'LTH CM RLE')
        con10 = owf.addConstrs((y[hour,wt] <= availability_1st_term_STH[wt][hour]
                               + gp.quicksum((23-t.index(tt))*n[tt,wt] for tt in t)/
                                 (23-t.index(hour)+0.1) + 1.0-theta[wt]
                                 for hour in t for wt in i), name = 'Availability STH')
        con11 = owf.addConstrs((yL[day,wt] <= availability_1st_term_LTH[wt][day]
                                +(d.index(d[-1])-gp.quicksum(d.index(dd)*nL[dd,wt] for dd in d))/
                                (d.index(d[-1])-d.index(day)+0.1) for day in d for wt in i),
                                name = "Availability LTH")
        con12 = owf.addConstrs((y[hour,wt] <= 1 - x[hour,wt] for hour in t for wt in i),
                               name = "Unavailability from maintenance")
        con13 = owf.addConstr((v>=1/N_t*gp.quicksum(m[hour,wt]+n[hour,wt] for hour in t for wt in i))
                             , name = "STH vessel")
        con14 = owf.addConstrs((vL[day] >= 1/N_t*gp.quicksum(mL[day,wt]+nL[day,wt] for wt in i)
                             for day in d), name = "LTH vessel")
        con15 = owf.addConstr((gp.quicksum(x[hour,wt] for hour in t for wt in i)<=B*W+q),
                             name = "Overtime")
        con16 = owf.addConstrs((gp.quicksum((mL[day,wt]+nL[day,wt])*tau[wt] for wt in i) <= B*W 
                               for day in d), name = 'Max daily repairs')
        con17 = owf.addConstrs((P[hour,wt]<=R*(fw[wt][hour]+1e-4)*y[hour,wt] for hour in t for wt in i),
                              name = "STH power")
        con18 = owf.addConstrs((PL[day,wt]<=24*R*(fwL[wt][day]+1e-4)*(yL[day,wt]-mL[day,wt]*tau[wt]/24) 
                                for day in d for wt in i),name = "LTH power")
        con19 = owf.addConstrs((m[hour,wt]+n[hour,wt]<=a_STH[wt][hour] for hour in t for wt in i),
                              name = "STH access")
        #con20 = owf.addConstrs((mL[day,wt]+nL[day,wt]<=a_LTH[wt][day] for day in d for wt in i),
        #                      name = "LTH access")
        con21 = owf.addConstrs((m[hour,wt]+n[hour,wt]<=0 for wt in i for hour in t 
                                if t.index(hour) < 5 or t.index(hour) > 20-tau[wt]))
        curtail = owf.addConstrs(gp.quicksum(P[hour,wt] for wt in i) <= 
                                 gp.quicksum(fw[wt][hour] for wt in i)*R*C for hour in t)
        
        
        # Set objective
        owf.setObjective(obj_fun, gp.GRB.MAXIMIZE)

        owf.update()
        
        # Solve model
        owf.optimize()

        
        if owf.SolCount == 0:
            empty_df = pd.DataFrame(np.zeros((1,len(i))), columns = i, index = [t[0]])
            pm_hourly_df = pd.concat([pm_hourly_df,empty_df])
            cm_hourly_df = pd.concat([cm_hourly_df,empty_df])
            x_hourly = pd.concat([x_hourly,empty_df])
            avail_hourly = pd.concat([avail_hourly,empty_df])
            power_hourly = pd.concat([power_hourly,empty_df])
            break
        
        mm = pd.DataFrame(np.array(owf.getAttr('X',m).values()).reshape(-1,len(i)), columns = i, index = t)
        nn = pd.DataFrame(np.array(owf.getAttr('X',n).values()).reshape(-1,len(i)), columns = i, index = t)
        xx = pd.DataFrame(np.array(owf.getAttr('X',x).values()).reshape(-1,len(i)), columns = i, index = t)
        yy = pd.DataFrame(np.array(owf.getAttr('X',y).values()).reshape(-1,len(i)), columns = i, index = t)
        PP = pd.DataFrame(np.array(owf.getAttr('X',P).values()).reshape(-1,len(i)), columns = i, index = t)

        pm_hourly_df = pd.concat([pm_hourly_df,mm])
        cm_hourly_df = pd.concat([cm_hourly_df,nn])
        x_hourly = pd.concat([x_hourly,xx])
        avail_hourly = pd.concat([avail_hourly,yy])
        power_hourly = pd.concat([power_hourly,PP])
        
        rle -= 1
        rle[rle<0] = 0
        
        for wt in i:
            if rle[wt] == 30: theta[wt] = 1
            if (np.any(mm[wt]+nn[wt] >= 0.5)): theta[wt] = 0
    
        
        total_profit += s.X
        total_cost += np.sum(np.array(fw[(Din+run)*24:(Din+run+1)*24][i])*np.array(
            Pi[(Din+run)*24:(Din+run+1)*24]).reshape(-1,1)*R*C) - s.X
        
        
        if run == 0: 
            v1.append(round(owf.getAttr("X",vL).values()[0]))
            vessel_rentals += v.X + owf.getAttr("X",vL).values()[0]
            rentals_utilized += v.X
        elif run < opt_hor-1:
            v1.append(round(owf.getAttr("X",vL).values()[0]))
            rentals_utilized += v.X
            vessel_rentals += owf.getAttr("X",vL).values()[0] + (1-v1[run-1])*v.X # the second term is saying: if
                                                                                  # the maintenance was not scheduled in 
                                                                                  # LTH of the previous horizon, include 
                                                                                  # current vessel 
        
        total_downtime += 24*N_t - np.sum(owf.getAttr("X",y).values())
        total_prod_loss += np.sum(np.array(fw[(Din+run)*24:(Din+run+1)*24])*R*C) - np.sum(np.array(PP))
    
    opt_time = round(time()-t1,2)
    print(f'Optimization finished. Total time: {opt_time} sec')
    
    # Add the cost of the unutilized vessels:
    total_cost += (vessel_rentals-rentals_utilized)*Omega
    
    hourly_schedule = (pm_hourly_df, cm_hourly_df, x_hourly, avail_hourly, power_hourly)
    total_PM = round(np.sum(np.array(pm_hourly_df)))
    total_CM = round(np.sum(np.array(cm_hourly_df)))
    metrics = pd.Series([total_profit, total_cost, vessel_rentals, rentals_utilized, total_downtime, 
               total_prod_loss, total_PM, total_CM, opt_time], index=['total_profit_$', 'total_cost_$', 'vessel_rentals', 
                                                                      'rentals_utilized', 'total_downtime_hours', 
                                                                      'total_prod_loss_MWh', 'total_PM', 'total_CM', 
                                                                      'opt_time_sec'])
        
    return hourly_schedule, metrics




############################ Model Generation #################################
def TBS(Din, data, verbose = False):
    
    I = data[0]
    N_t = len(I)
    T = data[1]
    D = data[2]
    
    scalars = data[3]
    Kappa = scalars[0]      #($/PM task) Cost of each PM task
    Fi = scalars[1]        #($/CM task) Cost of each CM task
    Psi = scalars[2]         #($/hour) Maintenance crew hourly cost
    Omega = scalars[3]     #($/day) Vessel daily rental cost
    Q = scalars[4]           #($/hour) Maintenance crew overtime cost
    R = scalars[5]            #(MW) Wind turbine rated power output
    W = scalars[6]             #(hours) Number of workhours with standard payment
    B = scalars[7]             #(-) Number of maintenance crews
    C = scalars[8]
    
    rle = data[4].copy(True)
    tau = data[5]
    Pi = data[6]
    Pi_daily = data[7]
    fw = data[8]
    fwL = data[9]
    a_STH = data[10]
    a_LTH = data[11]
    
    
    pm_hourly_df = pd.DataFrame(columns = I)
    cm_hourly_df = pd.DataFrame(columns = I)
    x_hourly = pd.DataFrame(columns = I)
    avail_hourly = pd.DataFrame(columns = I)
    power_hourly = pd.DataFrame(columns = I)
    
    
    total_profit = 0
    total_cost = 0
    v1 = [] # vessel rental for day Din+1
    vessel_rentals = 0
    rentals_utilized = 0
    total_downtime = 0
    total_prod_loss = 0
    
    
    theta = pd.Series(np.zeros(len(I)), index = I)
    theta[rle <= 30] = 1

    owf = gp.Model('Offshore Windfarm Maintenance Scheduling')
    
    opt_hor = 60
    
    t1 = time()
    
    run = -1
    
    while any(np.sum(x_hourly,0) < 1):
        
        run += 1
        
        if verbose:
            print('*********************')
            print(f'Run number = {run}')
            print(f'theta = {theta}')
            print(f'rle = {rle}')
            print('*********************')
        
        i = I[:N_t]
        t = T[(Din+run)*24:(Din+run+1)*24] # Hours in a day (STH)
        if (run + 10 < opt_hor) | (N_t >= 50):
            d = D[Din+run+1:Din+opt_hor] # Days considered in optimization (LTH)
            
        else:
            d = D[Din+run+1:Din+run+11] #to completely avoid infeasible solutions 
            #we should look at which turbines still need repair, find the one that 
            #has the most inaccessible days and use this number of days as an extension of the LTH
        #print(t)
        
        owf.update()
        
        # Continuous variables
        P = owf.addVars(t, i, lb=0, name="P") # Hourly power generation for the STH
        PL = owf.addVars(d, i, lb=0, name="PL") # Daily power generation for the LTH
        s = owf.addVar(name = "s") #Profit obtained in the STH
        l = owf.addVars(d, name = "l") #Profit obtained in day d of the LTH

        # Integer variable
        q = owf.addVar(lb = 0, vtype = gp.GRB.INTEGER, name = "q") # Overtime hours

        # Binary Variables
        m = owf.addVars(t, i, vtype = gp.GRB.BINARY, name = "m") #Preventive maintenance is scheduled at hour t, wt i in the STH
        mL = owf.addVars(d, i, vtype = gp.GRB.BINARY, name = "mL") #Preventive maintenance is scheduled at day d, wt i in the LTH
        n = owf.addVars(t, i, vtype = gp.GRB.BINARY, name = "n") #Corrective maintenance is scheduled at hour t, wt i in the STH
        nL = owf.addVars(d, i, vtype = gp.GRB.BINARY, name = "nL") #Corrective maintenance is scheduled at day d, wt i in the LTH
        y = owf.addVars(t, i, vtype = gp.GRB.BINARY, name = "y") #Turbine availability at hour t, wt i in the STH
        yL = owf.addVars(d, i, vtype = gp.GRB.BINARY, name = "yL") #Turbine availability at day d, wt i in the LTH
        v = owf.addVar(vtype = gp.GRB.BINARY, name = "v") #Vessel is rented in the STH
        vL = owf.addVars(d, vtype = gp.GRB.BINARY, name = "vL") #Vessel is rented at day d in the LTH
        x = owf.addVars(t, i, vtype = gp.GRB.BINARY, name = "x") #Turbine is under maintenance at hour t, wt i in the STH
        
        # Additional penalty variable and cost coefficient for TBS
        penalty = owf.addVars(i, lb=0)
        penalty_cost = 1000

        owf.update()
        
        availability_1st_term_STH = np.ones((24,N_t))
        availability_1st_term_STH[24*np.array(rle)/np.arange(1,25).reshape((-1,1)) <= 0] = 0.
        availability_1st_term_STH = pd.DataFrame(availability_1st_term_STH, columns=i, index=t)
        
        availability_1st_term_LTH = np.ones((len(d),N_t))
        availability_1st_term_LTH[np.array(rle)-1-np.arange(len(d)).reshape((-1,1)) <= 0] = 0.
        availability_1st_term_LTH = pd.DataFrame(availability_1st_term_LTH, columns=i, index=d)

        # Constraints
        obj_fun = s + gp.quicksum(l[day] for day in d) #objective function
        con1 = owf.addConstr(s == gp.quicksum(Pi[hour]*P[hour,wt]-Kappa*m[hour,wt]-Fi*n[hour,wt]
                                         -Psi*x[hour,wt] for hour in t for wt in i)-Omega*v-Q*q
                            , name = "STH profit")
        # The penalty term is added in the LTH profit formulation
        con2 = owf.addConstrs((l[day] == gp.quicksum(Pi_daily[day]*PL[day,wt]-Kappa*mL[day,wt]-Fi*nL[day,wt]
                                                     -Psi*tau[wt]*(mL[day,wt]+nL[day,wt])-penalty_cost*penalty[wt] 
                                           for wt in i)-Omega*vL[day] for day in d), name = "LTH profit")
        con3 = owf.addConstrs((gp.quicksum(m[hour,wt]+n[hour,wt] for hour in t)+
                              gp.quicksum(mL[day,wt]+nL[day,wt] for day in d) >= 
                              theta[wt] for wt in i), name = "Force maintenance")
        con4 = owf.addConstrs((gp.quicksum(x[t[t.index(hour)+t_hat],wt] for t_hat in range(tau[wt]))
                              >= tau[wt]*(m[hour,wt]+n[hour,wt]) for hour in t for wt in i 
                              if t.index(hour) <= len(t)-tau[wt]), name = "Consequtive")
        con5 = owf.addConstrs((gp.quicksum(x[hour,wt] for wt in i) <= B for hour in t), 
                             name = 'Max tasks per hour')
        con6 = owf.addConstrs((m[hour,wt]<=24*rle[wt]/(t.index(hour)+1) for hour in t for wt in i),
                              name = 'STH PM RLE')
        con7 = owf.addConstrs((mL[day,wt]<=rle[wt]/(d.index(day)+2) for day in d for wt in i),
                              name = 'LTH PM RLE')
        #con8 = owf.addConstrs((n[hour,wt]<=(t.index(hour)+1)/(24*rle[wt]+0.1) for hour in t for wt in i),
        #                      name = 'STH CM RLE')
        #con9 = owf.addConstrs((nL[day,wt]<=(d.index(d[-1])+1-rle[wt])/
        #                       (d.index(d[-1])-d.index(day)+0.1) for day in d for wt in i),
        #                      name = 'LTH CM RLE')
        con10 = owf.addConstrs((y[hour,wt] <= availability_1st_term_STH[wt][hour]
                               + gp.quicksum((23-t.index(tt))*n[tt,wt] for tt in t)/
                                 (23-t.index(hour)+0.1) + 1.0-theta[wt]
                                 for hour in t for wt in i), name = 'Availability STH')
        con11 = owf.addConstrs((yL[day,wt] <= availability_1st_term_LTH[wt][day]
                                +(d.index(d[-1])-gp.quicksum(d.index(dd)*nL[dd,wt] for dd in d))/
                                (d.index(d[-1])-d.index(day)+0.1) for day in d for wt in i),
                                name = "Availability LTH")
        con12 = owf.addConstrs((y[hour,wt] <= 1 - x[hour,wt] for hour in t for wt in i),
                               name = "Unavailability from maintenance")
        con13 = owf.addConstr((v>=1/N_t*gp.quicksum(m[hour,wt]+n[hour,wt] for hour in t for wt in i))
                             , name = "STH vessel")
        con14 = owf.addConstrs((vL[day] >= 1/N_t*gp.quicksum(mL[day,wt]+nL[day,wt] for wt in i)
                             for day in d), name = "LTH vessel")
        con15 = owf.addConstr((gp.quicksum(x[hour,wt] for hour in t for wt in i)<=B*W+q),
                             name = "Overtime")
        con16 = owf.addConstrs((gp.quicksum((mL[day,wt]+nL[day,wt])*tau[wt] for wt in i) <= B*W 
                               for day in d), name = 'Max daily repairs')
        con17 = owf.addConstrs((P[hour,wt]<=R*(fw[wt][hour]+1e-4)*y[hour,wt] for hour in t for wt in i),
                              name = "STH power")
        con18 = owf.addConstrs((PL[day,wt]<=24*R*(fwL[wt][day]+1e-4)*(yL[day,wt]-mL[day,wt]*tau[wt]/24) 
                                for day in d for wt in i),name = "LTH power")
        con19 = owf.addConstrs((m[hour,wt]+n[hour,wt]<=a_STH[wt][hour] for hour in t for wt in i),
                              name = "STH access")
        con20 = owf.addConstrs((mL[day,wt]+nL[day,wt]<=a_LTH[wt][day] for day in d for wt in i),
                              name = "LTH access")
        con21 = owf.addConstrs((m[hour,wt]+n[hour,wt]<=0 for wt in i for hour in t 
                                if t.index(hour) < 5 or t.index(hour) > 20-tau[wt]))
        curtail = owf.addConstrs(gp.quicksum(P[hour,wt] for wt in i) <= 
                                 gp.quicksum(fw[wt][hour] for wt in i)*R*C for hour in t)
        # Constraint to penalize deviation from rle of each turbine
        penalization = owf.addConstrs(gp.quicksum((rle[wt]-d.index(day))*mL[day,wt] for day in d 
                                                  if rle[wt]-d.index(day) > 0)+
                                      gp.quicksum((rle[wt])*m[hour,wt]/24 for hour in t) 
                                      <= penalty[wt] for wt in i)
        
        
        # Set objective
        owf.setObjective(obj_fun, gp.GRB.MAXIMIZE)

        owf.update()
        
        # Solve model
        owf.optimize()

        
        if owf.SolCount == 0:
            empty_df = pd.DataFrame(np.zeros((1,len(i))), columns = i, index = [t[0]])
            pm_hourly_df = pd.concat([pm_hourly_df,empty_df])
            cm_hourly_df = pd.concat([cm_hourly_df,empty_df])
            x_hourly = pd.concat([x_hourly,empty_df])
            avail_hourly = pd.concat([avail_hourly,empty_df])
            power_hourly = pd.concat([power_hourly,empty_df])
            break
        
        mm = pd.DataFrame(np.array(owf.getAttr('X',m).values()).reshape(-1,len(i)), columns = i, index = t)
        nn = pd.DataFrame(np.array(owf.getAttr('X',n).values()).reshape(-1,len(i)), columns = i, index = t)
        xx = pd.DataFrame(np.array(owf.getAttr('X',x).values()).reshape(-1,len(i)), columns = i, index = t)
        yy = pd.DataFrame(np.array(owf.getAttr('X',y).values()).reshape(-1,len(i)), columns = i, index = t)
        PP = pd.DataFrame(np.array(owf.getAttr('X',P).values()).reshape(-1,len(i)), columns = i, index = t)

        pm_hourly_df = pd.concat([pm_hourly_df,mm])
        cm_hourly_df = pd.concat([cm_hourly_df,nn])
        x_hourly = pd.concat([x_hourly,xx])
        avail_hourly = pd.concat([avail_hourly,yy])
        power_hourly = pd.concat([power_hourly,PP])
        
        rle -= 1
        rle[rle<0] = 0
        
        for wt in i:
            if rle[wt] == 30: theta[wt] = 1
            if (np.any(mm[wt]+nn[wt] >= 0.5)): theta[wt] = 0
    
        
        total_profit += s.X
        total_cost += np.sum(np.array(fw[(Din+run)*24:(Din+run+1)*24][i])*np.array(
            Pi[(Din+run)*24:(Din+run+1)*24]).reshape(-1,1)*R*C) - s.X
        
        
        if run == 0: 
            v1.append(round(owf.getAttr("X",vL).values()[0]))
            vessel_rentals += v.X + owf.getAttr("X",vL).values()[0]
            rentals_utilized += v.X
        elif run < opt_hor-1:
            v1.append(round(owf.getAttr("X",vL).values()[0]))
            rentals_utilized += v.X
            vessel_rentals += owf.getAttr("X",vL).values()[0] + (1-v1[run-1])*v.X # the second term is saying: if
                                                                                  # the maintenance was not scheduled in 
                                                                                  # LTH of the previous horizon, include 
                                                                                  # current vessel 
        
        total_downtime += 24*N_t - np.sum(owf.getAttr("X",y).values())
        total_prod_loss += np.sum(np.array(fw[(Din+run)*24:(Din+run+1)*24])*R*C) - np.sum(np.array(PP))
    
    opt_time = round(time()-t1,2)
    print(f'Optimization finished. Total time: {opt_time} sec')
    
    # Add the cost of the unutilized vessels:
    total_cost += (vessel_rentals-rentals_utilized)*Omega
    
    hourly_schedule = (pm_hourly_df, cm_hourly_df, x_hourly, avail_hourly, power_hourly)
    total_PM = round(np.sum(np.array(pm_hourly_df)))
    total_CM = round(np.sum(np.array(cm_hourly_df)))
    metrics = metrics = pd.Series([total_profit, total_cost, vessel_rentals, rentals_utilized, total_downtime, 
               total_prod_loss, total_PM, total_CM, opt_time], index=['total_profit_$', 'total_cost_$', 'vessel_rentals', 
                                                                      'rentals_utilized', 'total_downtime_hours', 
                                                                      'total_prod_loss_MWh', 'total_PM', 'total_CM', 
                                                                      'opt_time_sec'])
        
    return hourly_schedule, metrics


############################ Model Generation #################################
def CMS(Din, data, verbose = False):
    
    I = data[0]
    N_t = len(I)
    T = data[1]
    D = data[2]
    
    scalars = data[3]
    Kappa = scalars[0]      #($/PM task) Cost of each PM task
    Fi = scalars[1]        #($/CM task) Cost of each CM task
    Psi = scalars[2]         #($/hour) Maintenance crew hourly cost
    Omega = scalars[3]     #($/day) Vessel daily rental cost
    Q = scalars[4]           #($/hour) Maintenance crew overtime cost
    R = scalars[5]            #(MW) Wind turbine rated power output
    W = scalars[6]             #(hours) Number of workhours with standard payment
    B = scalars[7]             #(-) Number of maintenance crews
    C = scalars[8]
    
    rle = data[4].copy(True)
    tau = data[5]
    Pi = data[6]
    Pi_daily = data[7]
    fw = data[8]
    fwL = data[9]
    a_STH = data[10]
    a_LTH = data[11]
    
    
    pm_hourly_df = pd.DataFrame(columns = I)
    cm_hourly_df = pd.DataFrame(columns = I)
    x_hourly = pd.DataFrame(columns = I)
    avail_hourly = pd.DataFrame(columns = I)
    power_hourly = pd.DataFrame(columns = I)
    
    
    total_profit = 0
    total_cost = 0
    v1 = [] # vessel rental for day Din+1
    vessel_rentals = 0
    rentals_utilized = 0
    total_downtime = 0
    total_prod_loss = 0
    
    
    theta = pd.Series(np.zeros(len(I)), index = I)
    theta[rle <= 30] = 1

    owf = gp.Model('Offshore Windfarm Maintenance Scheduling')
    
    opt_hor = 60
    
    t1 = time()
    
    run = -1
    
    while any(np.sum(x_hourly,0) < 1):
        
        run += 1
        
        if verbose:
            print('*********************')
            print(f'Run number = {run}')
            print(f'theta = {theta}')
            print(f'rle = {rle}')
            print('*********************')
        
        i = I[:N_t]
        t = T[(Din+run)*24:(Din+run+1)*24] # Hours in a day (STH)
        if (run + 10 < opt_hor) | (N_t >= 50):
            d = D[Din+run+1:Din+opt_hor] # Days considered in optimization (LTH)
            
        else:
            d = D[Din+run+1:Din+run+11] #to completely avoid infeasible solutions 
            #we should look at which turbines still need repair, find the one that 
            #has the most inaccessible days and use this number of days as an extension of the LTH
        #print(t)
        
        owf.update()
        
        # Continuous variables
        P = owf.addVars(t, i, lb=0, name="P") # Hourly power generation for the STH
        PL = owf.addVars(d, i, lb=0, name="PL") # Daily power generation for the LTH
        s = owf.addVar(name = "s") #Profit obtained in the STH
        l = owf.addVars(d, name = "l") #Profit obtained in day d of the LTH

        # Integer variable
        q = owf.addVar(lb = 0, vtype = gp.GRB.INTEGER, name = "q") # Overtime hours

        # Binary Variables
        m = owf.addVars(t, i, vtype = gp.GRB.BINARY, name = "m") #Preventive maintenance is scheduled at hour t, wt i in the STH
        mL = owf.addVars(d, i, vtype = gp.GRB.BINARY, name = "mL") #Preventive maintenance is scheduled at day d, wt i in the LTH
        n = owf.addVars(t, i, vtype = gp.GRB.BINARY, name = "n") #Corrective maintenance is scheduled at hour t, wt i in the STH
        nL = owf.addVars(d, i, vtype = gp.GRB.BINARY, name = "nL") #Corrective maintenance is scheduled at day d, wt i in the LTH
        y = owf.addVars(t, i, vtype = gp.GRB.BINARY, name = "y") #Turbine availability at hour t, wt i in the STH
        yL = owf.addVars(d, i, vtype = gp.GRB.BINARY, name = "yL") #Turbine availability at day d, wt i in the LTH
        v = owf.addVar(vtype = gp.GRB.BINARY, name = "v") #Vessel is rented in the STH
        vL = owf.addVars(d, vtype = gp.GRB.BINARY, name = "vL") #Vessel is rented at day d in the LTH
        x = owf.addVars(t, i, vtype = gp.GRB.BINARY, name = "x") #Turbine is under maintenance at hour t, wt i in the STH

        owf.update()
        
        availability_1st_term_STH = np.ones((24,N_t))
        availability_1st_term_STH[24*np.array(rle)/np.arange(1,25).reshape((-1,1)) <= 0] = 0.
        availability_1st_term_STH = pd.DataFrame(availability_1st_term_STH, columns=i, index=t)
        
        availability_1st_term_LTH = np.ones((len(d),N_t))
        availability_1st_term_LTH[np.array(rle)-1-np.arange(len(d)).reshape((-1,1)) <= 0] = 0.
        availability_1st_term_LTH = pd.DataFrame(availability_1st_term_LTH, columns=i, index=d)

        # Constraints
        obj_fun = s + gp.quicksum(l[day] for day in d) #objective function
        con1 = owf.addConstr(s == gp.quicksum(Pi[hour]*P[hour,wt]-Kappa*m[hour,wt]-Fi*n[hour,wt]
                                         -Psi*x[hour,wt] for hour in t for wt in i)-Omega*v-Q*q
                            , name = "STH profit")
        con2 = owf.addConstrs((l[day] == gp.quicksum(Pi_daily[day]*PL[day,wt]-Kappa*mL[day,wt]-Fi*nL[day,wt]
                                                     -Psi*tau[wt]*(mL[day,wt]+nL[day,wt]) for wt in i) 
                               -Omega*vL[day] for day in d), name = "LTH profit")
        con3 = owf.addConstrs((gp.quicksum(m[hour,wt]+n[hour,wt] for hour in t)+
                              gp.quicksum(mL[day,wt]+nL[day,wt] for day in d) >= 
                              theta[wt] for wt in i), name = "Force maintenance")
        con4 = owf.addConstrs((gp.quicksum(x[t[t.index(hour)+t_hat],wt] for t_hat in range(tau[wt]))
                              >= tau[wt]*(m[hour,wt]+n[hour,wt]) for hour in t for wt in i 
                              if t.index(hour) <= len(t)-tau[wt]), name = "Consequtive")
        con5 = owf.addConstrs((gp.quicksum(x[hour,wt] for wt in i) <= B for hour in t), 
                             name = 'Max tasks per hour')
        con6 = owf.addConstrs((m[hour,wt]<=24*rle[wt]/(t.index(hour)+1) for hour in t for wt in i),
                              name = 'STH PM RLE')
        con7 = owf.addConstrs((mL[day,wt]<=rle[wt]/(d.index(day)+2) for day in d for wt in i),
                              name = 'LTH PM RLE')
        con8 = owf.addConstrs((n[hour,wt]<=(t.index(hour)+1)/(24*rle[wt]+0.1) for hour in t for wt in i),
                              name = 'STH CM RLE')
        con9 = owf.addConstrs((nL[day,wt]<=(d.index(d[-1])+1-rle[wt])/
                               (d.index(d[-1])-d.index(day)+0.1) for day in d for wt in i),
                              name = 'LTH CM RLE')
        con10 = owf.addConstrs((y[hour,wt] <= availability_1st_term_STH[wt][hour]
                               + gp.quicksum((23-t.index(tt))*n[tt,wt] for tt in t)/
                                 (23-t.index(hour)+0.1) + 1.0-theta[wt]
                                 for hour in t for wt in i), name = 'Availability STH')
        con11 = owf.addConstrs((yL[day,wt] <= availability_1st_term_LTH[wt][day]
                                +(d.index(d[-1])-gp.quicksum(d.index(dd)*nL[dd,wt] for dd in d))/
                                (d.index(d[-1])-d.index(day)+0.1) for day in d for wt in i),
                                name = "Availability LTH")
        con12 = owf.addConstrs((y[hour,wt] <= 1 - x[hour,wt] for hour in t for wt in i),
                               name = "Unavailability from maintenance")
        con13 = owf.addConstr((v>=1/N_t*gp.quicksum(m[hour,wt]+n[hour,wt] for hour in t for wt in i))
                             , name = "STH vessel")
        con14 = owf.addConstrs((vL[day] >= 1/N_t*gp.quicksum(mL[day,wt]+nL[day,wt] for wt in i)
                             for day in d), name = "LTH vessel")
        con15 = owf.addConstr((gp.quicksum(x[hour,wt] for hour in t for wt in i)<=B*W+q),
                             name = "Overtime")
        con16 = owf.addConstrs((gp.quicksum((mL[day,wt]+nL[day,wt])*tau[wt] for wt in i) <= B*W 
                               for day in d), name = 'Max daily repairs')
        con17 = owf.addConstrs((P[hour,wt]<=R*(fw[wt][hour]+1e-4)*y[hour,wt] for hour in t for wt in i),
                              name = "STH power")
        con18 = owf.addConstrs((PL[day,wt]<=24*R*(fwL[wt][day]+1e-4)*(yL[day,wt]-mL[day,wt]*tau[wt]/24) 
                                for day in d for wt in i),name = "LTH power")
        con19 = owf.addConstrs((m[hour,wt]+n[hour,wt]<=a_STH[wt][hour] for hour in t for wt in i),
                              name = "STH access")
        con20 = owf.addConstrs((mL[day,wt]+nL[day,wt]<=a_LTH[wt][day] for day in d for wt in i),
                              name = "LTH access")
        con21 = owf.addConstrs((m[hour,wt]+n[hour,wt]<=0 for wt in i for hour in t 
                                if t.index(hour) < 5 or t.index(hour) > 20-tau[wt]))
        curtail = owf.addConstrs(gp.quicksum(P[hour,wt] for wt in i) <= 
                                 gp.quicksum(fw[wt][hour] for wt in i)*R*C for hour in t)
        # Add a constraint to restrict preventive maintenance
        restrict_PM = owf.addConstr(gp.quicksum( gp.quicksum(m[hour,wt] for hour in t) + 
                                                 gp.quicksum(mL[day,wt] for day in d) 
                                               for wt in i) <= 0)
        
        
        # Set objective
        owf.setObjective(obj_fun, gp.GRB.MAXIMIZE)

        owf.update()
        
        # Solve model
        owf.optimize()

        
        if owf.SolCount == 0:
            empty_df = pd.DataFrame(np.zeros((1,len(i))), columns = i, index = [t[0]])
            pm_hourly_df = pd.concat([pm_hourly_df,empty_df])
            cm_hourly_df = pd.concat([cm_hourly_df,empty_df])
            x_hourly = pd.concat([x_hourly,empty_df])
            avail_hourly = pd.concat([avail_hourly,empty_df])
            power_hourly = pd.concat([power_hourly,empty_df])
            break
        
        mm = pd.DataFrame(np.array(owf.getAttr('X',m).values()).reshape(-1,len(i)), columns = i, index = t)
        nn = pd.DataFrame(np.array(owf.getAttr('X',n).values()).reshape(-1,len(i)), columns = i, index = t)
        xx = pd.DataFrame(np.array(owf.getAttr('X',x).values()).reshape(-1,len(i)), columns = i, index = t)
        yy = pd.DataFrame(np.array(owf.getAttr('X',y).values()).reshape(-1,len(i)), columns = i, index = t)
        PP = pd.DataFrame(np.array(owf.getAttr('X',P).values()).reshape(-1,len(i)), columns = i, index = t)

        pm_hourly_df = pd.concat([pm_hourly_df,mm])
        cm_hourly_df = pd.concat([cm_hourly_df,nn])
        x_hourly = pd.concat([x_hourly,xx])
        avail_hourly = pd.concat([avail_hourly,yy])
        power_hourly = pd.concat([power_hourly,PP])
        
        rle -= 1
        rle[rle<0] = 0
        
        for wt in i:
            if rle[wt] == 30: theta[wt] = 1
            if (np.any(mm[wt]+nn[wt] >= 0.5)): theta[wt] = 0
    
        
        total_profit += s.X
        total_cost += np.sum(np.array(fw[(Din+run)*24:(Din+run+1)*24][i])*np.array(
            Pi[(Din+run)*24:(Din+run+1)*24]).reshape(-1,1)*R*C) - s.X
        
        
        if run == 0: 
            v1.append(round(owf.getAttr("X",vL).values()[0]))
            vessel_rentals += v.X + owf.getAttr("X",vL).values()[0]
            rentals_utilized += v.X
        elif run < opt_hor-1:
            v1.append(round(owf.getAttr("X",vL).values()[0]))
            rentals_utilized += v.X
            vessel_rentals += owf.getAttr("X",vL).values()[0] + (1-v1[run-1])*v.X # the second term is saying: if
                                                                                  # the maintenance was not scheduled in 
                                                                                  # LTH of the previous horizon, include 
                                                                                  # current vessel 
        
        total_downtime += 24*N_t - np.sum(owf.getAttr("X",y).values())
        total_prod_loss += np.sum(np.array(fw[(Din+run)*24:(Din+run+1)*24])*R*C) - np.sum(np.array(PP))
    
    opt_time = round(time()-t1,2)
    print(f'Optimization finished. Total time: {opt_time} sec')
    
    # Add the cost of the unutilized vessels:
    total_cost += (vessel_rentals-rentals_utilized)*Omega
    
    hourly_schedule = (pm_hourly_df, cm_hourly_df, x_hourly, avail_hourly, power_hourly)
    total_PM = round(np.sum(np.array(pm_hourly_df)))
    total_CM = round(np.sum(np.array(cm_hourly_df)))
    metrics = pd.Series([total_profit, total_cost, vessel_rentals, rentals_utilized, total_downtime, 
               total_prod_loss, total_PM, total_CM, opt_time], index=['total_profit_$', 'total_cost_$', 'vessel_rentals', 
                                                                      'rentals_utilized', 'total_downtime_hours', 
                                                                      'total_prod_loss_MWh', 'total_PM', 'total_CM', 
                                                                      'opt_time_sec'])
        
    return hourly_schedule, metrics

