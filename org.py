def read_zipcodes_information(filename):
    #Speed and distances
    data_speedIn = pd.read_excel(f'{filename}.xlsx', sheet_name='SpeedIn', index_col=0, dtype={'ave_speed_Kmhr': float})
    data_speedB = pd.read_excel(f'{filename}.xlsx', sheet_name='SpeedB', index_col=0)
    data_distanceB = pd.read_excel(f'{filename}.xlsx', sheet_name='DistanceB', index_col=0)
    speedIn = data_speedIn['ave_speed_Kmhr'].tolist()
    speedB = data_speedB.values.tolist() 
    distanceB = data_distanceB.values.tolist()

    #Expected demand and areas
    data_demand = pd.read_excel(f'{filename}.xlsx', sheet_name='Demand-Y1', dtype={'Zipcode': object})
    data_areas = pd.read_excel(f'{filename}.xlsx', sheet_name='Areas', dtype={'Zipcode': object})
    return speedIn, speedB, distanceB, data_demand, data_areas

def read_pending_orders(data):
    #data = pd.read_excel(f'{filename}.xlsx', sheet_name='Pending List', dtype={'Zipcode': object, 'Date': object})
    data['TW'] = data['TW'].fillna(tw)
    return data

def unstack_data(data, column):
    ptable = data.pivot_table(index='Zipcode', columns='Weekday', values=column, aggfunc=np.sum)
    ptable.columns = ['1-Monday', '2-Tuesday', '3-Wednesday', '4-Thursday', '5-Friday', '6-Weekend']
    ptable = ptable.reset_index()
    ptable = ptable[['1-Monday', '2-Tuesday', '3-Wednesday', '4-Thursday', '5-Friday', '6-Weekend']]
    ptable = ptable.fillna(0)
    ptable = ptable.values.tolist()
    ptable.insert(0,[0,0,0,0,0,0])
    return ptable

def time_inside_function(zipcode, stops):
    time_inside = kCA*math.sqrt((stops)*areas[zipcode])/speedIn[zipcode]
    return time_inside

def time_function(i,j,n):
    if i == j:
        time = 0        
    else:
        if j>0:
            time = kCA*math.sqrt((n)*areas[j])/speedIn[j] + distanceB[i][j]/speedB[i][j] + ut*n
        else:
            time = distanceB[i][j]/speedB[i][j] + ut*n
    return 1000*time

def time_function_val(j,n): 
    # Validation of capacity of a particular zipcode in order to split the demand
    if j>0:
        time = kCA*math.sqrt((n)*areas[j])/speedIn[j] + 2*distanceB[0][j]/speedB[0][j] + ut*n
    else:
        time = 0
    return time

def time_between_function(i, j):
    time_between = distanceB[i][j]/speedB[i][j]
    return time_between

def capacity_list_function(vehicles, S): 
    #vehicles is the number of routes, S is max number of stops, M is large number
    capacity_list = [S for i in range(vehicles)]
    capacity_list.append(M)
    return capacity_list

def time_list_function(vehicles, Tmax): 
    #vehicles is the number of routes, Tmax is max lenght of route, M is large number
    time_list = [1000*Tmax for i in range(vehicles)]
    time_list.append(1000*M)
    return time_list

def fuel_function(j, q):
    distance = kCA*math.sqrt((q)*areas[j])
    liters = distance*(fe+(ff-fe)*q/Q)
    return liters


def distance_function (i,j):
    distance = distanceB[i][j] + kCA*math.sqrt((total_demand[j])*areas[j])
    #distance in + distance out
    return distance

def create_data_model():
    """Stores the data for the problem."""
    data = {}
    data['distance_matrix'] = transit_c
    data['post'] = pospt_c
    data['fixed_cost'] = fc*1000
    data['demands'] = total_demand
    data['vehicle_capacities'] = capacity_list_function(routes,S)
    data['time_capacities'] = time_list_function(routes,Tmax)
    data['num_vehicles'] = routes+1
    data['depot'] = 0
    return data

def print_solution(data, manager, routing, assignment):
    sent = [[0 for k in range(routes)] for i in range(zipcodes)] #solution matrix
    routes_results, sol_results = [],[]
    total_cost, total_load, total_labor, total_fuel, total_routes, unloaded, late_postponed, total_distance = 0,0,0,0,0,0,0,0 
    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        if vehicle_id ==data['num_vehicles']-1:
            plan_output = 'Postponed zipcodes:\n'
        else:
            plan_output = 'Route for vehicle {}:\n'.format(vehicle_id)
        route_cost, route_load, route_time, route_fuel, activator, route_distance = 0,0,0,0,0,0
        while not routing.IsEnd(index):
            if activator==1: #we need this to safe the previous node  (but only after the depot, i.e. activator=1)
                i_index = node_index
            node_index = manager.IndexToNode(index)
            route_load += data['demands'][node_index]
            plan_output += ' {0} Stops({1}) -> '.format(node_index, route_load)
            previous_index = index            
            index = assignment.Value(routing.NextVar(index))
            route_cost += routing.GetArcCostForVehicle(
                previous_index, index, vehicle_id)           
            if vehicle_id ==data['num_vehicles']-1:
                if late_time[node_index]>1:
                    late_postponed += demand[node_index][0]
            else:
                total_fuel += fuel_function (node_index,data['demands'][node_index])
                route_fuel += fuel_function (node_index,data['demands'][node_index])
                sent[node_index][vehicle_id] = data['demands'][node_index] #solution
                if activator==1: #route time can be computed after first node (depot)
                    route_time += time_function (i_index,node_index, total_demand[node_index])
                    route_distance += distance_function(i_index,node_index)
            activator=1        
            #print('route distance=', route_cost), here you can prove that fixed cost is allocated to first arc in route       
        if vehicle_id <=data['num_vehicles']-2:
            plan_output += ' {0} Stops({1})\n'.format(manager.IndexToNode(index),                                                     route_load)
            plan_output += 'Cost of the route: {}$\n'.format(route_cost/1000)
            plan_output += 'Stops of the route: {}\n'.format(route_load)
            plan_output += 'Time of the route: {} hrs\n'.format(round(route_time/1000),1)
            plan_output += 'Distance of the route: {} km\n'.format(round(route_distance),1)
            unloaded += route_load
            total_labor += route_cost/1000
            if route_cost>0:
                total_routes += 1
                routes_results.append ([date_today, vehicle_id+1, route_distance, route_load, route_time, total_labor, route_fuel])
        else:
            plan_output += '\nTotal Stops postponed: {}\n'.format(route_load)
        print(plan_output)
        total_cost += route_cost
        total_load += route_load
        total_distance += route_distance                    
    print('Total orders: {0} ->  Total orders allocated: {1}({3}%), Total orders postponed: {2}({4}%)'
          .format(total_load,unloaded,total_load-unloaded,
                  round(unloaded/total_load*100,1),
                  round((total_load-unloaded)/total_load*100,1)))
    real_cost = total_labor+lc*ut*unloaded
    transp_cost = total_labor-fc*total_routes
    unloading_cost = lc*ut*unloaded
    fixed_cost = fc*total_routes
    fuel_cost = gc*total_fuel
    if real_cost>0:
        print('Labor Cost: {0}$({1}%) [Transp.: {2}$({3}%), Unl.: {4}$({5}%)]'
              .format(round(transp_cost+unloading_cost,1),round((transp_cost+unloading_cost)/real_cost*100,1),
                  round(transp_cost,1),round(transp_cost/real_cost*100,1),
                      round(unloading_cost,1),round(unloading_cost/real_cost*100,1)))
        print('Fixed cost: {0}$({1}%)'.format(fixed_cost,round(fixed_cost/real_cost*100,1)))
        print('Fuel cost: {0}$({1}%)$'.format(round(fuel_cost,1),round(fuel_cost/real_cost*100,1)))
        print('Total Cost (Labor+Fixed+Fuel): {}$'.format(round(real_cost,1)))
    else:
        print('The best solution found today to pospone all the orders. Please note that you have {0} critical orders'
              .format(sum(late_orders)))
    #arccost begin with the fixed cost, therefore it is necessary to remove it
    if sum(late_orders)>0:
        print('Critical orders: {0}, Error: {1}, Critical fill rate: {2}%,'.format(sum(late_orders),late_postponed,
                                                                            round(1-late_postponed/sum(late_orders),3)*100))
    else:
        print('Critical orders: {0}, Error: {1}, Critical fill rate: {2}%,'.format(sum(late_orders),late_postponed,
                                                                            0))
    print ('Total distance: {}km'.format(round(total_distance,1)))    
    print('Model cost: {}$'.format(round(total_cost/1000,1))) 
    #arccost begin with the fixed cost, therefore it is necessary to remove it    
    sol_results=[date_today, fuel_cost, transp_cost+unloading_cost, fixed_cost, total_cost/1000+unloading_cost, 
                        total_routes,unloaded, route_load,sum(total_demand),late_postponed, total_distance]
    return sent, sol_results, routes_results

def consolidation_heuristics(to_print = False):
    """Solve the CVRP problem."""
    # Instantiate the data problem.
    data = create_data_model()
    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']),
                                           data['num_vehicles'], data['depot'])
    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)
    # Create and register a transit callback.
    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['distance_matrix'][from_node][to_node]
    def pending_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['post'][to_node]
    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    print('transit_callback_index')
    print(transit_callback_index)
    pending_callback_index = routing.RegisterTransitCallback(pending_callback)
    # Define cost of each arc.
    for i in range(data['num_vehicles']-1):
        routing.SetArcCostEvaluatorOfVehicle(transit_callback_index, i) #Transit cost
        routing.SetFixedCostOfVehicle(data['fixed_cost'], i) #Fixed cost
    routing.SetArcCostEvaluatorOfVehicle(pending_callback_index, data['num_vehicles']-1) #Postponement and/or NonService cost
     # Add Capacity constraint.
    def demand_callback(from_index): #
        """Returns the demand of the node."""
        # Convert from routing variable Index to demands NodeIndex.
        from_node = manager.IndexToNode(from_index)        
        return data['demands'][from_node]
    demand_callback_index = routing.RegisterUnaryTransitCallback(
        demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,  # null capacity slack
        data['vehicle_capacities'],  # vehicle maximum capacities
        True,  # start cumul to zero
        'Capacity')
        
        
    # Add time constraint.
    def time_callback(from_index,to_index): #
        """Returns the demand of the node."""
        # Convert from routing variable Index to NodeIndex in time
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return time_matrix[from_node][to_node]    
    time_callback_index = routing.RegisterTransitCallback(time_callback) 
    routing.AddDimensionWithVehicleCapacity(
        time_callback_index,
        0,  # null capacity slack
        data['time_capacities'],  # vehicle maximum capacities
        True,  # start cumul to zero
        'Time')
    # Setting solution heuristic-procedure.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    search_parameters.time_limit.seconds = 5 #10 # 60 #20 #3000
    search_parameters.log_search = True
    # Solve the problem.
    assignment = routing.SolveWithParameters(search_parameters)
    # Print solution on console.
    if assignment:
        sent, sol_results, routes_results = print_solution(data, manager, routing, assignment)   
    return sent, sol_results, routes_results, data

def data_export(data, save= False):  #save = True or False   <--- valor modificlabe
    # Data export
    data_new2 = pending_list.copy()
    data_sol = [] #New list of orders sent
    for k in range(routes):
        for i in range(zipcodes):
            if data[i][k]>0:
                for a in range(int(data[i][k])):
                    #if i == 75: <-- Por alguna razón le botaba a José Enrique 
                        #print(i,k)
                        #print(data_new2[data_new2.Zipcode == zipnames[i]].shape)
                    sol = data_new2[data_new2.Zipcode == zipnames[i]].iloc[0]
                    data_new2 = data_new2.drop(data_new2[data_new2.Zipcode == zipnames[i]].index[0]) #updating pending list
                    data_sol.append([sol[0], sol[1], sol[2], sol[3], k+1, date_today])

    #Excel outputs
    #if save == True:
       # today = date.today()
        #data_new2.to_excel(f'Output2-NewPendingList-{today}-Operation.xlsx', index=False)
        #sent_DF.to_excel(f'Output1-SentList-{today}-Operation.xlsx', index=False)

    #List to data-frame
    sent_DF = pd.DataFrame(data_sol, columns=['ID','Date','Zipcode','Weight','Route','Date_Routed'])
    
    return  sent_DF, data_new2





# Data preprocessing
start_time = time.time()
#Convert dataframes to lists: zipcodes, areas, covering matrix, average and deviation
zipcodes = len(data_areas) #number of zipcodes
zipnames = data_areas['Zipcode'].tolist() #names
areas = data_areas['Area km2'].tolist() #areas

##This model will define the deliveries for next Working day
date_today = max(pending_list['Date']) + timedelta(days=1) #MAX+1 of every date
print(date_today)
print('NUM_DAY', datetime.weekday(date_today))
if datetime.weekday(date_today) == 6: #deliveries are not available on sundays
    date_today = date_today + timedelta(days=1) 

#Average matrix per zipcode per weekday
mu = unstack_data(data_demand, 'Average clients')

#computation of unitary postponement cost
expected = [(t+1)*tw_p[t]*sum(mu[i])/len(mu[i])+1 for i in range(zipcodes) for t in range(len(tw_p))]
penalty_ca_d = [kCA*math.sqrt((expected[i])*areas[i]) for i in range(zipcodes)]
penalty_d = [(distanceB[0][i]+penalty_ca_d[i])/(expected[i]) for i in range(zipcodes)]
penalty_c = [0 for i in range(zipcodes)]
for i in range(zipcodes): ##Pending max stops per zipcode
    if speedIn[i]>0:
        #penalty_c[i] = lc*(penalty_d[i]/speedIn[i])+fc/expected[i]+gc*(penalty_d[i]*ff*Q/(tw*expected[i]))
        penalty_c[i] = lc*(penalty_d[i]/speedIn[i])+ fc/expected[i]+ gc*penalty_d[i]/expected[i]*(fe+(ff-fe)*(expected[i]/Q))


#Demand computation        
data_ag = pending_list['Date'].groupby([pending_list['Date'], pending_list['Zipcode'],pending_list['TW']]).count().reset_index(name='Stops') #aggregate per zipcode and date 
demand = [[0 for t in range(len(tw_p))] for i in range(zipcodes)] #demand definition according to weekday
late_time = [0 for i in range(zipcodes)]
late_orders = [0 for i in range(zipcodes)]
total_demand = [0 for i in range(zipcodes)]
for i in range(len(data_ag)): #evaluate every order (factura)    
    #compute total demand per zipcode
    total_demand[zipnames.index(data_ag['Zipcode'][i])] += data_ag['Stops'][i]
    #number of weekends between dates. This is critical cause there are not deliveries on sunday
    count = 0
    for d_ord in range(data_ag['Date'][i].toordinal(), date_today.toordinal()):  
        d = date.fromordinal(d_ord)
        if (d.weekday() == 6):
            count += 1    
    t = (date_today-data_ag['Date'][i]).days - count #computation of orders remaining days for delivery
    if t<1:
        demand[zipnames.index(data_ag['Zipcode'][i])][len(tw_p)-1] += data_ag['Stops'][i] #if there was a weekend 
    else:
        if t >= data_ag['TW'][i]:
            demand[zipnames.index(data_ag['Zipcode'][i])][0]+= data_ag['Stops'][i]
            if late_time[zipnames.index(data_ag['Zipcode'][i])] < t-data_ag['TW'][i]+1:
                late_time[zipnames.index(data_ag['Zipcode'][i])] = t-data_ag['TW'][i]+1 #this is 1 if it is the last due date, or it will bigger if there are more days than TW 
        else:
            demand[zipnames.index(data_ag['Zipcode'][i])][len(tw_p)-t-1] = data_ag['Stops'][i]
late_orders = [row[0] for row in demand] #number of late (critical) orders in each zipcode
print('Elapsed time:',round(time.time()-start_time,2), 'seconds')





# Cost computation
start_time = time.time()
#transportation (transit) cost includes: time between zipcodes, time inside and fuel consumption
transit_c = [[0 for i in range(zipcodes)] for j in range(zipcodes)]
time_matrix = [[0 for i in range(zipcodes)] for j in range(zipcodes)]
#post_c could be the postonement cost or the penalization cost  
pospt_c = [0 for i in range(zipcodes)]
time_inside, time_between, fuel = 0,0,0
for j in range(zipcodes): 
    if late_time[j]>0: #computation of post or error cost
        pospt_c[j] = late_orders[j]*late_time[j]*M #penalty cost depends on the number of orders and the number of days of the latest order
    else:
        pospt_c[j] = int(penalty_c[j]*total_demand[j]*1000)
    if total_demand[j] > S or time_function_val(j,total_demand[j]) > Tmax:
        stops_act = 0
        cont_stops = 1
        while stops_act == 0:
            max_stops_j = total_demand[j] - cont_stops
            if time_function_val(j,max_stops_j) > Tmax:
                cont_stops += 1
            else:
                stops_act = 1
        total_demand[j] = min(S, max_stops_j)  #We are only sending the maximum number of possible stops
    if j > 0:
        time_inside = time_inside_function(j, total_demand[j])
        fuel = fuel_function (j, total_demand[j])
    for i in range(zipcodes): #labor cost = inside time + time between zipcodes
        if i != j:
            time_between = time_between_function(i,j)                
            transit_c[i][j] = int((lc*(time_inside+time_between)+gc*fuel)*1000)
            time_matrix[i][j] = time_function(i,j, total_demand[j])
                #transit_c[i][j] = int((lc*(time_inside+time_between)+ gc*fuel + total_demand[j]*ut)*1000) 
                #the unloading cost is computed outside the objective funciton, it is for every order (even it is postponed)
print('Elapsed time:',round(time.time()-start_time,2), 'seconds')
