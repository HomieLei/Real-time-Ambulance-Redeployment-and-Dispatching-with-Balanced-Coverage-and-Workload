###############################################################

# Contains Clock variable and classes for Activity, 
#   CTStat (continuous-time statistic), DTStat (discrete-time
#   statistic), Entity, EventNotice, EventCalendar, FIFOQueue,
#   and Resource objects.

###############################################################

import math
import numpy as np

# Keeps track of simulation clock time
Clock = 0
    
class CTStat:
    '''
    Class of objects for continuous-time statistics

    Class attributes:
        InstanceList: list of CTStat objects instantiated
            in simulation model

    Instance attributes:
        Area: float
        Tlast: float, clock time at last call of Record (last update)
        TClear: float, clock time at last call of Clear
        Xlast: float, state value at last call of Record

    Instance attributes:
        Record
        Mean
        Clear
    '''

    InstanceList = []

    def __init__(self):
        '''
        Initializes variables when a CTStat instance is created
        '''

        self.Area = 0.0
        self.Tlast = 0.0
        self.TClear = 0.0
        self.Xlast = 0.0
        self.Max = -math.inf
        self.Min = math.inf

        # Append self to class attribute InstanceList
        self.__class__.InstanceList.append(self)
        
    def Record(self,X):
        '''
        Updates Area, Tlast, and Xlast 
        Updates observed Max and Min
        Note that this method should be called after the variable value changes

        Input:
            X: float, new value of variable monitored for CTStat instance
        '''

        self.Area += self.Xlast * (Clock - self.Tlast)
        self.Tlast = Clock
        self.Xlast = X

        if X > self.Max:
            self.Max = X

        if X < self.Min:
            self.Min = X

    def Mean(self):
        '''
        Returns the sample mean up through the current time but does not update any values

        Output:
            mean: float
        '''

        mean = 0.0
        if (Clock - self.TClear) > 0.0:
           mean = ((self.Area + self.Xlast * (Clock - self.Tlast)) 
            / (Clock - self.TClear))
        return mean
    
    def Clear(self):
        '''
        Resets Area to 0.0 and sets Tlast and TClear to current clock time
        '''

        self.Area = 0.0
        self.Tlast = Clock
        self.TClear = Clock

class DTStat():
    '''
    Class of objects for discrete-time statistics

    Class attributes:
        InstanceList: list of DTStat objects instantiated
            in simulation model

    Instance attributes:
        Sum: float, current sum of observations
        SumOfSquares: float, current sum of squared observations
        NumberOfObservations: integer, current number of observations

    Instance attributes:
        Record
        Mean
        StdDev
        N
        Clear
    '''

    InstanceList = []

    def __init__(self):
        '''
        Initializes variables when a DTStat instance is created
        '''

        self.Sum = 0.0
        self.SumOfSquares = 0.0
        self.NumberOfObservations = 0.0
        self.Max = -math.inf
        self.Min = math.inf

        # Append self to class attribute InstanceList
        self.__class__.InstanceList.append(self)
    
    def Record(self,X):
        '''
        Updates Sum, SumOfSquares, and NumberOfObservations
        Updates observed Max and Min

        Input:
            X: float, newest observation value to incorporate in Sum
                and SumOfSquares
        '''

        self.Sum += + X
        self.SumOfSquares += X * X
        self.NumberOfObservations += 1

        if X > self.Max:
            self.Max = X

        if X < self.Min:
            self.Min = X
        
    def Mean(self):
        '''
        Returns the sample mean of the DTStat based on 
            observations collected thus far

        Output:
            mean: float
        '''

        mean = 0.0
        if self.NumberOfObservations > 0.0:
            mean = self.Sum / self.NumberOfObservations
        return mean

    def StdDev(self):
        '''
        Returns the sample standard deviation of the DTStat based on
            observations collected thus far

        Output:
            stddev: float, nonnegative
        '''

        stddev = 0.0
        if self.NumberOfObservations > 1.0:
            stddev = math.sqrt((self.SumOfSquares - self.Sum**2 
                / self.NumberOfObservations) / (self.NumberOfObservations - 1))
        return stddev
            
    def N(self):
        '''
        Returns NumberOfObservations

        Output
            integer, nonnegative
        '''
        
        return self.NumberOfObservations
    
    def Clear(self):
        '''
        Resets Sum, SumOfSquares, and NumberOfObservations to 0.0
        '''
        
        self.Sum = 0.0
        self.SumOfSquares = 0.0
        self.NumberOfObservations = 0.0

class Entity():
    '''
    Class of objects for modeling generic simulation entities

    Instance attributes:
        CreateTime: float, value of Clock at creation time
    '''

    def __init__(self,call_location, service_time, tp, hospital, closest_vehicle_num, closest_vehicle_location):
        '''
        Assigns a new instance the current Clock time at creation time
        Add additional problem-specific attributes here
        '''

        self.CreateTime = Clock
        self.CallLocation = call_location
        self.ServiceTime = service_time
        self.PatientType = tp
        self.HospitalLocation = hospital
        self.ClosestVehicleNum = closest_vehicle_num
        self.ClosestVehicleLocation = closest_vehicle_location

class EventNotice():
    '''
    Class of objects for modeling event notices

    Instance attributes:
        EventTime: float
        EventType: string
        WhichObject: Entity object
    '''

    def __init__(self):
        '''
        Initializes EventTime, EventType, and WhichObject attributes
        Add additional problem-specific attributes here
        '''

        self.EventTime = 0.0
        self.EventType = ""
        self.WhichObject = None
        
        
class EventCalendar:
    '''
    Class of objects for modeling event calendars, which are
        lists of event notices ordered by time
    Based on a blueprint created by Steve Roberts

    Note: event insertion into the event calendar is
        implemented naively for the sake of readibility 
        and teachability, and is not optimized using
        efficient sorting/searching algorithms

    Instance attributes:
        ThisCalendar: list of events ordered by their occurence time

    Instance methods:
        Schedule
        Remove
        N

    '''

    def __init__(self):
        '''
        Initializes event calendar as empty list by default
        '''

        self.ThisCalendar = []   
    
    def Schedule(self,addedEvent):
        '''
        Adds EventNotice to ThisCalendar using its EventTime
            so that events in the event calendar remain
            sorted by event time in increasing order

        Input:
            addedEvent: EventNotice object
        '''
        
        # If there are no events in the calendar, simply append
        #   the new event 
        if len(self.ThisCalendar) == 0:  
            self.ThisCalendar.append(addedEvent)
        
        # If the new event's time is after the last event
        #   on the calendar, add the new event to the end
        elif self.ThisCalendar[-1].EventTime <= addedEvent.EventTime:
            self.ThisCalendar.append(addedEvent)

        # Otherwise, go through events one at a time
        #   from the start to the end of the event calendar
        #   and insert the new event in its proper place
        else:
            for rep in range(len(self.ThisCalendar)):
                if self.ThisCalendar[rep].EventTime > addedEvent.EventTime:
                    break
            self.ThisCalendar.insert(rep,addedEvent)
    
    def Remove(self):
        '''
        Removes the next event from the event calendar and returns it

        Output:
            EventNotice object
        '''

        if len(self.ThisCalendar) > 0:
            return self.ThisCalendar.pop(0)
        
    def N(self):
        '''
        Returns current number of events on the event calendar

        Output
            integer, nonnegative
        '''

        return len(self.ThisCalendar)
    
class FIFOQueue:
    '''
    Class of objects for FIFO (first-in-first-out) Queues

    Class attributes:
        InstanceList: list of FIFOQueue objects instantiated
            in simulation model

    Instance attributes:
        WIP: CTStat object, for number in queue
            (work-in-progress) over time
        ThisQueue: list of Entity objects

    Instance methods:
        NumQueue
        Add
        Remove
        Mean
    '''

    InstanceList = []

    def __init__(self):
        '''
        Initializes FIFOQueue attributes
        '''

        self.WIP = CTStat()
        self.ThisQueue = []

        # Append self to class attribute InstanceList
        self.__class__.InstanceList.append(self)
        
    def NumQueue(self):
        '''
        Returns current number in queue

        Output:
            integer, nonnegative
        '''
        
        return len(self.ThisQueue)
        
    def Add(self,X):
        '''
        Adds an entity to the end of the queue

        Input:
            X: Entity object
        '''

        self.ThisQueue.append(X)
        numqueue = self.NumQueue()
        self.WIP.Record(float(numqueue))    
    
    def Remove(self):
        '''
        Removes and returns the first entity from the queue
            and updates queue statistics

        Output:
            remove: Entity object
        '''

        if len(self.ThisQueue) > 0:
            # remove = self.ThisQueue.pop(0)
            # self.WIP.Record(float(self.NumQueue()))
            index = np.argmax([i.PatientType for i in self.ThisQueue])
            remove = self.ThisQueue.pop(index)
            self.WIP.Record(float(self.NumQueue()))
            return remove
        
    def Mean(self):
        '''
        Returns the average number in queue up to the current time

        Output:
            float, nonnegative
        '''
        return self.WIP.Mean()

class Activity:
    '''
    Class of objects for modeling an activity 
        between nodes in a stochastic 
        activity network (SAN)

    Works with class Node

    Instance attributes:
        Destination: Node object 
        CompletionTime: float, nonnegative,
            random variable realization
    '''

    def __init__(self):
        self.Destination = None
        self.CompletionTime = 0

class Node:
    '''
    Class of objects for modeling a node 
        in a stochastic activity network (SAN)

    Works with class Activity

    Instance attributes:
        Incoming: list of Node objects
        Outgoing: list of Node objects
    '''

    def __init__(self):
        self.Incoming = []
        self.Outgoing = []
        
class Resource:
    '''
    Class of objects for resources 

    Class attributes:
        InstanceList: list of FIFOQueue objects instantiated
            in simulation model

    Instance attributes:
        CurrentNumBusy: integer, current number of busy resources
        NumberOfUnits: integer, current total number of resources
        NumBusyStat: CTStat object, for number of busy resources
            over time

    Instance methods:
        Seize
        Free
        Mean
        SetUnits
    '''

    # This is a generic Resource object that also keeps track of statistics
    # on number of busy resources

    InstanceList = []

    def __init__(self, position, status, workload, to_scene, service_on_scene, to_hospital, redeployment):
        '''
        Initializes attributes
        '''
        
        self.CurrentNumBusy = 0
        self.NumberOfUnits = 0
        self.NumBusyStat = CTStat()
        self.position = position
        self.status = status
        self.workload = workload
        self.to_scene = to_scene
        self.service_on_scene = service_on_scene
        self.to_hospital = to_hospital
        self.redeployment = redeployment
        # Append self to class attribute InstanceList
        self.__class__.InstanceList.append(self)
        
    def Seize(self, Units, vehicle_num, time_to_scene, time_to_service, time_to_hospital, time_to_redeployment):
        '''
        Attempts to seize Units of resource and then update statistics
        If Units is greater than the number of units available, 
            no units are seized and the method returns False
        Otherwise, Units are seized and the method returns True

        Input:
            Units: integer, nonnegative

        Output:
            free: Boolean
        '''

        # Computes number of non-busy units
        available = self.NumberOfUnits - self.CurrentNumBusy

        if available >= Units:
            self.CurrentNumBusy += Units
            self.NumBusyStat.Record(float(self.CurrentNumBusy))
            seize = True
        else:
            seize = False
        total_service_time = time_to_scene + time_to_service + time_to_hospital + time_to_redeployment
        self.status[vehicle_num] = 'busy'
        self.workload[vehicle_num] += total_service_time
        self.to_scene[vehicle_num] += time_to_scene
        self.service_on_scene[vehicle_num] += time_to_service
        self.to_hospital[vehicle_num] += time_to_hospital
        self.redeployment[vehicle_num] += time_to_redeployment
        return seize
        
    def Free(self, Units, vehicle_num):
        '''
        Attempts to free Units of resource and then update statistics
        If Units is greater than the number of busy units, 
            no units are freed and the method returns False
        Otherwise, Units are freed and the method returns True

        Input:
            Units: integer, nonnegative

        Output:
            free: Boolean
        '''

        if self.CurrentNumBusy >= Units:
            self.CurrentNumBusy -= Units
            self.NumBusyStat.Record(float(self.CurrentNumBusy))
            free = True
        else:
            free = False

        self.status[vehicle_num] = 'idle'
        
        return free
    
    def Mean(self):
        '''
        Returns time-average number of busy resources up to current time
        
        Output:
            float, nonnegative
        '''

        return self.NumBusyStat.Mean()
        
    def SetUnits(self, Units):
        '''
        Sets the capacity of the resource (number of identical units)

        Input:
            Units: integer, nonnegative
        '''

        self.NumberOfUnits = Units
        
    def NextShift(self):
        for vehicle_num in self.workload:
            self.workload[vehicle_num] = 0
            self.to_scene[vehicle_num] = 0
            self.service_on_scene[vehicle_num] = 0
            self.to_hospital[vehicle_num] = 0
            self.redeployment[vehicle_num] = 0