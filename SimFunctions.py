###############################################################

# Contains SimFunctionsInit, Schedule, SchedulePlus,
#   and ClearStats functions, which operate on discrete
#   event simulation objects defined in SimClasses.

###############################################################

import SimClasses

def SimFunctionsInit(calendar):
    '''
    Initializes simulation replication
    Clears states from previous replication 
    Typically called before the first replication and between replications

    Input:
        calendar: EventCalendar object
    '''
    
    # Reset simulation clock to time 0
    SimClasses.Clock = 0.0
    
    # Empty the event calendar
    calendar.ThisCalendar = []
        
    # Empty queues
    for Q in SimClasses.FIFOQueue.InstanceList:
        Q.ThisQueue = []

    # Reinitialize resources
    for Re in SimClasses.Resource.InstanceList:
        Re.CurrentNumBusy = 0.0
    
    # Clear statistics
    for CT in SimClasses.CTStat.InstanceList:
        CT.Clear()
        CT.Xlast = 0.0   
        
    for DT in SimClasses.DTStat.InstanceList:
        DT.Clear()
 
def Schedule(calendar,EventType, TimeUntilEvent):
    '''
    Creates EventNotice object with given EventType and EventTime
    Schedules event to occur at time SimClasses.Clock + TimeUntilEvent

    Input:
        calendar: EventCalendar object
        EventType: string
        TimeUntilEvent: float, nonnegative, cannot be before current clock time
    '''
    
    addedEvent = SimClasses.EventNotice()
    addedEvent.EventType = EventType
    addedEvent.EventTime = SimClasses.Clock + TimeUntilEvent
    calendar.Schedule(addedEvent)
    

def SchedulePlus(calendar,EventType, TimeUntilEvent, TheObject):
    '''
    Same functionality as function Schedule(calendar, EventType, TimeUntilEvent)
    With addition of setting TheObject as the WhichObject attribute

    Input:
        calendar: EventCalendar object
        EventType: string
        TimeUntilEvent: float, nonnegative, cannot be before current clock time
        TheObject: PythonSim class object, e.g. Entity object
    '''
    
    addedEvent = SimClasses.EventNotice()
    addedEvent.EventType = EventType
    addedEvent.EventTime = SimClasses.Clock + TimeUntilEvent
    addedEvent.WhichObject = TheObject
    calendar.Schedule(addedEvent)
    
    
def ClearStats():
    '''
    Clears all DT and CT statistics, i.e. clears
        all statistics in DTStat.InstanceList and CTStat.InstanceList
    '''

    for CT in SimClasses.CTStat.InstanceList:
        CT.Clear()
        CT.Xlast = 0.0   
        
    for DT in SimClasses.DTStat.InstanceList:
        DT.Clear()