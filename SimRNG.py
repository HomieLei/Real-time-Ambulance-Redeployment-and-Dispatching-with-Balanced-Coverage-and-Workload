###############################################################

# This random number generator and random-variate generation 
#   functions are translations of the C programs found in 
#   Law, A. M. and Kelton, W. D., ''Simulation Modeling and 
#   Analysis'', Singapore: The McGraw-Hill Book Co, pp. 430-431.

# For very large or serious real-world applications, 
#   we recommend using tools found in the numpy python package,
#   and do not recommend using the random number generator
#   in this file, which is mainly used for illustrative
#   and educational purposes.

# This random number generator is a prime modulus 
#   multiplicative linear congruential generator (PMMLCG)
#   based on Marse and Robert's (1983) generator UNIRAN.

# There is support for 100 streams, with seeds spaced
#   100,000 apart.

###############################################################

import math

# Define constants
MODLUS = 2147483647
MULT1 = 24112
MULT2 = 26143

def InitializeRNSeed():
    '''
    Set the default streams for the 100 streams.

    Output:
        list of 100 integers
    '''

    zrng = []
    zrng.append(1973272912)
    zrng.append(281629770)
    zrng.append(20006270)
    zrng.append(1280689831)
    zrng.append(2096730329)
    zrng.append(1933576050)
    zrng.append(913566091)
    zrng.append(246780520)
    zrng.append(1363774876)
    zrng.append(604901985)
    zrng.append(1511192140)
    zrng.append(1259851944)
    zrng.append(824064364)
    zrng.append(150493284)
    zrng.append(242708531)
    zrng.append(75253171)
    zrng.append(1964472944)
    zrng.append(1202299975)
    zrng.append(233217322)
    zrng.append(1911216000)
    zrng.append(726370533)
    zrng.append(403498145)
    zrng.append(993232223)
    zrng.append(1103205531)
    zrng.append(762430696)
    zrng.append(1922803170)
    zrng.append(1385516923)
    zrng.append(76271663)
    zrng.append(413682397)
    zrng.append(726466604)
    zrng.append(336157058)
    zrng.append(1432650381)
    zrng.append(1120463904)
    zrng.append(595778810)
    zrng.append(877722890)
    zrng.append(1046574445)
    zrng.append(68911991)
    zrng.append(2088367019)
    zrng.append(748545416)
    zrng.append(622401386)
    zrng.append(2122378830)
    zrng.append(640690903)
    zrng.append(1774806513)
    zrng.append(2132545692)
    zrng.append(2079249579)
    zrng.append(78130110)
    zrng.append(852776735)
    zrng.append(1187867272)
    zrng.append(1351423507)
    zrng.append(1645973084)
    zrng.append(1997049139)
    zrng.append(922510944)
    zrng.append(2045512870)
    zrng.append(898585771)
    zrng.append(243649545)
    zrng.append(1004818771)
    zrng.append(773686062)
    zrng.append(403188473)
    zrng.append(372279877)
    zrng.append(1901633463)
    zrng.append(498067494)
    zrng.append(2087759558)
    zrng.append(493157915)
    zrng.append(597104727)
    zrng.append(1530940798)
    zrng.append(1814496276)
    zrng.append(536444882)
    zrng.append(1663153658)
    zrng.append(855503735)
    zrng.append(67784357)
    zrng.append(1432404475)
    zrng.append(619691088)
    zrng.append(119025595)
    zrng.append(880802310)
    zrng.append(176192644)
    zrng.append(1116780070)
    zrng.append(277854671)
    zrng.append(1366580350)
    zrng.append(1142483975)
    zrng.append(2026948561)
    zrng.append(1053920743)
    zrng.append(786262391)
    zrng.append(1792203830)
    zrng.append(1494667770)
    zrng.append(1923011392)
    zrng.append(1433700034)
    zrng.append(1244184613)
    zrng.append(1147297105)
    zrng.append(539712780)
    zrng.append(1545929719)
    zrng.append(190641742)
    zrng.append(1645390429)
    zrng.append(264907697)
    zrng.append(620389253)
    zrng.append(1502074852)
    zrng.append(927711160)
    zrng.append(364849192)
    zrng.append(2049576050)
    zrng.append(638580085)
    zrng.append(547070247)
    return zrng

ZRNG = InitializeRNSeed()

def lcgrand(Stream):
    '''
    Obtains the next Uniform(0,1) random variate from Stream.

    Input: 
        Stream: integer, random number stream
    
    Output: 
        lcgrand: float
    '''

    MODLUS = 2147483647
    MULT1 = 24112
    MULT2 = 26143
    zi = ZRNG[Stream-1]  
    lowprd = (zi & 65535) * MULT1
    hi31 = (zi // 65536) * MULT1 + lowprd // 65536
    zi = ((lowprd & 65535) - MODLUS) + ((hi31 & 32767) * 65536) + (hi31 // 32768)  
    if zi < 0:
        zi += MODLUS
    lowprd = (zi & 65535) * MULT2
    hi31 = (zi // 65536) * MULT2 + (lowprd // 65536)
    zi = ((lowprd & 65535) - MODLUS) + ((hi31 & 32767) * 65536) + (hi31 // 32768)
    if zi < 0:
        zi += MODLUS
    ZRNG[Stream-1] = zi
    lcgrand = (zi // 128 | 1) / 16777216.0
    return lcgrand
    
def lcgrandst(zset,Stream):
    '''
    Set the seed for Stream to desired value zset.

    Input: 
        zset: integer
        Stream: integer, random number stream
    '''

    ZRNG[Stream-1] = zset
    
def lcgrandgt(Stream):
    '''
    Returns the current seed for Stream.

    Input:
        Stream: integer, random number stream

    Output:
        integer
    '''

    return ZRNG[Stream-1]

def Expon(Mean, Stream):
    '''
    Obtains an exponential random variate with given Mean
    using the next Uniform(0,1) in Stream. 

    Input: 
        Mean: integer, positive
        Stream: integer, random number stream

    Output:
        float
    '''

    Mean = float(Mean)
    return -math.log(1 - lcgrand(Stream)) * Mean
        
def Uniform(Lower, Upper, Stream):
    '''
    Obtains a Uniform(Lower,Upper) random variate 
    using the next Uniform(0,1) in Stream. 

    Input:
        Lower: float
        Upper: float, must be greater than Lower
        Stream: integer, random number stream

    Output:
        float
    '''

    Lower = float(Lower)
    Upper = float(Upper)
    return Lower + (Upper - Lower) * lcgrand(Stream)
        
def RandomInteger(prob_distrib, Stream):
    '''
    Obtains an random integer distributed according
    to the cumulative distribution function prob_distrib
    using the next Uniform(0,1) in Stream. 

    Input:
        prob_distrib: function, CDF of random integer
            to generate
        Stream: integer, random number stream

    Output:
        integer
    '''
    U = lcgrand(Stream)
    random_integer = 1
    while U >= prob_distrib[random_integer-1]:
        random_integer = random_integer + 1
    return random_integer
                
def Erlang(m, Mean, Stream):
    '''
    Obtains an Erlang random variate with m phases and
    given Mean using the next Uniform(0,1) in Stream.

    Input:
        m: integer, positive, number of phases
        Mean: float, positive
        Stream: integer, random number stream

    Output:
        float
    '''

    Mean = float(Mean)
    mean_exponential = Mean / m
    Sum = 0.0
    for i in range(0,m,1):
        Sum = Sum + Expon(mean_exponential, Stream)
    erlang = Sum
    return erlang
    
def Triangular(a, b, c, Stream):
    '''
    Obtains a Triangular random variate with lower
    limit a, mode b, and upper limit c, using the 
    next Uniform(0,1) in Stream.

    Input:
        a: float
        b: float, must be greater than a
        c: float, must be greater than b
        Stream: integer, random number stream

    Output:
        float
    '''

    a = float(a)
    b = float(b)
    c = float(c)
    Standardb = (b - a) / (c - a)
    U = lcgrand(Stream)
    if U <= Standardb:
        triangular = math.sqrt(Standardb * U)
    else:
        triangular = 1 - math.sqrt((1 - Standardb) * (1 - U))
    triangular = a + (c - a) * triangular
    return triangular

def Normal(Mean, Variance, Stream):
    '''
    Obtains a Normal random variate with given
    Mean and Variance using the next Uniform(0,1) in Stream.

    Input:
        Mean: float
        Variance: float, must be positive
        Stream: integer, random number stream

    Output:
        float
    '''

    Mean = float(Mean)
    Variance = float(Variance)
    U1 = lcgrand(Stream)
    U2 = lcgrand(Stream)
    V1 = 2 * U1 - 1
    V2 = 2 * U2 - 1
    W = V1 ** 2 + V2 ** 2
    while (W > 1):
        U1 = lcgrand(Stream)
        U2 = lcgrand(Stream)
        V1 = 2 * U1 - 1
        V2 = 2 * U2 - 1
        W = V1 ** 2 + V2 ** 2
    Y = math.sqrt(-2 * math.log(W) / W)
    normal = V1 * Y
    normal = Mean + math.sqrt(Variance) * normal
    return normal
    
def Lognormal(MeanPrime, VariancePrime, Stream):
    '''
    Obtains a Lognormal random variate with given
    MeanPrime and VariancePrime using the next 
    Uniform(0,1) in Stream.
    
    Input:
        MeanPrime: float, desired mean for lognormal
        VariancePrime: float, desired variance for
            lognormal, must be positive
        Stream: integer, random number stream

    Output:
        float
    '''

    MeanPrime = float(MeanPrime)
    VariancePrime = float(VariancePrime)
    Mean = math.log(MeanPrime ** 2 / math.sqrt(MeanPrime ** 2 + VariancePrime))
    Variance = math.log(1 + VariancePrime / MeanPrime ** 2)
    lognormal = math.exp(Normal(Mean, Variance, Stream))
    return lognormal