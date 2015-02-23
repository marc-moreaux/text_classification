

from datetime import datetime
from math import log, exp, sqrt, isnan, copysign
import sys
import re
import gc



###########################################################################
## Global parameters

train       = '/media/marc/MYLINUXLIVE/data/train.csv'  # path to training file
train_valid = '/media/marc/MYLINUXLIVE/data/train_valid.csv'
train_test  = '/media/marc/MYLINUXLIVE/data/train_test.csv'

label       = '/media/marc/MYLINUXLIVE/data/trainLabels.csv'  # path to label file of training data
label_valid = '/media/marc/MYLINUXLIVE/data/trainLabels_valid.csv'
label_test  = '/media/marc/MYLINUXLIVE/data/trainLabels_test.csv'

test  = '/media/marc/MYLINUXLIVE/data/test.csv'  # path to testing file

if sys.argv[1] == '1':
    print "SCHOOL COMPUTER"
    train       = '/home/people/m/moreaux/data/train.csv'  # path to training file
    train_valid = '/home/people/m/moreaux/data/train_valid.csv'
    train_test  = '/home/people/m/moreaux/data/train_test.csv'

    label       = '/home/people/m/moreaux/data/trainLabels.csv'  # path to label file of training data
    label_valid = '/home/people/m/moreaux/data/trainLabels_valid.csv'
    label_test  = '/home/people/m/moreaux/data/trainLabels_test.csv'

if sys.argv[1] == '2':
    print "SCHOOL COMPUTER VBOX"
    train       = '/vbox_data/data/train.csv'  # path to training file
    train_valid = '/vbox_data/data/train_valid.csv'
    train_test  = '/vbox_data/data/train_test.csv'

    label       = '/vbox_data/data/trainLabels.csv'  # path to label file of training data
    label_valid = '/vbox_data/data/trainLabels_valid.csv'
    label_test  = '/vbox_data/data/trainLabels_test.csv'





goodFeatIdx = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,117,118,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,119,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,120,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,121,122,123,124,125,126,127,128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143,144,145]
sameFeat = [0,29,58,87,116] # indexes of same features
mean5 = [ 0.02436, -0.0064185714285714, 618889.22367429, 618890.54749, 1.0543465334966, 0.059579818733216, 0.86829923814954, 0.18710997668475, 0.48828487341396, 0.31881, -0.046917142857143, 0.26766428571429, -0.040928571428571, 0.0745, 7.0484528571429, 0.4172980693322, 6.6242485714286, 8.4148614285714, 0.887768847231, -0.044420732569019, 0.86580343748149, 2528.6242714286, 2042.7330442857, 0.30792, -0.04657, 0.31071571428571, 2.8159057142857, 0.47087379071268, 0.47872133731683 ]
std5  = [ 0.37833990088304, 0.33602779279773, 400138.25163426, 400138.38151881, 0.45270002292835, 0.13513256575886, 0.26728160815194, 0.33899171823347, 0.27822487205236, 0.58017911684889, 0.26517707699206, 0.56165621992515, 0.27718862392012, 0.4340358994179, 9.8809850042776, 0.28243426355141, 8.1098755825395, 6.769063853486, 0.23895674404638, 0.85737470930449, 0.25034278943089, 1666.9023787093, 1451.3741280884, 0.5766653202001, 0.26589185575014, 0.57758902224881, 5.8285933708966, 0.28487838489089 ]
mean  = [ 0.22286082609966, 0.25372901238414, 0.14086704172456, 0.14086816877608, 0.218018245256, 0.031041402710335, 0.28915311653864, 0.04128743317118, 0.1480238184033, 0.23553563576215, 0.30525693915628, 0.19977904432753, 0.30342927378392, 0.23248570680994, 0.088860718662256, -0.0076013501595277, 0.26167021128057, 0.047658626465481, 0.28758121025037, 0.021016487194053, 0.3164575509962, 0.13580943142031, 0.11529782976295, 0.22691795345092, 0.33564129093959, 0.23017107422915, 0.00035514572183871, 0.065478779992211, 0.14438783982582, -0.08821621565139, -0.097283619690604, -0.049882371837154, -0.049881472732898, -0.064873867875063, -0.013080568077945, -0.10333728197705, 0.011545810817867, 0.010609659089997, -0.013377642086757, -0.10980263900446, 0.017028718591531, -0.089804757463995, 0.00047724821246821, -0.1114287405364, -0.058897301273351, -0.042567325573875, 0.010725897596154, -0.099632065503987, 0.045999930112593, -0.10428183465943, -0.045568501087608, -0.037995558094089, -0.016575596093166, -0.11247322412826, -0.015913440556129, 0.033312129494629, -0.063534873755062, 0.011203160324143, -0.065221020114995, -0.072732079083468, -0.03018436730831, -0.030183445429499, -0.063919177806218, -0.011945490045466, -0.0621367650583, -0.035714523651697, -0.090741481620846, -0.024913488428856, -0.06349934775495, 0.063409403209429, -0.046126403403149, 0.025771403473063, -0.12174292335018, -0.031929298045028, -0.10934481646695, 0.0034765305295115, -0.0339333954076, 0.045253256181679, -0.058893212476627, -0.031985256413922, -0.025798112376745, -0.026806833855013, -0.077684644486008, -0.027377350562019, 0.022270504395412, -0.054506755694721, -0.09114820672354, 0.061951095757519, 0.094643531947345, 0.037297831934427, 0.037298835265789, 0.059404234682169, 0.025199560039041, 0.08043779404578, 0.011086442820003, 0.034925245256763, 0.20566801225516, 0.06474379921268, 0.036333867823772, 0.066999245166571, 0.074434264309685, -0.082730040108399, 0.1734703894604, 0.21998470300452, 6.0780719510745e-05, 0.037962761628598, 0.026712270351416, 0.077193904907123, 0.036032768271267, 0.030488792262002, 0.19606321584213, 0.099266995748578, 0.19906888041621, 0.08678251643463, 0.23397711411856, 0.033318014024823, -0.13137468609135, -0.17835684555636, -0.098098134513607, -0.098102085878751, -0.14862943424936, -0.031214904626934, -0.20411686354442, -0.028205163157146, -0.10281724112987, -0.40291251750145, -0.19669875161077, -0.31655103395237, -0.23449735808298, -0.33316862280495, 0.22704098533248, -0.075042439981989, -0.32974277224421, -0.061921835310658, -0.19197851097156, -0.13898194383947, -0.2304764087685, -0.094288442190057, -0.08199295155432, -0.37959873934413, -0.2447504180731, -0.38594916352773, -0.14272029604632, -0.18141426466092, -0.097760807451822 ]
std   = [ 1.2601155962165, 1.2997781757533, 1.0525409240073, 1.0525410678881, 1.1652583775605, 0.97838531198209, 1.3269201341206, 0.98181237443719, 1.0878340620176, 1.1431720660738, 1.444380781377, 1.1568909133549, 1.3983814188893, 1.1833722101284, 0.90860950987371, 1.0431065085841, 0.5649433633085, 1.042685493819, 1.4298311269073, 0.55376377882472, 1.3795322690667, 1.04642074765, 1.0456248844651, 1.1458635804685, 1.3902997752835, 1.1446286048678, 0.73408061773135, 1.0599774376776, 1.0840738010318, 0.88969012735912, 0.86498085113889, 0.97638334364025, 0.97638314714497, 0.93146841133833, 1.0173675198884, 0.83308256003657, 0.9688371478156, 0.94668402569309, 0.91019986203755, 0.76196334331616, 0.89098782182971, 0.74505041311788, 0.8233775833408, 1.0466265486436, 0.99833888404769, 1.1348740814553, 0.97282604749904, 0.78777597171903, 0.72749532810249, 0.80400394901074, 0.97871764829341, 0.97973077266107, 0.91041625293646, 0.77077154767761, 0.91041118834155, 0.63986493771259, 0.99094389821717, 0.94906921203297, 0.94875489753011, 0.94066043905343, 0.98569081749591, 0.98569066203348, 0.94047259928184, 1.0130982858554, 0.90199005328897, 1.038280072051, 0.98158878853316, 0.95109160817672, 0.85415071345103, 0.90870991270229, 0.83329114377762, 0.8669502690679, 1.0658131118929, 1.0091099959929, 1.3970309252923, 0.99856454378416, 0.92198082364582, 0.70968029716553, 0.88885704329075, 0.98810721068827, 0.9874854987096, 0.95153836796792, 0.88761296942345, 0.95175483597699, 0.68397077964101, 1.0067662841364, 0.98219765477349, 1.0763594458926, 1.0531348257214, 1.0156491431586, 1.0156491198919, 1.0521948370275, 0.94796094837331, 1.1059206235646, 0.9944086602587, 1.0275590965076, 0.98214292737163, 1.1733285581873, 1.0550168852901, 1.1519683034039, 1.044213876393, 1.0128200637609, 0.94130384015377, 0.54617547261907, 1.0072416466889, 1.123806117672, 0.59866475972989, 1.1190163759593, 1.0137235318178, 1.0135129012576, 0.98385251420187, 1.1074801108532, 0.98309569111171, 0.65491618760012, 0.93735173788409, 1.0264743113023, 0.69217234338754, 0.66975852459268, 0.94946020666284, 0.94945989716023, 0.83756731662593, 1.0391675575133, 0.60195882066878, 1.0131740068412, 0.92681802081751, 0.85710329746521, 0.27345459559606, 0.88499164376148, 0.55345445135389, 0.95313469110103, 0.90760214813818, 0.9852163053506, 0.95055799048802, 0.97393289977687, 0.29992143904426, 1.8103712139354, 0.48442175871172, 0.95468322113861, 0.95991322769794, 0.8658831658709, 0.50748300667926, 0.86378388621157, 1.7679054107383, 0.95044604952923, 0.9304673803808 ]

L1 =    .1
L2 =    .1
alpha = .1
beta =  .1

###########################################################################
## Classes

class Parameters:
    def __init__(self):
        self.hashTrick_toDo     = []
        self.hashTrick_toDo_all = []
        self.idx_toDo           = []
        self.idx_toDoAll        = []
        self.idxNb_P_feat       = [ 0, 2, 2, 786837, 192912, 51217, 37612, 41951, 50551, 136527, 2, 2, 2, 2, 2, 131, 136405, 204, 111, 279097, 970, 810, 500, 420, 2, 2, 2, 497, 139005, 134989,2, 2, 786837, 192912, 51217, 37612, 41951, 50551, 136527, 2, 2, 2, 2, 2, 131, 136405, 204, 111, 279097, 970, 810, 500, 420, 2, 2, 2, 497, 139005, 134989,2, 2, 786837, 192912, 51217, 37612, 41951, 50551, 136527, 2, 2, 2, 2, 2, 131, 136405, 204, 111, 279097, 970, 810, 500, 420, 2, 2, 2, 497, 139005, 134989,2, 2, 786837, 192912, 51217, 37612, 41951, 50551, 136527, 2, 2, 2, 2, 2, 131, 136405, 204, 111, 279097, 970, 810, 500, 420, 2, 2, 2, 497, 139005, 134989,2, 2, 786837, 192912, 51217, 37612, 41951, 50551, 136527, 2, 2, 2, 2, 2, 131, 136405, 204, 111, 279097, 970, 810, 500, 420, 2, 2, 2, 497, 139005, 134989 ]
        self.idx_dico           = {}
        self.feat_toDo          = []
        self.feat_toDoAll       = []
        self.neighbhor_toDo     = False
        self.eq_toDo            = False
        self.eqAll_toDo         = False


        self.elected_hashTrick  = {} # hashTrick index + new logloss
        self.elected_idx        = {} # idx  number     + new logloss
        self.elected_feat       = {} # feat number     + new logloss
        self.elected_neighbhor  = (False,0)
        self.elected_eq         = (False,0)
        self.elected_eqAll      = (False,0)


    def try_idx(self,to_add):
        if type(to_add) is int:
            if to_add > 29:
                print "ERROR : Parameters.try_idx(", str(to_add), ') with to_add is int'
                exit()
            self.idx_toDo.append(to_add)

        elif type(to_add) is list:
            for i in to_add:
                if to_add > 29:
                    print "ERROR : Parameters.try_idx(", str(to_add), ') with to_add is list'
                    exit()
                self.idx_toDo.append(i)

        self.refresh_todo_all()

    def try_feat(self,to_add):
        if type(to_add) is int:
            if to_add > 29:
                print "ERROR : Parameters.try_idx(", str(to_add), ') with to_add is int'
                exit()
            self.feat_toDo.append(to_add)

        elif type(to_add) is list:
            for i in to_add:
                if to_add > 29:
                    print "ERROR : Parameters.try_idx(", str(to_add), ') with to_add is list'
                    exit()
                self.feat_toDo.append(i)

        self.refresh_todo_all()

    def try_neighbhor(self):
        #
        self.neighbhor_toDo = True

    def try_eq(self):
        #
        self.eq_toDo = True

    def try_eqAll(self):
        #
        self.eqAll_toDo = True



    def add_idx(self, to_add, loss):
        if type(to_add) is int:
            if to_add > 29:
                print "ERROR : Parameters.add_idx(", str(to_add), ') with to_add is over 29'
                exit()
            self.elected_idx[to_add] = loss

        self.refresh_todo()

    def add_feat(self, to_add, loss):
        if type(to_add) is int:
            if to_add > 29:
                print "ERROR : Parameters.add_feat(", str(to_add), ') with to_add is over 29'
                exit()
            self.elected_feat[to_add] = loss

        self.refresh_todo()

    def add_neighbhor(self, loss):
        self.elected_neighbhor = (True,loss)
        self.refresh_todo()

    def add_eq(self, loss):
        self.elected_eq = (True,loss)
        self.refresh_todo()

    def add_eqAll(self, loss):
        self.elected_eqAll = (True,loss)
        self.refresh_todo()



    def refresh_todo(self):
        self.hashTrick_toDo     = [ k for k,v in self.elected_hashTrick.items() ]
        self.idx_toDo           = [ k for k,v in self.elected_idx.items()       ]
        self.feat_toDo          = [ k for k,v in self.elected_feat.items()      ]

        self.neighbhor_toDo     = self.elected_neighbhor[0]
        self.eq_toDo            = self.elected_eq[0]
        self.eqAll_toDo         = self.elected_eqAll[0]

        self.refresh_todo_all()

    def refresh_todo_all(self):
        self.hashTrick_toDo_all  = [ i+j for i in self.hashTrick_toDo for j in sameFeat]
        self.idx_toDoAll         = [ i+j for i in self.idx_toDo  for j in sameFeat]
        self.feat_toDoAll        = [ i+j for i in self.feat_toDo for j in sameFeat]
        for i in self.idx_toDo: self.idx_dico[i%29] = {} # {3: {}, 4: {}}


    def try_reset(self):
        self.hashTrick_toDo     = []
        self.hashTrick_toDo_all = []
        self.idx_toDo           = []
        self.idx_toDoAll        = []
        self.idx_dico           = {}
        self.neighbhor_toDo     = False
        self.eq_toDo            = False
        self.eqAll_toDo         = False

        self.refresh_todo()


class Dimentions:
    def __init__(self,param):
        self.hashTrick = 2 ** 3
        self.copy      = 146
        self.idx       = sum( param.idxNb_P_feat[i] for i in param.idx_toDoAll ) # is the amount of indexes in the features to indexize
        self.neighbhor = 5
        self.eq        = 45
        self.eqAll     = 10440

class Weights:
    def __init__(self,D):
        self.hashTrick  = [[0.] * D.hashTrick for k in range(33)]
        self.copy       = [[0.] * D.copy      for k in range(33)]
        self.idx        = [[0.] * D.idx       for k in range(33)]
        self.neighbhor  = [[0.] * D.neighbhor for k in range(33)]
        self.eq         = [[0.] * D.eq        for k in range(33)]
        self.eqAll      = [[0.] * D.eqAll     for k in range(33)]

class N_accumulator:
    def __init__(self,D):
        self.hashTrick  = [[0.]     * D.hashTrick for k in range(33)]
        self.copy       = [[1e-15]  * D.copy      for k in range(33)]
        self.idx        = [[0.]     * D.idx       for k in range(33)]
        self.neighbhor  = [[1e-15]  * D.neighbhor for k in range(33)]
        self.eq         = [[1e-15]  * D.eq        for k in range(33)]
        self.eqAll      = [[1e-15]  * D.eqAll     for k in range(33)]

class Z_accumulator:
    def __init__(self,D):
        self.hashTrick  = [[0.]     * D.hashTrick for k in range(33)]
        self.copy       = [[1e-15]  * D.copy      for k in range(33)]
        self.idx        = [[0.]     * D.idx       for k in range(33)]
        self.neighbhor  = [[1e-15]  * D.neighbhor for k in range(33)]
        self.eq         = [[1e-15]  * D.eq        for k in range(33)]
        self.eqAll      = [[1e-15]  * D.eqAll     for k in range(33)]


class X_input:
    def __init__(self,D,param):
        self.hashTrick  = [0]  * 146
        self.copy       = [0.] * D.copy
        self.idx        = [0]  * len(param.idx_toDoAll) # size is the amount of features to indexize
        self.neighbhor  = [0.] * D.neighbhor
        self.eq         = [0] * D.eq
        self.eqAll      = [0] * D.eqAll



def getDicoIdx(dico, val):
    """Maps an entry val with a number"""
    if val not in dico:
        dico[val] = len(dico)
    return dico[val]

def indexize(featureID):
    """insert a indexed feature in x.indexed"""
    for k,v in enumerate( [featureID+i for i in sameFeat] ): # [1:featureID, 2:featureID+29 ...]
        x.indexed[k] = getDicoIdx( param.idx_dico[featureID] , x.copy[v] )





###########################################################################
## Functions

def update_eq(x,i,j,feat):
    idxInX = i*9 + j - i*(i+1)/2 -1
    if feat == 0:
        print "OULAH !!!" 


    if x.eq[ idxInX ] == 0 :
        x.eq[ idxInX ] = feat
    elif x.eq[ idxInX ] == feat :
        x.eq[ idxInX ] = 1
    else :
        x.eq[ idxInX ] = 0
    return x

def update_eqAll(x,i,j,feat):
    idxInX = i*144 + j - i*(i+1)/2 -1
    if feat == 0:
        print "OULAH !!!" 


    if x.eqAll[ idxInX ] == 0 :
        x.eqAll[ idxInX ] = feat
    elif x.eqAll[ idxInX ] == feat :
        x.eqAll[ idxInX ] = 1
    else :
        x.eqAll[ idxInX ] = 0
    return x

def data(D,x,param, path, label_path=None):
    for t, line in enumerate(open(path)):
        hashList = {}

        # if t == 2:
        #     print x.eqAll
        #     return

        # initialize our generator
        if t == 0:
            # check if labels
            if label_path:
                label = open(label_path)
                label.readline()  # we don't need the headers
            continue
        # parse x
        for m, feat in enumerate(line.rstrip().split(',')):
            if m == 0:
                ID = int(feat)
                x.idx = [0] * len(param.idx_toDoAll)
                x.eq = [0] * D.eq
                x.eqAll = [0] * D.eqAll

            else:
                m = goodFeatIdx[m]


                ##################################
                #  ==> HASH TRICK
                # one-hot encode everything with hash trick
                if m in param.hashTrick_toDo_all:
                    x.hashTrick[m] = abs(hash(str(m) + '_' + feat)) % D.hashTrick
                else :
                    x.hashTrick[m] = 0




                ##################################
                #  ==> INDEX
                # care about the indexized part of the inputs
                if m in param.idx_toDoAll : 
                    x.idx[ param.idx_toDoAll.index(m) ] = getDicoIdx(param.idx_dico[m%29] , feat)




                ##################################
                #  ==> COPY
                # copy the features
                if m in param.feat_toDo:
                    if feat == '':
                        x.copy[m] = 0
                    elif feat == 'YES':
                        x.copy[m] = 1
                    elif feat == 'NO':
                        x.copy[m] = -1
                    elif re.search('([^,]+=)', feat ):
                        x.copy[m] = getDicoIdx(hashList,feat)
                    else :
                        x.copy[m] = ( float(feat) - mean[m-1] ) / std[m-1]
                        if len(sys.argv) > 3 :
                            if sys.argv[3] == '1':
                                x.copy[m] = ( float(feat) - mean[(m-1)%29] ) / std[(m-1)%29]
                            if sys.argv[3] == '2':
                                x.copy[m] = float(feat)
                else :
                    x.copy[m] = 0





                ##################################
                #  ==> NEIGHBHOR
                # look if neighboor is empty
                # if feature 5 == -1 then neighbor is empty
                if param.neighbhor_toDo :
                    if not (m-5)%29 and feat == -1:
                        x.neighbhor[(m-5)/29] = 1 # (m-5)/29 is in range [0,4]
                    else:
                        x.neighbhor[(m-5)/29] = 0




                ##################################
                #  ==> EQ
                # cross equality hash
                if param.eq_toDo:
                    # idx 3 and 4 are hashs
                    if (m-3)%29 == 0   or   (m-4)%29 == 0:
                        hashInnerIdx = (m-3)/29 if (m-3)%29==0 else (m-4)/29+5
                        for i in range(0,hashInnerIdx):
                            x = update_eq(x, i,hashInnerIdx, hash(feat))
                        for j in range(hashInnerIdx+1,10):
                            x = update_eq(x, hashInnerIdx,j, hash(feat))




                ##################################
                #  ==> EQ_ALL
                # cross equality between all features
                if param.eqAll_toDo:
                    hashInnerIdx = m-1
                    for i in range(0,hashInnerIdx):
                        x = update_eqAll(x, i,hashInnerIdx, hash(feat))
                    for j in range(hashInnerIdx+1,145):
                        x = update_eqAll(x, hashInnerIdx,j , hash(feat))



        # parse y, if provided
        if label_path:
            # use float() to prevent future type casting, [1:] to ignore id
            y = [float(y) for y in label.readline().split(',')[1:]]
        yield (ID, x, y) if label_path else (ID, x)

def logloss(p, y):
    p = max(min(p, 1. - 10e-15), 10e-15)
    return -log(p) if y == 1. else -log(1. - p)

def justPredict(label,w,x):
    wTx = 0.
    # HASH TRICK
    for i in x.hashTrick:
        wTx += w.hashTrick[label][i] * 1.

    # ONE HOT ENCODING
    for i in x.idx:
        wTx += w.idx[label][i] * 1.

    # ORIGINAL FEATUNRE
    for k,v in enumerate(x.copy) :
        wTx +=   w.copy[label][k] * x.copy[k]

    # NEIGHBOR PARAM
    for k,v in enumerate(x.neighbhor) :
        wTx +=   w.neighbhor[label][k] * x.neighbhor[k]

    # HASH PAIR EQUALITY
    for k,v in enumerate(x.eq) :
        wTx +=   w.eq[label][k] * x.eq[k]

    # ALL FEATURE PAIR EQUALITY
    for k,v in enumerate(x.eqAll) :
        wTx +=   w.eqAll[label][k] * x.eqAll[k]

    return 1. / (1. + exp(-max(min(wTx, 30.), -30.)))  # bounded sigmoid


def predict(label,w,z,n,x):
    wTx = 0.
    
    # for i in x.hashTrick: # do wTx
    #     wTx += w.hashTrick[label][i] * 1.  # w[i] * x[i], but if i in x we got x[i] = 1.

    # HASH TRICK
    for i in x.hashTrick:
        sign = -1. if z.hashTrick[label][i] < 0 else 1.
        if sign * z.hashTrick[label][i] <= L1:
            w.hashTrick[label][i] = 0.
        else:
            w.hashTrick[label][i] = (sign * L1 - z.hashTrick[label][i]) / ((beta + sqrt(n.hashTrick[label][i])) / alpha + L2)
        wTx += w.hashTrick[label][i] * 1.

    # ONE HOT ENCODING
    for i in x.idx:
        sign = -1. if z.idx[label][i] < 0 else 1.
        if sign * z.idx[label][i] <= L1:
            w.idx[label][i] = 0.
        else:
            w.idx[label][i] = (sign * L1 - z.idx[label][i]) / ((beta + sqrt(n.idx[label][i])) / alpha + L2)
        wTx += w.idx[label][i] * 1.

    # ORIGINAL FEATUNRE
    for k,v in enumerate(x.copy) :
        sign = -1. if z.copy[label][k] < 0 else 1.
        if sign * z.copy[label][k] <= L1:
            w.copy[label][k] = 0.
        else:
            w.copy[label][k] = (sign * L1 - z.copy[label][k]) / ((beta + sqrt(n.copy[label][k])) / alpha + L2)
        wTx +=   w.copy[label][k] * x.copy[k]

    # NEIGHBOR PARAM
    for k,v in enumerate(x.neighbhor) :
        sign = -1. if z.neighbhor[label][k] < 0 else 1.
        if sign * z.neighbhor[label][k] <= L1:
            w.neighbhor[label][k] = 0.
        else:
            w.neighbhor[label][k] = (sign * L1 - z.neighbhor[label][k]) / ((beta + sqrt(n.neighbhor[label][k])) / alpha + L2)
        wTx +=   w.neighbhor[label][k] * x.neighbhor[k]

    # HASH PAIR EQUALITY
    for k,v in enumerate(x.eq) :
        sign = -1. if z.eq[label][k] < 0 else 1.
        if sign * z.eq[label][k] <= L1:
            w.eq[label][k] = 0.
        else:
            w.eq[label][k] = (sign * L1 - z.eq[label][k]) / ((beta + sqrt(n.eq[label][k])) / alpha + L2)
        wTx +=   w.eq[label][k] * x.eq[k]

    # ALL FEATURE PAIR EQUALITY
    for k,v in enumerate(x.eqAll) :
        sign = -1. if z.eqAll[label][k] < 0 else 1.
        if sign * z.eqAll[label][k] <= L1:
            w.eqAll[label][k] = 0.
        else:
            w.eqAll[label][k] = (sign * L1 - z.eqAll[label][k]) / ((beta + sqrt(n.eqAll[label][k])) / alpha + L2)
        wTx +=   w.eqAll[label][k] * x.eqAll[k]


    # for k,v in enumerate(x.copy) :
    #     wTx +=   w.copy[label][k] * x.copy[k]


    return 1. / (1. + exp(-max(min(wTx, 30.), -30.)))  # bounded sigmoid

def update(alpha, label, p, y, w,z,n,x,param):

    # if param.hashTrick_toDo :
    #     for i in x.hashTrick:
    #         n.hashTrick[label][i] += (p - y)**2 * .1
    #         w.hashTrick[label][i] -= alpha / sqrt(n.hashTrick[label][i])  *  (p - y) * 1.


    # HASH TRICK
    if param.hashTrick_toDo :
        for i in x.hashTrick:
            g = (p-y)*1.
            sigma = (sqrt(n.hashTrick[label][i] + g**2) - sqrt(n.hashTrick[label][i])) / alpha
            z.hashTrick[label][i] += g - sigma * w.hashTrick[label][i]
            n.hashTrick[label][i] += g**2 

    # ONE HOT ENCODING
    for i in x.idx:
        g = (p-y)*1.
        sigma = (sqrt(n.idx[label][i] + g**2) - sqrt(n.idx[label][i])) / alpha
        z.idx[label][i] += g - sigma * w.idx[label][i]
        n.idx[label][i] += g**2 

    # ORIGINAL FEATUNRE
    for k,v in enumerate(x.copy):
        g = (p-y)*x.copy[k]
        sigma = (sqrt(n.copy[label][k] + g**2) - sqrt(n.copy[label][k])) / alpha
        z.copy[label][k] += g - sigma * w.copy[label][k]
        n.copy[label][k] += g**2

    # NEIGHBOR PARAM
    if param.neighbhor_toDo:
        for k,v in enumerate(x.neighbhor):
            g = (p-y)*x.neighbhor[k]
            sigma = (sqrt(n.neighbhor[label][k] + g**2) - sqrt(n.neighbhor[label][k])) / alpha
            z.neighbhor[label][k] += g - sigma * w.neighbhor[label][k]
            n.neighbhor[label][k] += g**2

    # HASH PAIR EQUALITY
    if param.eq_toDo:
        for k,v in enumerate(x.eq):
            g = (p-y)*x.eq[k]
            sigma = (sqrt(n.eq[label][k] + g**2) - sqrt(n.eq[label][k])) / alpha
            z.eq[label][k] += g - sigma * w.eq[label][k]
            n.eq[label][k] += g**2

    # ALL FEATURE PAIR EQUALITY
    if param.eqAll_toDo:
        for k,v in enumerate(x.eqAll):
            g = (p-y)*x.eqAll[k]
            sigma = (sqrt(n.eqAll[label][k] + g**2) - sqrt(n.eqAll[label][k])) / alpha
            z.eqAll[label][k] += g - sigma * w.eqAll[label][k]
            n.eqAll[label][k] += g**2


    # for k,v in enumerate(x.copy):
    #     n.copy[label][k] += ( (p-y)*x.copy[k] )**2
    #     w.copy[label][k] -= (p-y)*x.copy[k] * alpha / sqrt(n.copy[label][k])

    # if param.neighbhor_toDo:
    #     for k,v in enumerate(x.neighbhor):
    #         n.neighbhor[label][k] += ( (p-y)*x.neighbhor[k] )**2
    #         w.neighbhor[label][k] -= (p-y)*x.neighbhor[k] * alpha / sqrt(n.neighbhor[label][k])

    # if param.eq_toDo:
    #     for k,v in enumerate(x.eq):
    #         n.eq[label][k] += ( (p-y)*x.eq[k] )**2
    #         w.eq[label][k] -= (p-y)*x.eq[k] * alpha / sqrt(n.eq[label][k])

    # if param.eqAll_toDo:
    #     for k,v in enumerate(x.eqAll):
    #         n.eqAll[label][k] += ( (p-y)*x.eqAll[k] )**2
    #         w.eqAll[label][k] -= (p-y)*x.eqAll[k] * alpha / sqrt(n.eqAll[label][k])



###########################################################################
## Training and testing 

def testing(D, w,x, param, isTest = True):
    train_dataset = train_test if isTest else train_valid
    label_dataset = label_test if isTest else label_valid

    loss = 0.
    for t,(ID, x, y) in enumerate(data(D,x,param, train_dataset, label_dataset)):
        for k in range(33):
            p = justPredict(k,w,x)
            loss += logloss(p, y[k])

        if t == 10000:
            return loss/33./float(t)



def onLineTraining(D, w,z,n,x, param):
    start = datetime.now()

    loss = 0.
    prevLoss = 1.
    end = False
    for ID, x, y in data(D,x,param, train, label):
        # get predictions and train on all labels
        for k in range(33):
            p = predict(k,w,z,n,x)
            update(alpha, k, p, y[k], w,z,n,x,param)
            loss += logloss(p, y[k])  # for progressive validation

        # print out progress, so that we know everything is working
        if ID % 1000 == 0:
            validLoss = (loss/33.)/ID #testing(D, w,x, param, False)
            curentLoss = loss/33./ID
            ratioPrev = 100.-curentLoss/prevLoss*100.
            print('%s\tencountered: %d\t current logloss: %f\t difference : %f' % (
                datetime.now(), ID, curentLoss , ratioPrev ) )
            

            if ratioPrev < 3 and ID > 5000 : 
                end = True
            else :
                prevLoss = curentLoss


        if ID % 1360000 == 0 or end == True:
            return testing(D,w,x, param, True)
        

    # with open('./submission1234.csv', 'w') as outfile:
    #     outfile.write('id_label,pred\n')
    #     for ID, x in data(test):
    #         for k in K:
    #             p = predict(x, w[k])
    #             outfile.write('%s_y%d,%s\n' % (ID, k+1, str(p)))
    #             if k == 12:
    #                 outfile.write('%s_y14,0.0\n' % ID)

    print('Done, elapsed time: %s' % str(datetime.now() - start))
    #
    return loss/33.






###########################################################################
## Chose what to do here

def bigPrint(msg):
    print "**************************************************************************"
    print "*******"
    print "*******  => "+ msg
    print "*******"
    print "**************************************************************************"


def parameter_elector(chooseFrom):
    bigPrint(str(chooseFrom))
    param = Parameters()
    endLoop = False
    prevLoss = 1.
    for i in range(15):
        if not endLoop:
            best_training = (0,10) # initialisation with bad loss

            for new_try in chooseFrom:


                #######
                ## Try using new parameter 'i'
                ## If not already used
                ######

                # skip parameter already in final set
                param.try_reset()
                if new_try in param.feat_toDo:
                    continue
                if new_try-29 in param.idx_toDo:
                    continue 
                if new_try==29*2   and param.neighbhor_toDo:
                    continue
                if new_try==29*2+1 and param.eq_toDo:
                    continue
                if new_try==29*2+2 and param.eqAll_toDo:
                    continue


                # try parameters
                if new_try < 29:
                    param.try_feat(new_try)
                elif new_try < 29*2:
                    param.try_idx(new_try-29)
                elif new_try < 29*2+1:
                    param.try_neighbhor()
                elif new_try < 29*2+2:
                    param.try_eq()
                else :
                    param.try_eqAll()

                #######
                ## create D,w,x,n depending on new parameters
                ######
                # gc.get_referrers(thatobject)

                D = Dimentions(param)
                w = Weights(D)
                n = N_accumulator(D)
                z = Z_accumulator(D)
                x = X_input(D,param)

                print ''
                print '**************************************'
                print 'training started for ',str(param.feat_toDo),str(param.idx_toDo),str(param.neighbhor_toDo),str(param.eq_toDo),str(param.eqAll_toDo)
                loss = onLineTraining( D,w,z,n,x, param)
                print '\n======>  ',str(loss),' <======\t sizeIdx : ',str(D.idx)
                if loss < best_training[1] :
                    best_training = (new_try,loss)

                del x
                del z
                del n
                del w
                del D
                gc.collect()

            #######
            ## update param with new best trial
            ######
            if best_training[0] < 29:
                param.add_feat( best_training[0], best_training[1] )
            elif best_training[0] < 29*2:
                param.add_idx( best_training[0]-29, best_training[1] )
            elif best_training[0] < 29*2+1:
                param.add_neighbhor( best_training[1] )
            elif best_training[0] < 29*2+2:
                param.add_eq( best_training[1] )
            else :
                param.add_eqAll( best_training[1] )

            #######
            ## termination condition
            ######
            currentLoss = best_training[1]
            print prevLoss,"==>",currentLoss
            print 100.-currentLoss/prevLoss*100.
            ratio = 100.-currentLoss/prevLoss*100.
            if ratio < 3 :
                print "\n\nTraining ended with ",str(param.feat_toDo),str(param.idx_toDo),str(param.neighbhor_toDo),str(param.eq_toDo),str(param.eqAll_toDo)
                print "Last improvment was ",ratio,"%"
                endLoop = True
            prevLoss = currentLoss




# features
if int(sys.argv[2]) <= -1:
    parameter_elector(  [17]  )
    exit()

# features
if int(sys.argv[2]) <= 0:
    parameter_elector(  [i for i in range(0,29)]  )

# features & oneHot
if int(sys.argv[2]) <= 1:
    parameter_elector(  [i for i in range(0,29*2)]  )

# features & neighbor
if int(sys.argv[2]) <= 2:
    parameter_elector(  [i for i in range(0,29)]+[29*2]  )

# features & eq
if int(sys.argv[2]) <= 3:
    parameter_elector(  [i for i in range(0,29)]+[29*2+1]  )

# features & eqAll
if int(sys.argv[2]) <= 4:
    parameter_elector(  [i for i in range(0,29)]+[29*2+2]  )

# all the features together
if int(sys.argv[2]) <= 5:
    parameter_elector(  [i for i in range(0,29*2+3)]  )




# param = Parameters()
# # param.add_idx(  17 , 0.081280612904  )
# param.add_idx(  3  , 0.0663415727417 )
# D = Dimentions(param)
# w = Weights(D)
# n = N_accumulator(D)
# z = Z_accumulator(D)
# x = X_input(D,param)

# loss = onLineTraining( D,w,z,n,x, param)
# print 'training ended for ',str(param.idx_toDo),str(param.feat_toDo),' with loss of : ', str(loss),' \t sizeIdx : ',str(D.idx)

