import numpy as np
from sklearn import metrics
'''
ctr_se = np.array([0.0034845027043735773, 0.06421403026414234, 0.31375048771296216, 0.5199022250923756, 0.6546323022841578, 0.7368592089809032, 0.79957744551047, 0.8493931428054546, 0.8937449804960219, 0.9337351333573712, 0.9891020819056435])
ctr_sp = np.array([6.987389486306633e-06, 0.00023304242102194545, 0.0027624288181834844, 0.010361145681971502, 0.025482224796159473, 0.051238087363232605, 0.09373013449941917, 0.1642160771657835, 0.28916264683498116, 0.5167109069202516, 0.90678248165739])

ctr_sp = 1-ctr_sp

print ctr_se
print ctr_sp
print metrics.auc(ctr_sp, ctr_se)

mf_se=np.array([0.0, 0.00991247114570264, 0.31306340424287005, 0.534801920337634, 0.6552082756595234, 0.7343896525608811, 0.7881940127911141, 0.8315411802506508, 0.8730170233625729, 0.9029180582523583, 0.9716382863010584])
mf_sp=np.array([0.0, 4.978041986718956e-05, 0.004030537153085315, 0.016134736540995055, 0.03729347774759432, 0.06926784326622139, 0.11682222364822452, 0.18988312126873408, 0.31535087969004083, 0.5372382510700638, 0.8778635892612725])

mf_sp = 1-mf_sp
print mf_se 
print mf_sp
print metrics.auc(mf_sp, mf_se)

model_se = np.array([0.06673025619749758, 0.33411999571430767, 0.5515528285832902, 0.6829257049461563, 0.7663564454695702, 0.8230071981746575, 0.8637650066901267, 0.895456836502046, 0.9206175630444489, 0.9464259295551223, 0.9951820895508127])
model_sp = np.array([0.00021418871631441604, 0.0027884526863573805, 0.011568529691439355, 0.029385421516055183, 0.059937104407910814, 0.10823411596514099, 0.18268947470046776, 0.2944533472574052, 0.45457496879856873, 0.6537082008233139, 0.9548987191927271])

model_sp = 1-model_sp
print model_se 
print model_sp
print metrics.auc(model_sp, model_se)

modelCTR_se = np.array([0.06651005003949488, 0.33635444991633134, 0.551890581569428, 0.6838218090665649, 0.7665934929222508, 0.8240223332042765, 0.8643378620423119, 0.89643968480727, 0.9210198155874754, 0.9465191051937256, 0.9955237491093805])
modelCTR_sp = np.array([0.0002,0.0028,0.0116,0.0294,0.0603,0.1088,0.1836,0.2957,0.4561,0.6554,0.9554])
'''
def computeAUC(x,y):
    x = np.array(x)
    y = np.array(y)
    x = 1-x
    print y 
    print x 
    print metrics.auc(x,y)


if __name__ == "__main__":
    #HoCTR
    print 'HoCTR:'
    specificity = [0.0005290624897287463, 0.0670901275482694, 0.1663485099157805, 0.22794178182628017, 0.2660248962834301, 0.29310142260416805, 0.3140909016634009, 0.3377058659743991, 0.3774722287164941, 0.4764726713865341, 0.8341067079350882]
    sensitivity = [1.2922169590474286e-07, 6.226359918658782e-06, 4.614762335476712e-05, 0.00012746424766066564, 0.0002874455249641509, 0.000730215747607871, 0.0022799937806747476, 0.009207159556391597, 0.047781523760771535, 0.22581264732510434, 0.6762335040180094]

    computeAUC(specificity, sensitivity)
    #CTR
    print 'CTR:'
    sensitivity = [0.0, 0.0, 0.0008021763392857143, 0.005272614988059212, 0.06935523291192873, 0.16625523647682372, 0.23305777555700113, 0.27376416119878827, 0.3069415564544228, 0.38116927593276734, 0.7656744568888376]

    specificity = [0.0, 0.0, 4.036367748849236e-08, 3.713855446275937e-07, 1.2355034521589012e-05, 6.360382414229073e-05, 0.00015029470896227873, 0.0004564141431066351, 0.006232665641373288, 0.09739045304491276, 0.5631845027114568]

    computeAUC(specificity, sensitivity)
    #fm HoPMF
    y = [0.06673025619749758, 0.33411999571430767, 0.5515528285832902, 0.6829257049461563, 0.7663564454695702, 0.8230071981746575, 0.8637650066901267, 0.895456836502046, 0.9206175630444489, 0.9464259295551223, 1.0]
    x = [0.00021418871631441604, 0.0027884526863573805, 0.011568529691439355, 0.029385421516055183, 0.059937104407910814, 0.10823411596514099, 0.18268947470046776, 0.2944533472574052, 0.45457496879856873, 0.6537082008233139, 1.0]
    computeAUC(x,y)
    #fm HoCTR
    print 'fm HoCTR:'
    y = [0.06651005003949488, 0.33635444991633134, 0.551890581569428, 0.6838218090665649, 0.7665934929222508, 0.8240223332042765, 0.8643378620423119, 0.89643968480727, 0.9210198155874754, 0.9465191051937256, 1.0]
    x = [0.0002149977438572579, 0.002808147972807602, 0.011640094440820873, 0.02953462938798069, 0.060281678264050835, 0.10883732845992869, 0.18357437261262186, 0.2957092936839687, 0.4560556090326984, 0.655424193266874, 1.0]
    computeAUC(x,y)


    #fm mf 
    print 'fm mf:'
    y = [0.0, 0.00991247114570264, 0.31306340424287005, 0.534801920337634, 0.6552082756595234, 0.7343896525608811, 0.7881940127911141, 0.8315411802506508, 0.8730170233625729, 0.9032368767909542, 1.0]
    x = [0.0, 4.978041986718956e-05, 0.004030537153085315, 0.016134736540995055, 0.03729347774759432, 0.06926784326622139, 0.11682222364822452, 0.18988312126873408, 0.31535087969004083, 0.5372334450656409, 1.0]
    computeAUC(x,y)


    #fm ctr 
    print 'fm ctr:'
    y = [0.0034845027043735773, 0.06421403026414234, 0.31375048771296216, 0.5199022250923756, 0.6546323022841578, 0.7368592089809032, 0.79957744551047, 0.8493931428054546, 0.8937449804960219, 0.9337351333573712, 1.0]
    x = [6.987389486306633e-06, 0.00023304242102194545, 0.0027624288181834844, 0.010361145681971502, 0.025482224796159473, 0.051238087363232605, 0.09373013449941917, 0.1642160771657835, 0.28916264683498116, 0.5167109069202516, 1.0]
    computeAUC(x,y)

    #del HoCTR
    print 'del HoCTR:'
    y = [0.0005290624897287463, 0.0670901275482694, 0.1663485099157805, 0.22794178182628017, 0.2660248962834301, 0.29310142260416805, 0.3140909016634009, 0.3377058659743991, 0.3774722287164941, 0.4764726713865341, 1.0]

    x = [1.2922169590474286e-07, 6.226359918658782e-06, 4.614762335476712e-05, 0.00012746424766066564, 0.0002874455249641509, 0.000730215747607871, 0.0022799937806747476, 0.009207159556391597, 0.047781523760771535, 0.22581264732510434, 1.0]

    computeAUC(x,y)



    #del HoPMF
    print 'del HoPMF:'
    y = [0.0003891592392730923, 0.067447349297805, 0.16615132285485637, 0.2288439329309133, 0.26668504787869757, 0.29001750265809517, 0.31231338546023535, 0.33761926586883667, 0.37598319495108673, 0.4717565644442156, 1.0]
    
    x = [1.1306488564221883e-07, 6.500860496887288e-06, 4.392668834457162e-05, 0.00012080962927075874, 0.0002677891846673, 0.0006656406935988428, 0.002047102274502602, 0.00827332441722025, 0.04332055524075309, 0.21186086388133898, 1.0]

    computeAUC(x,y)



    #del CTR
    print 'del CTR:'
    y = [0.0, 0.0, 0.0008021763392857143, 0.005272614988059212, 0.06935523291192873, 0.16625523647682372, 0.23305777555700113, 0.27376416119878827, 0.3069415564544228, 0.38116927593276734, 1.0]

    x = [0.0, 0.0, 4.036367748849236e-08, 3.713855446275937e-07, 1.2355034521589012e-05, 6.360382414229073e-05, 0.00015029470896227873, 0.0004564141431066351, 0.006232665641373288, 0.09739045304491276, 1.0]

    computeAUC(x,y)


    #del PMF 
    print 'del PMF:'
    y = [0.0, 0.0, 0.00017438616071428572, 0.004744233539082964, 0.06934621261923495, 0.16741244501310043, 0.233322433460959, 0.27108530978656425, 0.29910156810063826, 0.3417824062399521, 1.0]

    x = [0.0, 0.0, 2.4221819370027827e-08, 3.3106727935009686e-07, 1.2451523608110691e-05, 6.30046448263387e-05, 0.00014081056538686696, 0.0003405415247066658, 0.004217464722341676, 0.07644846155077341, 1.0]

    computeAUC(x,y)

