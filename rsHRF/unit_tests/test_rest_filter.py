import pytest
import numpy as np 
from ..processing import rest_filter 

def test_nextpow2():
    number_of_testcases = 10
    for _ in range(10):
        var = np.random.randint(10)
        out_exp = rest_filter.nextpow2(var)
        assert type(out_exp) == np.int_
        assert out_exp == np.ceil(np.log2(np.abs(var))).astype('long')

def test_rest_nextpow2_one35():
    tests = [288377, 344098, 159194, 581494, 182792, 7453  , 641928, 229007, 78752 , 540883]
    output_expected = [327680, 393216, 163840, 655360, 196608, 7680, 655360, 245760, 81920, 655360]
    for i in range(len(tests)):
        out = rest_filter.rest_nextpow2_one35(tests[i])
        assert type(i) == int
        assert out == output_expected[i]

def test_rest_IdealFilter():
    Data = np.random.random((10,15))
    SamplePeriod = np.random.uniform(1, 10)
    low = np.random.uniform(0, 1.7976931348623157e+308)
    high = np.random.uniform(0, 1.7976931348623157e+308)
    Band = [min(low, high), max(low, high)]
    out = rest_filter.rest_IdealFilter(Data, SamplePeriod, Band)
    assert type(out) == type(Data)
    assert out.shape == Data.shape
    Data = [[0.7700066884342136, 0.22668530780853569, 0.045017256299809016, 0.0020746842925888354, 0.833716775833151, 0.3934317692627358, 0.9023572022133992, 0.11361685468856986, 0.8732352442933455, 0.6984111146595482, 0.827278166239948, 0.4014690894109776, 0.6512541750474559, 0.9798568538584171, 0.07105471576346467], [0.6146247877956595, 0.20798872322544593, 0.1557395482294427, 0.8249348051196937, 0.14354041414659213, 0.34593452355881193, 0.07429627764435298, 0.6405220265186684, 0.8516331808661464, 0.4433116550423968, 0.30042869236862035, 0.18612111599971426, 0.12704622017818445, 0.46990952235095784, 0.8907077661690574], [0.4855095026782408, 0.5573389017032796, 0.09137977746449, 0.3330894195540509, 0.33213432216579675, 0.4395032806692851, 0.22494116304472322, 0.3566400165790914, 0.40804154667706316, 0.42461499775787825, 0.7121852997273149, 0.3727764162917079, 0.2887988984086659, 0.7247287273510564, 0.5239338697809935], [0.8716339747886043, 0.15619128280552452, 0.696364090325654, 0.9177191991157297, 0.8319987309612937, 0.6456922780217599, 0.0051598721876253695, 0.996842436437123, 0.3182856521838028, 0.8546561923347852, 0.7412763151762193, 0.9401686317712096, 0.36351240648005934, 0.19779569082701776, 0.07838004566940249], [0.21871681838065316, 0.1443851702929393, 0.8760280529266938, 0.20156544391380027, 0.7421484862386746, 0.7396888488541408, 0.16116395551806573, 0.6151552938502067, 0.03383778580119223, 0.2358291907185941, 0.9134818918058208, 0.4294297651978697, 0.924181459441245, 0.5022378947546713, 0.11232608036783698], [0.2673464837012276, 0.5262049450702206, 0.7392235589347469, 0.851905902689529, 0.9746743558118668, 0.006696941140641455, 0.3188906890585548, 0.28094286537787727, 0.6286190067832551, 0.06364809988839248, 0.6157225730943313, 0.11473841416831887, 0.043412619143316844, 0.6564290377768516, 0.23057454605073546], [0.06082974596905577, 0.8209651079227064, 0.32354366372875765, 0.23655581542207882, 0.5498465317493932, 0.8963013048400404, 0.11499287599532104, 0.9278633830642053, 0.5436122355425341, 0.708180957603159, 0.611711702308402, 0.11722147009730799, 0.097451720806632, 0.24477251120909282, 0.666813025607387], [0.7447693698040048, 0.11824032753192282, 0.031079339271260276, 0.6032056767976962, 0.5577027156069387, 0.0514968664793628, 2.7178981587550055e-07, 0.9899787550223412, 0.06950455667920963, 0.5407253259697238, 0.016012329026088068, 0.8058651414640251, 0.5034640120469774, 0.6004082603598118, 0.5654985808986618], [0.14042239631498998, 0.9668672933359546, 0.6517295299579315, 0.9349315965060094, 0.7956986179507451, 0.26703503211245827, 0.48455808504813813, 0.26162034586658767, 0.06021224147665072, 0.9281316184508571, 0.10328419443269332, 0.0396323870896379, 0.12901768076890507, 0.6109019069183127, 0.005204347791102348], [0.5830603937562561, 0.2585343972762071, 0.42987009353654504, 0.8926537007131609, 0.07323831065343378, 0.8161549512968289, 0.5043600505962732, 0.5710669141727454, 0.8668790985603594, 0.8380374696116155, 0.8991866659372136, 0.46789286975752453, 0.5916849488249912, 0.575857842033437, 0.9571812516914769]]
    SamplePeriod = 1.8678501456532843
    Band = [2.0861119360774208e+307, 1.358275705209553e+308]
    out = rest_filter.rest_IdealFilter(np.asarray(Data), SamplePeriod, Band)
    out_exp = [[(-0.08787186612928743+0j), (0.09056763157213091+0j), (-0.004036146869997932+0j), (-0.1488876452967051+0j), (0.04202438792235222+0j), (0.054374042202578465+0j), (0.06158163253393907+0j), (-0.0752785689675059+0j), (-0.05099890258012422+0j), (0.015924321021445184+0j), (0.03720716743198165+0j), (-0.0721410653170807+0j), (0.02884898298746091+0j), (0.03513109629646715+0j), (-0.08393813444803436+0j)], [(0.08787186612928743+0j), (-0.09056763157213091+0j), (0.004036146869997932+0j), (0.1488876452967051+0j), (-0.04202438792235222+0j), (-0.054374042202578465+0j), (-0.06158163253393907+0j), (0.0752785689675059+0j), (0.05099890258012422+0j), (-0.015924321021445184+0j), (-0.03720716743198165+0j), (0.0721410653170807+0j), (-0.02884898298746091+0j), (-0.03513109629646715+0j), (0.08393813444803436+0j)], [(-0.08787186612928743+0j), (0.09056763157213091+0j), (-0.004036146869997932+0j), (-0.1488876452967051+0j), (0.04202438792235222+0j), (0.054374042202578465+0j), (0.06158163253393907+0j), (-0.0752785689675059+0j), (-0.05099890258012422+0j), (0.015924321021445184+0j), (0.03720716743198165+0j), (-0.0721410653170807+0j), (0.02884898298746091+0j), (0.03513109629646715+0j), (-0.08393813444803436+0j)], [(0.08787186612928743+0j), (-0.09056763157213091+0j), (0.004036146869997932+0j), (0.1488876452967051+0j), (-0.04202438792235222+0j), (-0.054374042202578465+0j), (-0.06158163253393907+0j), (0.0752785689675059+0j), (0.05099890258012422+0j), (-0.015924321021445184+0j), (-0.03720716743198165+0j), (0.0721410653170807+0j), (-0.02884898298746091+0j), (-0.03513109629646715+0j), (0.08393813444803436+0j)], [(-0.08787186612928743+0j), (0.09056763157213091+0j), (-0.004036146869997932+0j), (-0.1488876452967051+0j), (0.04202438792235222+0j), (0.054374042202578465+0j), (0.06158163253393907+0j), (-0.0752785689675059+0j), (-0.05099890258012422+0j), (0.015924321021445184+0j), (0.03720716743198165+0j), (-0.0721410653170807+0j), (0.02884898298746091+0j), (0.03513109629646715+0j), (-0.08393813444803436+0j)], [(0.08787186612928743+0j), (-0.09056763157213091+0j), (0.004036146869997932+0j), (0.1488876452967051+0j), (-0.04202438792235222+0j), (-0.054374042202578465+0j), (-0.06158163253393907+0j), (0.0752785689675059+0j), (0.05099890258012422+0j), (-0.015924321021445184+0j), (-0.03720716743198165+0j), (0.0721410653170807+0j), (-0.02884898298746091+0j), (-0.03513109629646715+0j), (0.08393813444803436+0j)], [(-0.08787186612928743+0j), (0.09056763157213091+0j), (-0.004036146869997932+0j), (-0.1488876452967051+0j), (0.04202438792235222+0j), (0.054374042202578465+0j), (0.06158163253393907+0j), (-0.0752785689675059+0j), (-0.05099890258012422+0j), (0.015924321021445184+0j), (0.03720716743198165+0j), (-0.0721410653170807+0j), (0.02884898298746091+0j), (0.03513109629646715+0j), (-0.08393813444803436+0j)], [(0.08787186612928743+0j), (-0.09056763157213091+0j), (0.004036146869997932+0j), (0.1488876452967051+0j), (-0.04202438792235222+0j), (-0.054374042202578465+0j), (-0.06158163253393907+0j), (0.0752785689675059+0j), (0.05099890258012422+0j), (-0.015924321021445184+0j), (-0.03720716743198165+0j), (0.0721410653170807+0j), (-0.02884898298746091+0j), (-0.03513109629646715+0j), (0.08393813444803436+0j)], [(-0.08787186612928743+0j), (0.09056763157213091+0j), (-0.004036146869997932+0j), (-0.1488876452967051+0j), (0.04202438792235222+0j), (0.054374042202578465+0j), (0.06158163253393907+0j), (-0.0752785689675059+0j), (-0.05099890258012422+0j), (0.015924321021445184+0j), (0.03720716743198165+0j), (-0.0721410653170807+0j), (0.02884898298746091+0j), (0.03513109629646715+0j), (-0.08393813444803436+0j)], [(0.08787186612928743+0j), (-0.09056763157213091+0j), (0.004036146869997932+0j), (0.1488876452967051+0j), (-0.04202438792235222+0j), (-0.054374042202578465+0j), (-0.06158163253393907+0j), (0.0752785689675059+0j), (0.05099890258012422+0j), (-0.015924321021445184+0j), (-0.03720716743198165+0j), (0.0721410653170807+0j), (-0.02884898298746091+0j), (-0.03513109629646715+0j), (0.08393813444803436+0j)]]
    assert np.allclose(out, np.asarray(out_exp))