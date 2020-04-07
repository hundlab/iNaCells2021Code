2#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 10:10:44 2020

@author: grat05
"""

import numpy as np

exp = np.exp
log = np.log

class obj():
    pass

class HRD09_Ina:
    RGAS = 8314.0;
    FDAY = 96487.0;
    Gate = obj()
    Gate.m = 0.00106712815
    Gate.h = 0.9907975703
    Gate.j = 0.9939823939
    Gate.ml = 0.0008371146773
    Gate.hl = 0.3497149946
    caM = 0.0003647207545

    def __init__(self, TEMP = 310.0, naO = 140.0, naI = 9.889359792):
        self.TEMP = TEMP
        self.naO = naO
        self.naI = naI


        self.ammult1 = .32
        self.bmmult1 = np.log(.08)
        self.bmtau1 = 11.0

        self.bhmult1 = .13
        self.bhtau1 = -11.1
        self.ahmult1 = .135
        self.ahtau1 = -6.8
        self.bhmult2 = 3.56
        self.bhtau2 = 1/.079
        self.bhmult3 = 3.1E5
        self.bhtau3 = 1/.35

        self.bjmult1 = .3
        self.bjtau1 = 1/2.535E-7
        self.bjmult2 = -.1
        self.bjtau2 = 1     #not in HRd model
        self.ajmult1 = -1.2714E5
        self.ajtau1 = 1/.2444
        self.ajmult2 = 3.474E-5
        self.ajtau2 = 1/.04391
        self.ajmult3
        self.ajtau3


        self.vshiftbaseline = -3.25

        self.MaxGNa = 16.0;  # different from HRd08 - 8.25;

    def updateIna(self, vOld, dt):
        KMCAM = 0.3;
        deltaalpha = -.18;

        camfact = 1 / (1 + pow((KMCAM / self.caM), 4.0));
        vshift = self.vshiftbaseline * camfact;

        ENa = (self.RGAS * self.TEMP / self.FDAY) * log(self.naO / self.naI);

        Rate = obj()
        Rate.am = self.ammult1 * (vOld + 47.13) / (1 - exp(-.1 * (vOld + 47.13)));
        Rate.bm = exp(-(vOld / self.bmtau1) + self.bmmult1);

        if ((vOld - vshift) >= -40.0):
            Rate.ah = 0.0;
            Rate.bh = 1 / (self.bhmult1 * (1 + exp((vOld - vshift + 10.66) / (self.bhtau1))));
        else:
            Rate.ah = self.ahmult1 * exp((80.0 + vOld - vshift) / (self.ahtau1));
            Rate.bh = self.bhmult2 * exp((vOld - vshift)/self.bhtau2) + (self.bhmult3) * exp((vOld - vshift)/self.bhtau3);

        if ((vOld - vshift) >= -40.0):
            Rate.aj = 0.0;

            bj1a =  (self.bjmult1) * exp((vOld - vshift)/self.bjtau1); #why is bjtau1 so large?
            Rate.bj = bj1a / (exp(self.bjmult2 * (vOld - vshift + 32)/self.bjtau2) + 1); #this may need reformulated

        else:
            aj1a = (self.ajmult1) * exp((vOld - vshift)/self.ajtau1);
            aj1b = (self.ajmult2) * exp(-(vOld - vshift)/self.ajtau2);
            aj1c = (vOld - vshift + 37.78) / (1 + exp(.311 * (vOld - vshift + 79.23)));

            Rate.aj = (1 + camfact * deltaalpha) * (aj1a - aj1b) * aj1c;

            bj2a = .1212 * exp(-.01052 * (vOld - vshift));
            bj2b = 1 + exp(-.1378 * (vOld - vshift + 40.14));
            Rate.bj = bj2a / bj2b;

        ms = Rate.am / (Rate.am + Rate.bm);
        tm = 1 / (Rate.am + Rate.bm);
        self.Gate.m = ms - (ms - self.Gate.m) * exp(-dt / tm);

        hs = Rate.ah / (Rate.ah + Rate.bh);
        th = 1 / (Rate.ah + Rate.bh);
        self.Gate.h = hs - (hs - self.Gate.h) * exp(-dt / th);

        js = Rate.aj / (Rate.aj + Rate.bj);
        tj = 1 / (Rate.aj + Rate.bj);
        self.Gate.j = js - (js - self.Gate.j) * exp(-dt / tj);

        iNa = (self.MaxGNa * self.Gate.m * self.Gate.m * self.Gate.m * self.Gate.h * self.Gate.j) * (vOld - ENa);

        KMCAM = 0.3;
        deltag = 0.0095;

        camfact = 1 / (1 + pow((KMCAM / self.caM), 4.0));

        ENa = (self.RGAS * self.TEMP / self.FDAY) * log(self.naO / self.naI);

        Rate = obj()
        Rate.aml = .32 * (vOld + 47.13) / (1 - exp(-.1 * (vOld + 47.13)));
        Rate.bml = .08 * exp(-vOld / 11.0);

        hlinf = 1 / (1 + exp((vOld + 91) / 6.1));

        ms = Rate.aml / (Rate.aml + Rate.bml);
        tml = 1 / (Rate.aml + Rate.bml);
        self.Gate.ml = ms - (ms - self.Gate.ml) * exp(-dt / tml);

        thl = 600;
        self.Gate.hl = hlinf - (hlinf - self.Gate.hl) * exp(-dt / thl);

        iNal = (0.0065 + camfact * deltag) * self.Gate.ml * self.Gate.ml *self.Gate.ml * self.Gate.hl * (vOld - ENa);
        return iNa+iNal

class OHaraRudy_INa():
    num_params = 29
    param_bounds = [(-5,5)]*29
    RGAS = 8314.0;
    FDAY = 96487.0;

    KmCaMK = 0.15
    CaMKa = 1e-5

    def __init__(self, GNaFactor=0, GNaLFactor=0, \
                 mss_tauFactor=0, tm_mult1Factor=0, tm_tau1Factor=0,\
                 tm_mult2Factor=0, tm_tau2Factor=0,\
                 hss_tauFactor=0, thf_mult1Factor=0, thf_tau1Factor=0,\
                 thf_mult2Factor=0, thf_tau2Factor=0,\
                 ths_mult1Factor=0, ths_tau1Factor=0,\
                 ths_mult2Factor=0, ths_tau2Factor=0,\
                 Ahf_multFactor=0,\
                 tj_baselineFactor=0, tj_mult1Factor=0, tj_tau1Factor=0,\
                 tj_mult2Factor=0, tj_tau2Factor=0,\
                 hssp_tauFactor=0, tssp_multFactor=0, tjp_multFactor=0,\
                 mLss_tauFactor=0, hLss_tauFactor=0,\
                 thL_baselineFactor=0, thLp_multFactor=0,\
                 TEMP = 310.0, naO = 140.0, naI = 7):

        # scaling currents
        self.GNa = 75*np.exp(GNaFactor);
        self.GNaL = 0.0075*np.exp(GNaLFactor);

        #m gate
        self.mss_tau = 9.871*np.exp(mss_tauFactor)

        self.tm_mult1 = 6.765*np.exp(tm_mult1Factor)
        self.tm_tau1 = 34.77*np.exp(tm_tau1Factor)
        self.tm_mult2 = 8.552*np.exp(tm_mult2Factor)
        self.tm_tau2 = 5.955*np.exp(tm_tau2Factor)

        #h gate
        self.hss_tau = 6.086*np.exp(hss_tauFactor)

        self.thf_mult1 = 1.432e-5*np.exp(thf_mult1Factor)
        self.thf_tau1 = 6.285*np.exp(thf_tau1Factor)
        self.thf_mult2 = 6.149*np.exp(thf_mult2Factor)
        self.thf_tau2 = 20.27*np.exp(thf_tau2Factor)

        self.ths_mult1 = 0.009794*np.exp(ths_mult1Factor)
        self.ths_tau1 = 28.05*np.exp(ths_tau1Factor)
        self.ths_mult2 = 0.3343*np.exp(ths_mult2Factor)
        self.ths_tau2 = 56.66*np.exp(ths_tau2Factor)

        self.Ahf_mult = np.exp(Ahf_multFactor)

        #j gate
        self.tj_baseline = 2.038*np.exp(tj_baselineFactor)
        self.tj_mult1 = 0.02136*np.exp(tj_mult1Factor)
        self.tj_tau1 = 8.281*np.exp(tj_tau1Factor)
        self.tj_mult2 = 0.3052*np.exp(tj_mult2Factor)
        self.tj_tau2 = 38.45*np.exp(tj_tau2Factor)

        # phosphorylated gates
        self.hssp_tau = 6.086*np.exp(hssp_tauFactor)
        self.tssp_mult = 3.0*np.exp(tssp_multFactor)
        self.tjp_mult = 1.46*np.exp(tjp_multFactor)

        #late gates & late gate phosphorylation
        self.mLss_tau = 5.264*np.exp(mLss_tauFactor)
        self.hLss_tau = 7.488*np.exp(hLss_tauFactor)
        self.hLssp_tau = self.hLss_tau

        self.thL_baseline = 200.0*np.exp(thL_baselineFactor)
        self.thLp_mult = 3*np.exp(thLp_multFactor)

        self.TEMP = TEMP
        self.naO = naO
        self.naI = naI

        self.m = 0
        self.hf = 1
        self.hs = 1
        self.j = 1
        self.hsp = 1
        self.jp = 1
        self.mL = 0
        self.hL = 1
        self.hLp = 1

        self.recArray = []

    def update(self, vOld, dt, ret=[True]*3):
        ena = (self.RGAS * self.TEMP / self.FDAY) * log(self.naO / self.naI)

        mss = 1.0 / (1.0 + exp((-(vOld + 39.57)) / self.mss_tau));
        tm = 1.0 / (self.tm_mult1 * exp((vOld + 11.64) / self.tm_tau1) +
                           self.tm_mult2 * exp(-(vOld + 77.42) / self.tm_tau2));
        self.m = mss - (mss - self.m) * exp(-dt / tm);

        hss = 1.0 / (1 + exp((vOld + 82.90) / self.hss_tau));
        thf = 1.0 / (self.thf_mult1 * exp(-(vOld + 1.196) / self.thf_tau1) +
                            self.thf_mult2 * exp((vOld + 0.5096) / self.thf_tau2));
        ths = 1.0 / (self.ths_mult1 * exp(-(vOld + 17.95) / self.ths_tau1) +
                            self.ths_mult2 * exp((vOld + 5.730) / self.ths_tau2));
        Ahf = 0.99*self.Ahf_mult;
        Ahs = 1.0 - Ahf;
        self.hf = hss - (hss - self.hf) * exp(-dt / thf);
        self.hs = hss - (hss - self.hs) * exp(-dt / ths);
        h = Ahf * self.hf + Ahs * self.hs;

        jss = hss;
        tj = self.tj_baseline + 1.0 / (self.tj_mult1 * exp(-(vOld + 100.6) / self.tj_tau1) +
                                   self.tj_mult2 * exp((vOld + 0.9941) / self.tj_tau2));
        self.j = jss - (jss - self.j) * exp(-dt / tj);
        hssp = 1.0 / (1 + exp((vOld + 89.1) / self.hssp_tau));
        thsp = self.tssp_mult * ths;

        self.hsp = hssp - (hssp - self.hsp) * exp(-dt / thsp);
        hp = Ahf * self.hf + Ahs * self.hsp;
        tjp = self.tjp_mult * tj;

        self.jp = jss - (jss - self.jp) * exp(-dt / tjp);

        fINap = (1.0 / (1.0 + self.KmCaMK / self.CaMKa));
        oprob = self.m * self.m * self.m * ((1.0 - fINap) * h * self.j + fINap * hp * self.jp)
        INa = np.prod(np.array([self.GNa, oprob, (vOld - ena)])[ret]);

        mLss = 1.0 / (1.0 + exp((-(vOld + 42.85)) / self.mLss_tau));
        tmL = tm;
        self.mL = mLss - (mLss - self.mL) * exp(-dt / tmL);
        hLss = 1.0 / (1.0 + exp((vOld + 87.61) / self.hLss_tau));
        thL = self.thL_baseline;
        self.hL = hLss - (hLss - self.hL) * exp(-dt / thL);
        hLssp = 1.0 / (1.0 + exp((vOld + 93.81) / self.hLssp_tau));
        thLp = self.thLp_mult * thL;
        self.hLp = hLssp - (hLssp - self.hLp) * exp(-dt / thLp);


        fINaLp = (1.0 / (1.0 + self.KmCaMK / self.CaMKa));
        loprob = self.mL * ((1.0 - fINaLp) * self.hL + fINaLp * self.hLp)

        INaL = np.prod(np.array([self.GNaL, loprob, (vOld - ena)])[ret]);

        self.recArray.append([self.m, self.hf, self.hs, self.j, self.hsp,\
                              self.jp, self.mL, self.hL, self.hLp])

        return INa+INaL


class Grandi_Ina():
    RGAS = 8314.0;
    TEMP = 310.0;
    FDAY = 96487.0;
    gate = obj()
    gate.m = 0.001405627
    gate.h = .9867005
    gate.j = .991562

    def updateIna(self, vOld, dt, naO = 140.0, naI = 9.06):
        gna = 23.0;  # mS/uF
        Ena = self.RGAS * self.TEMP / self.FDAY * log(naO / naI);

        m_inf = 1 / ((1 + exp(-(56.86 + vOld) / 9.03))**2);
        tau_m = 0.1292 * exp(-((vOld + 45.79) / 15.54) **2) + 0.06487 * exp(-((vOld - 4.823) / 51.12)**2);

        if (vOld >= -40.0):
            alpha_h = 0.0;
            beta_h = (0.77 / (0.13 * (1 + exp(-(vOld + 10.66) / 11.1))));
        else:
            alpha_h = (0.057 * exp(-(vOld + 80) / 6.8));
            beta_h = ((2.7 * exp(0.079 * vOld) + 3.1E5 * exp(0.3485 * vOld)));

        h_inf =  1 / ((1 + exp((vOld + 71.55) / 7.43)) **2);
        tau_h = 1.0 / (alpha_h + beta_h);

        if (vOld >= -40.0):
            alpha_j = 0.0;
            beta_j = ((0.6 * exp(0.057 * vOld)) / (1 + exp(-0.1 * (vOld + 32))));
        else:
            alpha_j = \
              (((-2.5428E4 * exp(0.2444 * vOld) - 6.948E-6 * exp(-0.04391 * vOld)) * \
                (vOld + 37.78)) /
               (1 + exp(0.311 * (vOld + 79.23))));
            beta_j = ((0.02424 * exp(-0.01052 * vOld)) /
                    (1 + exp(-0.1378 * (vOld + 40.14))));

        tau_j = 1 / (alpha_j + beta_j);
        j_inf = 1 / ((1 + exp((vOld + 71.55) / 7.43)) **2);

        self.gate.m = m_inf - (m_inf - self.gate.m) * exp(-dt / tau_m);
        self.gate.h = h_inf - (h_inf - self.gate.h) * exp(-dt / tau_h);
        self.gate.j = j_inf - (j_inf - self.gate.j) * exp(-dt / tau_j);

        iNa = gna * self.gate.m * self.gate.m * self.gate.m * self.gate.h *\
                  self.gate.j * (vOld - Ena);

        return iNa

#10.1161/CIRCULATIONAHA.112.105320 P glynn
class Koval_ina:
    num_params = 22

    def __init__(self, P1a1Factor=1, P2a1Factor=1, P1a4Factor=1, P1a5Factor=1, P2a5Factor=1, P1b1Factor=1, P2b1Factor=1, P1b2Factor=1, P2b2Factor=1, P1b3Factor=1, P2b3Factor=1, P1b5Factor=1, P2b5Factor=1, P1a6Factor=1, P1b6Factor=1, P1a7Factor=1, P1b7Factor=1, P1a8Factor=1, P1b8Factor=1, P1a9Factor=1, P1b9Factor=1, gNaFactor = 1, TEMP = 310.0, naO = 140.0, naI = 8.35504003):

        self.P1a1 = P1a1Factor*7.5207
        self.P2a1 = P2a1Factor*0.1027
        self.P1a4 = P1a4Factor*0.188495
        self.P1a5 = P1a5Factor*7.0e-7
        self.P2a5 = P2a5Factor*7.7
        self.P1b1 = P1b1Factor*0.1917
        self.P2b1 = P2b1Factor*20.3
        self.P1b2 = P1b2Factor*0.2
        self.P2b2 = P2b2Factor*2.5
        self.P1b3 = P1b3Factor*0.22
        self.P2b3 = P2b3Factor*7.5
        self.P1b5 = P1b5Factor*0.0108469
        self.P2b5 = P2b5Factor*2e-5
        self.P1a6 = P1a6Factor*1000.0
        self.P1b6 = P1b6Factor*6.0448e-3
        self.P1a7 = P1a7Factor*1.05263e-5
        self.P1b7 = P1b7Factor*0.02
        self.P1a8 = P1a8Factor*4.0933e-13
        self.P1b8 = P1b8Factor*9.5e-4
        self.P1a9 = P1a9Factor*8.2
        self.P1b9 = P1b9Factor*0.022
        self.gNa  = gNaFactor#7.35

        self.C1 = 0.0003850597267
        self.C2 = 0.02639207662
        self.C3 = 0.7015088787
        self.IC2 = 0.009845083654
        self.IC3 = 0.2616851145
        self.IF = 0.0001436395221
        self.IM1 = 3.913769904e-05
        self.IM2 = 3.381242427e-08
        self.LC1 = 1.659002962e-13
        self.LC2 = 1.137084204e-11
        self.LC3 = 3.02240205e-10
        self.O = 9.754706096e-07
        self.LO = 4.202747013e-16
        self.OB = 0
        self.LOB = 0

        self.RGAS = 8314.4
        self.TEMP = TEMP
        self.FDAY = 96485
        self.RanolazineConc = 0
        self.naO = naO
        self.naI = naI

        self.valid = True
        self.recurseCounter = 0

        self.recArray = []


    def updateIna(self, Vm, dt, ret=[True]*3):
        Vm_rel = Vm
        vm_shift = 0#15
        Vm = Vm - vm_shift

        if not self.valid or dt < 0.0001:
            self.valid = False
            return np.nan

        a1 = self.P1a1/((self.P2a1*exp(-(Vm+2.5)/17)) + 0.20*exp(-(Vm+2.5)/150))
        a2 = self.P1a1/((self.P2a1*exp(-(Vm+2.5)/15)) + 0.23*exp(-(Vm+2.5)/150))
        a3 = self.P1a1/((self.P2a1*exp(-(Vm+2.5)/12)) + 0.25*exp(-(Vm+2.5)/150))
        a4 = 1.0/(self.P1a4*exp(-(Vm+7.0)/16.6) + 0.393956)
        a5 = self.P1a5*exp(-(Vm+7)/self.P2a5) #self.P1a5*exp(-Vm/self.P2a5)
        a6 = a4/self.P1a6
        a7 = self.P1a7*a4
        a8 = self.P1a8
        b1 = self.P1b1*exp(-(Vm+2.5)/self.P2b1)
        b2 = self.P1b2*exp(-(Vm-self.P2b2)/self.P2b1)
        b3 = self.P1b3*exp(-(Vm-self.P2b3)/self.P2b1)
        b5 = self.P1b5 + self.P2b5*(Vm+7.0)
        b4 = (a3*a4*a5)/(b3*b5)
        b6 = self.P1b6*a5
        b7 = self.P1b7*a5
        b8 = self.P1b8
        a9 = self.RanolazineConc*self.P1a9
        b9 = self.P1b9

        def dStates(C1, C2, C3, IC2, IC3, IF, IM1, IM2, LC1, LC2, LC3, O, LO, OB, LOB):
            dC1  = dt*(a5*self.IF+b3*self.O+b8*self.LC1+a2*self.C2   -(b5+a3+a8+b2)*self.C1)
            dC2  = dt*(a5*self.IC2+b2*self.C1+b8*self.LC2+a1*self.C3 -(b5+a2+a8+b1)*self.C2)
            dC3  = dt*(a5*self.IC3+b1*self.C2+b8*self.LC3       -(b5+a1+a8)*self.C3)
            dIC2 = dt*(b2*self.IF+b5*self.C2+a1*self.IC3        -(a2+a5+b1)*self.IC2)
            dIC3 = dt*(b1*self.IC2+b5*self.C3               -(a1+a5)*self.IC3)
            dIF  = dt*(b6*self.IM1+a4*self.O+b5*self.C1+a2*self.IC2  -(a6+b4+a5+b2)*self.IF)
            dIM1 = dt*(b7*self.IM2+a6*self.IF              -(a7+b6)*self.IM1)
            dIM2 = dt*(a7*self.IM1                    -(b7)*self.IM2)
            dLC1 = dt*(a8*self.C1+b3*self.LO+a2*self.LC2        -(b8+a3+b2)*self.LC1)
            dLC2 = dt*(a8*self.C2+b2*self.LC1+a1*self.LC3       -(b8+a2+b1)*self.LC2)
            dLC3 = dt*(b1*self.LC2+a8*self.C3              -(a1+b8)*self.LC3)
            dO   = dt*(b9*self.OB+b8*self.LO+a3*self.C1+b4*self.IF   -(a9+a8+b3+a4)*self.O)
            dLO  = dt*(a8*self.O+b9*self.LOB+a3*self.LC1        -(b8+a9+b3)*self.LO)
            dOB  = dt*(a9*self.O                      -(b9)*self.OB)
            dLOB = dt*(a9*self.LO                     -(b9)*self.LOB)
            return dC1, dC2, dC3, dIC2, dIC3, dIF, dIM1, dIM2, dLC1, dLC2, dLC3, dO, dLO, dOB, dLOB

        def RungeKutta4(df, states):
            k1 = np.array(df(*states))
            k2 = np.array(df(*(states+k1/2)))
            k3 = np.array(df(*(states+k2/2)))
            k4 = np.array(df(*(states+k3)))
            return 1./6*(k1+2*k2+2*k3+k4)

#        dC1,dC2,dC3,dIC2,dIC3,dIF,dIM1,dIM2,dLC1,dLC2,dLC3,dO,dLO,dOB,dLOB = \
#        RungeKutta4(dStates, np.array([self.C1, self.C2, self.C3, self.IC2, \
#                                       self.IC3, self.IF, self.IM1, self.IM2, \
#                                       self.LC1, self.LC2, self.LC3, self.O, \
#                                       self.LO, self.OB, self.LOB]))

        dC1,dC2,dC3,dIC2,dIC3,dIF,dIM1,dIM2,dLC1,dLC2,dLC3,dO,dLO,dOB,dLOB = \
        dStates(self.C1, self.C2, self.C3, self.IC2, \
                                       self.IC3, self.IF, self.IM1, self.IM2, \
                                       self.LC1, self.LC2, self.LC3, self.O, \
                                       self.LO, self.OB, self.LOB)
        if min(self.C1+dC1,self.C2+dC2,self.C3+dC3,self.IC2+dIC2,self.IC3+dIC3,self.IF+dIF,self.IM1+dIM1,self.IM2+dIM2,self.LC1+dLC1,self.LC2+dLC2,self.LC3+dLC3,self.O+dO,self.LO+dLO,self.OB+dOB,self.LOB+dLOB) < 0:
            dtn = dt/2
            self.recurseCounter += 1
            self.updateIna(Vm_rel, dtn, ret=ret)
            self.updateIna(Vm_rel, dtn, ret=ret)
            self.recurseCounter -= 1
            print('timestep reduced: ',dtn)
#            print(np.round([dC1, dC2, dC3, dIC2, dIC3, dIF, dIM1, dIM2, dLC1, dLC2, dLC3, dO, dLO, dOB, dLOB],4))
#            print(np.round([self.C1+dC1,self.C2+dC2,self.C3+dC3,self.IC2+dIC2,self.IC3+dIC3,self.IF+dIF,self.IM1+dIM1,self.IM2+dIM2,self.LC1+dLC1,self.LC2+dLC2,self.LC3+dLC3,self.O+dO,self.LO+dLO,self.OB+dOB,self.LOB+dLOB],4))
        else:
            self.C1 += dC1
            self.C2 += dC2
            self.C3 += dC3
            self.IC2+= dIC2
            self.IC3+= dIC3
            self.IF += dIF
            self.IM1+= dIM1
            self.IM2+= dIM2
            self.LC1+= dLC1
            self.LC2+= dLC2
            self.LC3+= dLC3
            self.O  += dO
            self.LO += dLO
            self.OB += dOB
            self.LOB+= dLOB


        if self.recurseCounter == 0:
            self.recArray.append((self.C1,self.C2,self.C3,self.IC2,self.IC3,self.IF,self.IM1,self.IM2,self.LC1,self.LC2,self.LC3,self.O,self.LO,self.OB,self.LOB))
        ENa = self.RGAS * self.TEMP / self.FDAY * log((self.naO) / (self.naI))
        iNa = np.prod(np.array([self.gNa, (self.O+self.LO), (Vm - ENa)])[ret])

        return iNa














class Koval_ina_2:
    def __init__(self):
        self.RGAS = 8314.472;
        self.TEMP = 310.0;
        self.FDAY=96485.3415;

        self.IC3_ina = self.IC3_mut_ina = 0.0;
        self.IC2_ina = self.IC2_mut_ina = 0.0;
        self.IF_ina = self.IF_mut_ina = 0.0;
        self.IM1_ina = self.IM1_mut_ina = 0.0;
        self.IM2_ina = self.IM2_mut_ina = 0.0;
        self.O_ina = self.O_mut_ina = 0.0;
        self.OB_ina = self.OB_mut_ina = 0.0;
        self.C2_ina = self.C2_mut_ina = 0.0;
        self.C1_ina = self.C1_mut_ina = 0.0;
        self.LC3_ina = self.LC3_mut_ina = 0.0;
        self.LC2_ina = self.LC2_mut_ina = 0.0;
        self.LC1_ina = self.LC1_mut_ina = 0.0;
        self.LO_ina = self.LO_mut_ina = 0.0;
        self.LOB_ina = self.LOB_mut_ina = 0.0;
        self.C3_ina = self.C3_mut_ina = 1.0;


    def updateIna(self, vOld, dt, naO = 140.0, naI = 8.35504003):
#/* Modified Rasmusson 05/04/11 */
         gna = 7.35;
         Ena=(self.RGAS*self.TEMP/self.FDAY)*log(naO/naI);

         drug_conc = 0.0;#10.0E-3;  #mM
         perc_mut = 0.0;#50.0;
         mut_flag = 1;   #1 for A572D, else Q573E

        #parameters changed for mutations
         P1a1 = 7.52067;#7.56939;
         P2a1 = 0.1027;
         P2a1_mut = 0.1027;
         P1a5  = 7.0E-7;
         P1a5_mut = 7.0E-7;
         P2a5 = 7.7;
         P2a5_mut = 7.7;
         P1b1 = 0.1917;
         P1b1_mut = 0.1917;
         P2b1 = 20.3;
         P2b1_mut = 20.3;
         P1b2 = 0.2;
         P1b2_mut = 0.2;
         P2b2 = 2.5;
         P2b2_mut = 2.5;
         P1b3 = 0.22;
         P1b3_mut = 0.22;
         P2b3 = 7.5;
         P2b3_mut = 7.5;
         P1b5 = 0.0108469;
         P2b5 = 2E-5;
         P2b5_mut = 2E-5;
         P1a6 = 1000.0; #100.0
         P1a6_mut = 1000.0; #100.0
         P1b6 = 0.0060448;
         P1a7 = 1.052632E-5;#0.3543E-3;
         P1a7_mut = 1.052632E-5;#0.3543E-3;
         P1b7 = 0.02;
         P1b7_mut = 0.02;
         P1a8 = 4.09327E-13;#8.23895E-17;#4.7E-7;
         P1b8 = 9.5E-4;
         P1b8_mut = 9.5E-4;
         P1a9 = 8.2; #association rate mM-1ms-1
         P1a9_mut = 8.2; #association rate mM-1ms-1
         P1b9 = 0.022; #dissociation rate ms-1
         P1b9_mut = 0.022; #dissociation rate ms-1

#        if(mut_flag==1){
#         mut_GNa = 9.75;
#          P1a1_mut = 6.89197;#4.63051;
#          P1b5_mut = 0.0604095;#0.0453576;
#          P1b6_mut = 0.00251214;#0.0026199;
#          P1a8_mut = 0.000144583;#6.52255E-5;
#       }
#        else{
#         mut_GNa = 11.6;#11.9;
#          P1a1_mut = 4.63051;#6.89197;#4.63051;
#          P1b5_mut = 0.0453576;#0.0604095;#0.0453576;
#          P1b6_mut = 0.0026199;# 0.00251214;#0.0026199;
#          P1a8_mut = 6.52255E-5;#0.000144583;#6.52255E-5;
#       }

         a1 = P1a1/(P2a1*exp(-(vOld+2.5)/17.0)+0.20*exp(-(vOld+2.5)/150.0));
         a2 = P1a1/(P2a1*exp(-(vOld+2.5)/15.0)+0.23*exp(-(vOld+2.5)/150.0));
         a3 = P1a1/(P2a1*exp(-(vOld+2.5)/12.0)+0.25*exp(-(vOld+2.5)/150.0));
         a4 = 1/(0.188495*exp(-(vOld+7.0)/16.6)+0.393956);#(P1a4*exp(vOld/P2a4)); #divide by 2 for mutant
         a5 = P1a5*exp(-(vOld+7.0)/P2a5);
         a6 = a4/P1a6;
         a7 = P1a7*a4;
         #a7 = P1a7*exp(vOld/P2a7);
         a8 = P1a8; #1.0E-7 WT, 0.5E-6 Mutant
         a9 = drug_conc*P1a9;

         b1 = P1b1*exp(-(vOld+2.5)/P2b1);
         b2 = P1b2*exp(-(vOld-P2b2)/P2b1);
         b3 = P1b3*exp(-(vOld-P2b3)/P2b1); #0.17 WT, 0.535 mutant
         b5 = P1b5+P2b5*(vOld+7.0);
         b4 = (a3*a4*a5)/(b3*b5);
         b6 = P1b6*a5;#P1b6*exp(-vOld/P2b6);
         b7 = P1b7*a5;
         #b7 = P1b7*exp(-vOld/P2b7);
         b8 = P1b8; #3.8E-3 WT, 6.0E-4 mutant
         b9 = P1b9;

         dIC3 = dt*(b1*self.IC2_ina+b5*self.C3_ina-a1*self.IC3_ina-a5*self.IC3_ina);
         dIC2 = dt*(a1*self.IC3_ina+b5*self.C2_ina+b2*self.IF_ina-b1*self.IC2_ina-a5*self.IC2_ina-a2*self.IC2_ina);
         dIF = dt*(a2*self.IC2_ina+b5*self.C1_ina+b6*self.IM1_ina+a4*self.O_ina-b2*self.IF_ina-a5*self.IF_ina-b4*self.IF_ina-a6*self.IF_ina);
         dIM1 = dt*(a6*self.IF_ina+b7*self.IM2_ina-b6*self.IM1_ina-a7*self.IM1_ina);
         dIM2 = dt*(a7*self.IM1_ina-b7*self.IM2_ina);
         #dC3 = dt*(a5*self.IC3+b1*self.C2+b8*self.LC3-b5*self.C3-a1*self.C3-a8*self.C3);
         dC2 = dt*(a1*self.C3_ina+a5*self.IC2_ina+b2*self.C1_ina+b8*self.LC2_ina-b1*self.C2_ina-b5*self.C2_ina-a2*self.C2_ina-a8*self.C2_ina);
         dC1 = dt*(a2*self.C2_ina+a5*self.IF_ina+b3*self.O_ina+b8*self.LC1_ina-b2*self.C1_ina-b5*self.C1_ina-a3*self.C1_ina-a8*self.C1_ina);
         dLC3 = dt*(a8*self.C3_ina+b1*self.LC2_ina-b8*self.LC3_ina-a1*self.LC3_ina);
         dLC2 = dt*(a1*self.LC3_ina+a8*self.C2_ina+b2*self.LC1_ina-a2*self.LC2_ina-b8*self.LC2_ina-b1*self.LC2_ina);
         dLC1 = dt*(a2*self.LC2_ina+a8*self.C1_ina+b3*self.LO_ina-a3*self.LC1_ina-b8*self.LC1_ina-b2*self.LC1_ina);
         dLO = dt*(a8*self.O_ina+a3*self.LC1_ina+b9*self.LOB_ina-b3*self.LO_ina-b8*self.LO_ina-a9*self.LO_ina);
         dO = dt*(b4*self.IF_ina+a3*self.C1_ina+b8*self.LO_ina+b9*self.OB_ina-a4*self.O_ina-b3*self.O_ina-a8*self.O_ina-a9*self.O_ina);
         dLOB = dt*(a9*self.LO_ina-b9*self.LOB_ina);
         dOB = dt*(a9*self.O_ina-b9*self.OB_ina);

         self.IC3_ina = self.IC3_ina + dIC3;
         self.IC2_ina = self.IC2_ina + dIC2;
         self.IF_ina = self.IF_ina + dIF;
         self.IM1_ina = self.IM1_ina + dIM1;
         self.IM2_ina = self.IM2_ina + dIM2;
         self.O_ina = self.O_ina + dO;
         self.OB_ina = self.OB_ina + dOB;
         self.C2_ina = self.C2_ina + dC2;
         self.C1_ina = self.C1_ina + dC1;
         self.LC3_ina = self.LC3_ina + dLC3;
         self.LC2_ina = self.LC2_ina + dLC2;
         self.LC1_ina = self.LC1_ina + dLC1;
         self.LO_ina = self.LO_ina + dLO;
         self.LOB_ina = self.LOB_ina + dLOB;

         self.C3_ina = 1.0-self.IC3_ina-self.IC2_ina-self.IF_ina-self.IM1_ina-self.IM2_ina-self.O_ina-self.C2_ina-self.C1_ina-self.LC3_ina-self.LC2_ina-self.LC1_ina-self.LO_ina-self.OB_ina-self.LOB_ina;

         return gna*(self.O_ina+self.LO_ina)*(vOld-Ena);


