#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 10:31:31 2020

@author: grat05
"""
import numpy as np
import pandas as pd

class ObjDict():
    pass

class OHaraRudy_INa():
    num_params = 33
    param_bounds = [(-3,3)]*2 + \
                   [(-3,3)]*3 + \
                   [(-3,3)]*2 + \
                   [(-3,3), (-20,20), (-20,20)] + \
                   [(-3,3)]*2 + \
                   [(-20,20)]*2 + \
                   [(-3,3)]*2 + \
                   [(-20,20)] + \
                   [(-3,3)] + [(-20,20)]*2 +\
                   [(-3,3)]*2 + \
                   [(-1,1)]*3 + \
                   [(-1,1)]*2 + \
                   [(-1,1)]*2 + \
                   [(-15,15)]*2 + \
                   [(-15,15), (-3,3)]    
                               
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
                 mss_shiftFactor=0, hss_shiftFactor=0,\
                 jss_shiftFactor=0, jss_tauFactor=0,
                 TEMP = 310.0, naO = 140.0, naI = 7):

        # scaling currents 0 
        self.GNa = 75*np.exp(GNaFactor);
        self.GNaL = 0.0075*np.exp(GNaLFactor);

        #m gate 2
        self.mss_tau = 9.871*np.exp(mss_tauFactor)

        self.tm_mult1 = 6.765*np.exp(tm_mult1Factor)
        self.tm_tau1 = 34.77*np.exp(tm_tau1Factor)
        self.tm_mult2 = 8.552*np.exp(tm_mult2Factor)
        self.tm_tau2 = 5.955*np.exp(tm_tau2Factor)

        #h gate 7
        self.hss_tau = 6.086*np.exp(hss_tauFactor)

        self.thf_mult1 = 1.432e-5*np.exp(thf_mult1Factor)
        self.thf_tau1 = 6.285*np.exp(thf_tau1Factor)
        self.thf_mult2 = 6.149*np.exp(thf_mult2Factor)
        self.thf_tau2 = 20.27*np.exp(thf_tau2Factor)
#12
        self.ths_mult1 = 0.009794*np.exp(ths_mult1Factor)
        self.ths_tau1 = 28.05*np.exp(ths_tau1Factor)
        self.ths_mult2 = 0.3343*np.exp(ths_mult2Factor)
        self.ths_tau2 = 56.66*np.exp(ths_tau2Factor)

        mixingodds = np.exp(Ahf_multFactor+np.log(99))
        self.Ahf_mult = mixingodds/(mixingodds+1)# np.exp(Ahf_multFactor)

        #j gate 17
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
        
        #added later 29
        self.mss_shift = 39.57+mss_shiftFactor
        self.hss_shift = 82.90+hss_shiftFactor
        self.jss_shift = 82.90+jss_shiftFactor
        self.jss_tau = 6.086*np.exp(jss_tauFactor)

        self.TEMP = TEMP
        self.naO = naO
        self.naI = naI

        self.recArrayNames = pd.Index(["m","hf","hs","j","hsp","jp","mL","hL","hLp"])
        self.state_vals = pd.Series([0,1,1,1,1,1,0,1,1], index=self.recArrayNames, dtype='float64')
        

        self.recArray = pd.DataFrame(columns=self.recArrayNames)
        
        self.retOptions = {'G': True, 'INa': True, 'INaL': True,\
                                 'Open': True, 'RevPot': True}
            

    def calc_taus_ss(self, vOld):
        tau = ObjDict()
        ss = ObjDict()
        
        ss.mss = 1.0 / (1.0 + np.exp((-(vOld + self.mss_shift)) / self.mss_tau));
        tau.tm = 1.0 / (self.tm_mult1 * np.exp((vOld + 11.64) / self.tm_tau1) +
                           self.tm_mult2 * np.exp(-(vOld + 77.42) / self.tm_tau2));

        ss.hss = 1.0 / (1 + np.exp((vOld + self.hss_shift) / self.hss_tau));
        tau.thf = 1.0 / (self.thf_mult1 * np.exp(-(vOld + 1.196) / self.thf_tau1) +
                            self.thf_mult2 * np.exp((vOld + 0.5096) / self.thf_tau2));
        tau.ths = 1.0 / (self.ths_mult1 * np.exp(-(vOld + 17.95) / self.ths_tau1) +
                            self.ths_mult2 * np.exp((vOld + 5.730) / self.ths_tau2));


        ss.jss = ss.hss#1.0 / (1 + np.exp((vOld + self.jss_shift) / self.jss_tau));#hss;
        tau.tj = self.tj_baseline + 1.0 / (self.tj_mult1 * np.exp(-(vOld + 100.6) / self.tj_tau1) +
                                   self.tj_mult2 * np.exp((vOld + 0.9941) / self.tj_tau2));

        ss.hssp = 1.0 / (1 + np.exp((vOld + 89.1) / self.hssp_tau));
        tau.thsp = self.tssp_mult * tau.ths;


        tau.tjp = self.tjp_mult * tau.tj;

        ss.mLss = 1.0 / (1.0 + np.exp((-(vOld + 42.85)) / self.mLss_tau));
        tau.tmL = tau.tm;

        ss.hLss = 1.0 / (1.0 + np.exp((vOld + 87.61) / self.hLss_tau));
        tau.thL = self.thL_baseline;

        ss.hLssp = 1.0 / (1.0 + np.exp((vOld + 93.81) / self.hLssp_tau));
        tau.thLp = self.thLp_mult * tau.thL;

        return tau, ss
    
    def ddycalc(self, vOld):
        d_vals = np.zeros(9)
        
        tau, _ = self.calc_taus_ss(vOld)
        
        d_vals[0] = -1 / tau.tm

        d_vals[1] = -1  / tau.thf
        d_vals[2] = -1  / tau.ths

        d_vals[3] = -1  / tau.tj

        d_vals[4] = -1  / tau.thsp


        d_vals[5] = -1  / tau.tjp

        d_vals[6] = -1  / tau.tmL

        d_vals[7] = -1  / tau.thL

        d_vals[8] = -1  / tau.thLp

        if np.isinf(d_vals).any():
            print("error")
        return d_vals

    def ddtcalc(self, vals, vOld):
        d_vals = np.zeros_like(vals)
        
        tau, ss = self.calc_taus_ss(vOld)
        
        d_vals[0] = (ss.mss-vals[0]) / tau.tm

        d_vals[1] = (ss.hss-vals[1]) / tau.thf
        d_vals[2] = (ss.hss-vals[2]) / tau.ths

        d_vals[3] = (ss.jss-vals[3]) / tau.tj

        d_vals[4] = (ss.hssp-vals[4]) / tau.thsp


        d_vals[5] = (ss.jss-vals[5]) / tau.tjp

        d_vals[6] = (ss.mLss-vals[6]) / tau.tmL

        d_vals[7] = (ss.hLss-vals[7]) / tau.thL

        d_vals[8] = (ss.hLssp-vals[8]) / tau.thLp

        if np.isinf(d_vals).any():
            print("error")
        return d_vals
    
    def getRevPot(self):
        return (self.RGAS * self.TEMP / self.FDAY) * np.log(self.naO / self.naI)

    def calcCurrent(self, vals, vOld, ret=[True]*3, setRecArray=True):
        vals = np.array(vals)
        if len(vals.shape) == 1:
            vals.shape = (9,-1)
        elif vals.shape[0] != 9 and vals.shape[-1] == 9:
            vals = vals.T
            
        m,hf,hs,j,hsp,jp,mL,hL,hLp = vals
            
        self.recArray = self.recArray.append(pd.DataFrame(vals.T, columns=self.recArrayNames))
        
        ena = self.getRevPot()
        
        Ahf = 0.99*self.Ahf_mult;
        Ahs = 1.0 - Ahf;
        
        h = Ahf *hf + Ahs *j#Ahf * hf + Ahs * hs; #
        hp = Ahf * hf + Ahs *hsp;
        
        fINap = (1.0 / (1.0 + self.KmCaMK / self.CaMKa));
#        oprob = self.m * self.m * self.m * ((1.0 - fINap) * h * self.j + fINap * hp * self.jp)

#        oprob = m**3 * ((1.0 - fINap) *  h  + fINap * hp * jp)
        oprob = m**3 * ((1.0 - fINap) *  h * j + fINap * hp * jp)

        
        INa = (self.GNa if self.retOptions['G'] else 1) *\
                (oprob if self.retOptions['Open'] else 1) *\
                ((vOld - ena) if self.retOptions['RevPot'] else 1);
        
        fINaLp = (1.0 / (1.0 + self.KmCaMK / self.CaMKa));
        loprob = mL * ((1.0 - fINaLp) * hL + fINaLp * hLp)

        INaL = (self.GNaL if self.retOptions['G'] else 1)*\
            (loprob if self.retOptions['Open'] else 1)*\
            ((vOld - ena) if self.retOptions['RevPot'] else 1);
        
        return (INa if self.retOptions['INa'] else 0)+\
            (INaL if self.retOptions['INaL'] else 0)

    def update(self, vOld, dt):
        ddt_vals = self.ddtcalc(self.state_vals, vOld)
        self.state_vals += dt*ddt_vals
        return self.calcCurrent(self.state_vals, vOld)
        
        
    # def update(self, vOld, dt, ret=[True]*3):
    #     ena = self.getRevPot()

    #     mss = 1.0 / (1.0 + np.exp((-(vOld + 39.57)) / self.mss_tau));
    #     tm = 1.0 / (self.tm_mult1 * np.exp((vOld + 11.64) / self.tm_tau1) +
    #                        self.tm_mult2 * np.exp(-(vOld + 77.42) / self.tm_tau2));

    #     self.m = mss - (mss - self.m) * np.exp(-dt / tm);

    #     hss = 1.0 / (1 + np.exp((vOld + 82.90) / self.hss_tau));
    #     thf = 1.0 / (self.thf_mult1 * np.exp(-(vOld + 1.196) / self.thf_tau1) +
    #                         self.thf_mult2 * np.exp((vOld + 0.5096) / self.thf_tau2));
    #     ths = 1.0 / (self.ths_mult1 * np.exp(-(vOld + 17.95) / self.ths_tau1) +
    #                         self.ths_mult2 * np.exp((vOld + 5.730) / self.ths_tau2));
    #     Ahf = 0.99*self.Ahf_mult;
    #     Ahs = 1.0 - Ahf;
    #     self.hf = hss - (hss - self.hf) * np.exp(-dt / thf);
    #     self.hs = hss - (hss - self.hs) * np.exp(-dt / ths);
    #     h = Ahf * self.hf + Ahs * self.hs;

    #     jss = hss;
    #     tj = self.tj_baseline + 1.0 / (self.tj_mult1 * np.exp(-(vOld + 100.6) / self.tj_tau1) +
    #                                self.tj_mult2 * np.exp((vOld + 0.9941) / self.tj_tau2));
    #     self.j = jss - (jss - self.j) * np.exp(-dt / tj);
    #     hssp = 1.0 / (1 + np.exp((vOld + 89.1) / self.hssp_tau));
    #     thsp = self.tssp_mult * ths;

    #     self.hsp = hssp - (hssp - self.hsp) * np.exp(-dt / thsp);
    #     hp = Ahf * self.hf + Ahs * self.hsp;
    #     tjp = self.tjp_mult * tj;

    #     self.jp = jss - (jss - self.jp) * np.exp(-dt / tjp);

    #     fINap = (1.0 / (1.0 + self.KmCaMK / self.CaMKa));
    #     oprob = self.m * self.m * self.m * ((1.0 - fINap) * h * self.j + fINap * hp * self.jp)
    #     INa = np.prod(np.array([self.GNa, oprob, (vOld - ena)])[ret]);

    #     mLss = 1.0 / (1.0 + np.exp((-(vOld + 42.85)) / self.mLss_tau));
    #     tmL = tm;
    #     self.mL = mLss - (mLss - self.mL) * np.exp(-dt / tmL);
    #     hLss = 1.0 / (1.0 + np.exp((vOld + 87.61) / self.hLss_tau));
    #     thL = self.thL_baseline;
    #     self.hL = hLss - (hLss - self.hL) * np.exp(-dt / thL);
    #     hLssp = 1.0 / (1.0 + np.exp((vOld + 93.81) / self.hLssp_tau));
    #     thLp = self.thLp_mult * thL;
    #     self.hLp = hLssp - (hLssp - self.hLp) * np.exp(-dt / thLp);


    #     fINaLp = (1.0 / (1.0 + self.KmCaMK / self.CaMKa));
    #     loprob = self.mL * ((1.0 - fINaLp) * self.hL + fINaLp * self.hLp)

    #     INaL = np.prod(np.array([self.GNaL, loprob, (vOld - ena)])[ret]);

    #     self.recArray.append([self.m, self.hf, self.hs, self.j, self.hsp,\
    #                           self.jp, self.mL, self.hL, self.hLp])

    #     return INa+INaL

