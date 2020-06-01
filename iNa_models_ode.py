#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 10:31:31 2020

@author: grat05
"""
import numpy as np
import pandas as pd

class ObjDict():
    def __repr__(self):
        return str(self.__dict__)
    
def isList(thing):
    return isinstance(thing, (list, tuple, np.ndarray))

class OHaraRudy_INa():
    num_params = 33
    param_bounds = [(-3,3)]*2 + \
                   [(-0.1,3)] + [(-3,3)] + [(-0.1,3)] +\
                   [(-3,3)] + [(-0.1,3)] +\
                   [(-1,3), (-3,3), (-0.1,3)] + \
                   [(-3,3)] + [(-0.1,3)] +\
                   [(-3,3)] + [(-1,3)] +\
                   [(-3,3)] + [(-1,3)] +\
                   [(-20,20)] + \
                   [(-3,3)] + [(-3,3)] + [(-1,3)] +\
                   [(-3,3)] + [(-1,3)] +\
                   [(-1,1)]*3 + \
                   [(-1,1)]*2 + \
                   [(-1,1)]*2 + \
                   [(-15,15)]*2 + \
                   [(-15,15), (-1,3)]    
                               
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
        self.lastVal = None            

    
    def calc_taus_ss(self, vOld):
        if self.lastVal is not None and np.array_equal(self.lastVal[0], vOld):
            return self.lastVal[1]
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

        tau.__dict__ = {key: min(max(value, 1e-8), 1e20) for key,value in tau.__dict__.items()}
        
        self.lastVal = (vOld, (tau, ss))
        return tau, ss
    
    def jac(self, vOld):
        vOld = np.array(vOld,ndmin=1)
        d_vals = np.squeeze(np.zeros((9,len(vOld))))
        
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

#        np.clip(d_vals, a_min=-1e15, a_max=None, out=d_vals)

        return np.diag(d_vals)


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

#        np.clip(d_vals, a_min=-1e15, a_max=1e15, out=d_vals)
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
        m,hf,hs,j,hsp,jp,mL,hL,hLp = self.state_vals
        tau, ss = self.calc_taus_ss(vOld)
        
        m = ss.mss - (ss.mss - m) * np.exp(-dt / tau.tm);
        hf = ss.hss - (ss.hss - hf) * np.exp(-dt / tau.thf);
        hs = ss.hss - (ss.hss - hs) * np.exp(-dt / tau.ths);
        j = ss.jss - (ss.jss - j) * np.exp(-dt / tau.tj);

        hsp = ss.hssp - (ss.hssp - hsp) * np.exp(-dt / tau.thsp);
        jp = ss.jss - (ss.jss - jp) * np.exp(-dt / tau.tjp);

        mL = ss.mLss - (ss.mLss - mL) * np.exp(-dt / tau.tmL);
        hL = ss.hLss - (ss.hLss - hL) * np.exp(-dt / tau.thL);
        hLp = ss.hLssp - (ss.hLssp - hLp) * np.exp(-dt / tau.thLp);


        self.state_vals = m,hf,hs,j,hsp,jp,mL,hL,hLp
        return self.calcCurrent(self.state_vals, vOld)
        


class OHaraRudy_Gratz_INa():
    num_params = 31
    param_bounds = [(-3,3), (-3,3),

                   (-3,3),

                   (-1,3), (-15,15),
                   (-3,3), (-0.1,3),
                   (-3,3), (-0.1,3),
                   
                   (-1,3), (-15,15), (-15,15),
                   (-3,3), (-15,15), (-1,3),
                   (-3,3), (-15,15), (-1,3),
                   (-5,5),
                   
                   (-1,3), (-15,15),
                   (-3,3), (-15,15), (-1,3),
                   
                   (-1,3), (-3,3), (-15,15),
                   
                   (-1,3), (-1,3),
                   (-3,3), (-3,3)]
                               
    RGAS = 8314.0;
    FDAY = 96487.0;

    KmCaMK = 0.15
    CaMKa = 1e-5

    def __init__(self, GNaFactor=0, GNaLFactor=0,

                 baselineFactor=0,

                 mss_tauFactor=0, mss_shiftFactor=0,
                 tm_mult1Factor=0, tm_tau1Factor=0,
                 tm_mult2Factor=0, tm_tau2Factor=0,
                 
                 hss_tauFactor=0, hss_shiftFactor=0, hsss_shiftFactor=0,
                 thf_maxFactor=0, thf_shiftFactor=0, thf_tauFactor=0,
                 ths_maxFactor=0, ths_shiftFactor=0, ths_tauFactor=0,
                 Ahf_multFactor=0,

                 jss_tauFactor=0, jss_shiftFactor=0,
                 tj_maxFactor=0, tj_shiftFactor=0, tj_tauFactor=0,

                 hssp_tauFactor=0, tssp_multFactor=0, tjp_multFactor=0,\

                 mLss_tauFactor=0, hLss_tauFactor=0,
                 thL_baselineFactor=0, thLp_multFactor=0,
                 
                 TEMP = 310.0, naO = 140.0, naI = 7):

        # scaling currents 0 
        self.GNa = 75*np.exp(GNaFactor);
        self.GNaL = 0.0075*np.exp(GNaLFactor);

        #fastest tau 2
        self.baseline = 2.038*np.exp(baselineFactor)

        #m gate 3
        self.mss_tau = 9.871*np.exp(mss_tauFactor)
        self.mss_shift = 39.57+mss_shiftFactor

        self.tm_mult1 = 6.765*np.exp(tm_mult1Factor)
        self.tm_tau1 = 34.77*np.exp(tm_tau1Factor)
        self.tm_mult2 = 8.552*np.exp(tm_mult2Factor)
        self.tm_tau2 = 5.955*np.exp(tm_tau2Factor)

        #h gate 9
        self.hss_tau = 6.086*np.exp(hss_tauFactor)
        self.hfss_shift = 82.90+hss_shiftFactor
        self.hsss_shift = 60*np.exp(hsss_shiftFactor)

        self.thf_max = 50*np.exp(thf_maxFactor)
        self.thf_shift = -78+thf_shiftFactor
        self.thf_tau = 15*np.exp(thf_tauFactor)
        
        self.ths_max = 50*np.exp(ths_maxFactor)
        self.ths_shift = -75+ths_shiftFactor
        self.ths_tau = 40*np.exp(ths_tauFactor)

        mixingodds = np.exp(Ahf_multFactor)
        self.Ahf_mult = mixingodds/(mixingodds+1)# np.exp(Ahf_multFactor)

        #j gate 19
        self.jss_tau = 6.086*np.exp(jss_tauFactor)
        self.jss_shift = 80+jss_shiftFactor
        
        self.tj_max = 200*np.exp(tj_maxFactor)
        self.tj_shift = -95+tj_shiftFactor
        self.tj_tau = 3*np.exp(tj_tauFactor)


        # phosphorylated gates 24
        self.hssp_tau = 6.086*np.exp(hssp_tauFactor)
        self.tssp_mult = 3.0*np.exp(tssp_multFactor)
        self.tjp_mult = 1.46*np.exp(tjp_multFactor)

        #late gates & late gate phosphorylation 27
        self.mLss_tau = 5.264*np.exp(mLss_tauFactor)
        self.hLss_tau = 7.488*np.exp(hLss_tauFactor)
        self.hLssp_tau = self.hLss_tau

        self.thL_baseline = 200.0*np.exp(thL_baselineFactor)
        self.thLp_mult = 3*np.exp(thLp_multFactor)
    
        
        self.TEMP = TEMP
        self.naO = naO
        self.naI = naI

        self.recArrayNames = pd.Index(["m","hf","hs","j","hsp","jp","mL","hL","hLp"])
        self.state_vals = pd.Series([0,1,1,1,1,1,0,1,1], index=self.recArrayNames, dtype='float64')
        

        self.recArray = pd.DataFrame(columns=self.recArrayNames)
        
        self.retOptions = {'G': True, 'INa': True, 'INaL': True,\
                                 'Open': True, 'RevPot': True}
        self.lastVal = None
            

    
    def calc_taus_ss(self, vOld):
        if self.lastVal is not None and np.array_equal(self.lastVal[0], vOld):
            return self.lastVal[1]
        tau = ObjDict()
        ss = ObjDict()
        
        ss.mss = 1.0 / (1.0 + np.exp((-(vOld + self.mss_shift+12)) / self.mss_tau));
        tau.tm = self.baseline/15+ 1.0 / (self.tm_mult1 * np.exp((vOld + 11.64) / self.tm_tau1) +
                           self.tm_mult2 * np.exp(-(vOld + 77.42) / self.tm_tau2));

        ss.hfss = 1.0 / (1 + np.exp((vOld + self.hfss_shift+5) / self.hss_tau));
        tau.thf = self.baseline/5 + (self.thf_max-self.baseline/5) / (1+np.exp((vOld-self.thf_shift)/self.thf_tau))

#        tau.thf = self.baseline/5 + 1/(6.149) * np.exp(-(vOld + 0.5096) / 15);
#        if vOld < -100:
#            tau.thf = self.baseline
#        tau.thf = np.clip(tau.thf, a_max=15, a_min=None)

        ss.hsss = 1.0 / (1 + np.exp((vOld + self.hsss_shift-5) / (self.hss_tau+8)));
        tau.ths = self.baseline + (self.ths_max-self.baseline) / (1+np.exp((vOld-self.ths_shift)/self.ths_tau))

#        tau.ths = self.baseline + 1.0 / (0.3343) * np.exp(-(vOld + 5.730) / 30);
#        if vOld < -100:
#            tau.ths = self.baseline
#        tau.ths = np.clip(tau.ths, a_max=20, a_min=None)

        ss.jss = 1.0 / (1 + np.exp((vOld + self.jss_shift+5) / (self.jss_tau)));#hss;
        tau.tj = self.baseline + (self.tj_max-self.baseline)/(1+np.exp(-1/self.tj_tau*(vOld-self.tj_shift)))
        if vOld > -60:
            tau.tj = 100 + (self.tj_max-100)/(1+np.exp(2/self.tj_tau*(vOld-(self.tj_shift+40))))

        ss.hssp = 1.0 / (1 + np.exp((vOld + 89.1) / self.hssp_tau));
        tau.thsp = self.tssp_mult * tau.ths;


        tau.tjp = self.tjp_mult * tau.tj;

        ss.mLss = 1.0 / (1.0 + np.exp((-(vOld + 42.85)) / self.mLss_tau));
        tau.tmL = tau.tm;

        ss.hLss = 1.0 / (1.0 + np.exp((vOld + 87.61) / self.hLss_tau));
        tau.thL = self.thL_baseline;

        ss.hLssp = 1.0 / (1.0 + np.exp((vOld + 93.81) / self.hLssp_tau));
        tau.thLp = self.thLp_mult * tau.thL;

#        tau.__dict__ = {key: min(max(value, 1e-8), 1e20) for key,value in tau.__dict__.items()}
        self.lastVal = (vOld, (tau, ss))
        return tau, ss
    
    def jac(self, vOld):
        vOld = np.array(vOld,ndmin=1)
        d_vals = np.squeeze(np.zeros((9,len(vOld))))
        
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

#        np.clip(d_vals, a_min=-1e15, a_max=None, out=d_vals)

        return np.diag(d_vals)


    def ddtcalc(self, vals, vOld):
        d_vals = np.zeros_like(vals)
        
        tau, ss = self.calc_taus_ss(vOld)
        
        d_vals[0] = (ss.mss-vals[0]) / tau.tm

        d_vals[1] = (ss.hfss-vals[1]) / tau.thf
        d_vals[2] = (ss.hsss-vals[2]) / tau.ths

        d_vals[3] = (ss.jss-vals[3]) / tau.tj

        d_vals[4] = (ss.hssp-vals[4]) / tau.thsp


        d_vals[5] = (ss.jss-vals[5]) / tau.tjp

        d_vals[6] = (ss.mLss-vals[6]) / tau.tmL

        d_vals[7] = (ss.hLss-vals[7]) / tau.thL

        d_vals[8] = (ss.hLssp-vals[8]) / tau.thLp

#        np.clip(d_vals, a_min=-1e15, a_max=1e15, out=d_vals)
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
        
        Ahf = self.Ahf_mult;
        Ahs = 1.0 - Ahf;
        
        h = Ahf * hf + Ahs * hs;
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
        m,hf,hs,j,hsp,jp,mL,hL,hLp = self.state_vals
        tau, ss = self.calc_taus_ss(vOld)
        
        m = ss.mss - (ss.mss - m) * np.exp(-dt / tau.tm);
        hf = ss.hss - (ss.hss - hf) * np.exp(-dt / tau.thf);
        hs = ss.hss - (ss.hss - hs) * np.exp(-dt / tau.ths);
        j = ss.jss - (ss.jss - j) * np.exp(-dt / tau.tj);

        hsp = ss.hssp - (ss.hssp - hsp) * np.exp(-dt / tau.thsp);
        jp = ss.jss - (ss.jss - jp) * np.exp(-dt / tau.tjp);

        mL = ss.mLss - (ss.mLss - mL) * np.exp(-dt / tau.tmL);
        hL = ss.hLss - (ss.hLss - hL) * np.exp(-dt / tau.thL);
        hLp = ss.hLssp - (ss.hLssp - hLp) * np.exp(-dt / tau.thLp);


        self.state_vals = m,hf,hs,j,hsp,jp,mL,hL,hLp
        return self.calcCurrent(self.state_vals, vOld)
        

class OHaraRudy_wMark_INa():
    num_params = 24
    param_bounds = [(-3,3),

                   (-3,3),

                   (-1,3), (-15,15),
                   (-3,3), (-0.1,3),
                   (-3,3), (-0.1,3),
                   
                   (-1,3), (-15,15), (-15,15),
                   (-3,3), (-15,15), (-1,3),
                   (-3,3), (-15,15), (-1,3),
                   (-5,5),
                   
                   (-1,3), (-15,15),
                   (-3,3), (-15,15), (-1,3),
                   (-3,3)
                   ]
                               
    RGAS = 8314.0;
    FDAY = 96487.0;

    KmCaMK = 0.15
    CaMKa = 1e-5

    def __init__(self, GNaFactor=0,

                 baselineFactor=0,

                 mss_tauFactor=0, mss_shiftFactor=0,
                 tm_mult1Factor=0, tm_tau1Factor=0,
                 tm_mult2Factor=0, tm_tau2Factor=0,
                 
                 hss_tauFactor=0, hss_shiftFactor=0, hsss_shiftFactor=0,
                 thf_maxFactor=0, thf_shiftFactor=0, thf_tauFactor=0,
                 ths_maxFactor=0, ths_shiftFactor=0, ths_tauFactor=0,
                 Ahf_multFactor=0,

                 jss_tauFactor=0, jss_shiftFactor=0,
                 tj_maxFactor=0, tj_shiftFactor=0, tj_tauFactor=0,
                 tj2_multFactor=0,
                 
                 TEMP = 310.0, naO = 140.0, naI = 7):

        # scaling currents 0 
        self.GNa = 75*np.exp(GNaFactor);

        #fastest tau 2
        self.baseline = 2.038*np.exp(baselineFactor)

        #m gate 3
        self.mss_tau = 9.871*np.exp(mss_tauFactor)
        self.mss_shift = 51.57+mss_shiftFactor

        self.tm_mult1 = 6.765*np.exp(tm_mult1Factor)
        self.tm_tau1 = 34.77*np.exp(tm_tau1Factor)
        self.tm_mult2 = 8.552*np.exp(tm_mult2Factor)
        self.tm_tau2 = 5.955*np.exp(tm_tau2Factor)

        #h gate 9
        self.hss_tau = 14.086*np.exp(hss_tauFactor)
        self.hfss_shift = -87.90+hss_shiftFactor
        self.hsss_shift = -87.90*np.exp(hsss_shiftFactor)

        self.thf_max = 30*np.exp(thf_maxFactor)
        self.thf_shift = -78+thf_shiftFactor
        self.thf_tau = 15*np.exp(thf_tauFactor)
        
        self.ths_max = 50*np.exp(ths_maxFactor)
        self.ths_shift = -75+ths_shiftFactor
        self.ths_tau = 40*np.exp(ths_tauFactor)

        mixingodds = np.exp(Ahf_multFactor)
        self.Ahf_mult = mixingodds/(mixingodds+1)# np.exp(Ahf_multFactor)

        #j gate 19
        self.jss_tau = 6.086*np.exp(jss_tauFactor)
        self.jss_shift = 95+jss_shiftFactor
        
        self.tj_max = 200*np.exp(tj_maxFactor)
        self.tj_shift = -95+tj_shiftFactor
        self.tj_tau = 3*np.exp(tj_tauFactor)
        
        self.tj2_mult = 10*(np.exp(tj2_multFactor))

       
        self.TEMP = TEMP
        self.naO = naO
        self.naI = naI

        self.recArrayNames = pd.Index(["m",
                                       "hf","jf","if","i2f",
                                       "hs","js","is","i2s"])
        self.state_vals = pd.Series([0,
                                     1,0,0,0,
                                     1,0,0,0], 
                                    index=self.recArrayNames, dtype='float64')
        self.num_states = len(self.recArrayNames)
        

        self.recArray = pd.DataFrame(columns=self.recArrayNames)
        
        self.retOptions = {'G': True, 'INa': True, 'INaL': True,\
                                 'Open': True, 'RevPot': True}
        self.lastVal = None
            

    
    def calc_taus_ss(self, vOld):
        
        vOld = np.array(vOld,ndmin=1)
        if self.lastVal is not None and np.array_equal(self.lastVal[0], vOld):
            return self.lastVal[1]
        tau = ObjDict()
        ss = ObjDict()
        
        num_a_b = 5
        a = np.empty((num_a_b, len(vOld)))
        b = np.empty((num_a_b, len(vOld)))
        
        
        ss.mss = 1.0 / (1.0 + np.exp((-(vOld + self.mss_shift)) / self.mss_tau));
        tau.tm = self.baseline/15+ 1.0 / (self.tm_mult1 * np.exp((vOld + 11.64) / self.tm_tau1) +
                           self.tm_mult2 * np.exp(-(vOld + 77.42) / self.tm_tau2));
        
        a[0] = ss.mss/tau.tm
        b[0] = (1-ss.mss)/tau.tm

        ss.hfss = 1.0 / (1 + np.exp((vOld - self.hfss_shift) / (self.hss_tau)))
        tau.thf = self.baseline/5 + (self.thf_max-self.baseline/5) / (1+np.exp((vOld-self.thf_shift)/self.thf_tau))

        a[1] = ss.hfss/tau.thf
        b[1] = (1-ss.hfss)/tau.thf

#        tau.thf = self.baseline/5 + 1/(6.149) * np.exp(-(vOld + 0.5096) / 15);
#        if vOld < -100:
#            tau.thf = self.baseline
#        tau.thf = np.clip(tau.thf, a_max=15, a_min=None)
        ss.hsss = 1.0 / (1 + np.exp((vOld - self.hsss_shift) / (self.hss_tau)))
        tau.ths = self.baseline + (self.ths_max-self.baseline) / (1+np.exp((vOld-self.ths_shift)/self.ths_tau))
        
        a[2] = ss.hsss/tau.ths
        b[2] = (1-ss.hsss)/tau.ths
        

#        tau.ths = self.baseline + 1.0 / (0.3343) * np.exp(-(vOld + 5.730) / 30);
#        if vOld < -100:
#            tau.ths = self.baseline
#        tau.ths = np.clip(tau.ths, a_max=20, a_min=None)

        ss.jss = 1.0 / (1 + np.exp((vOld + self.jss_shift) / (self.jss_tau)));#hss;
        tau.tj = self.baseline + (self.tj_max-self.baseline)/(1+np.exp(-1/self.tj_tau*(vOld-self.tj_shift)))
#        mask = vOld > -60
#        tau.tj[mask] = 100 + (self.tj_max-100)/(1+np.exp(2/self.tj_tau*(vOld[mask]-(self.tj_shift+40))))
#        tau.tj *= 0.001
#        tau.tj = 0.1

        a[3] = ss.jss/tau.tj
        b[3] = (1-ss.jss)/tau.tj
        
        a[4] = ss.jss/(tau.tj*self.tj2_mult)
        b[4] = (1-ss.jss)/(tau.tj*10)
        

#        tau.__dict__ = {key: min(max(value, 1e-8), 1e20) for key,value in tau.__dict__.items()}
        a = np.squeeze(a)
        b = np.squeeze(b)
        tau.__dict__ = {key: np.squeeze(value) for key,value in tau.__dict__.items()}
        ss.__dict__ = {key: np.squeeze(value) for key,value in ss.__dict__.items()}
        self.lastVal = (vOld, (tau, ss, a, b))
        return self.lastVal[1]
    
    def jac(self, vOld):
        d_vals = np.zeros(self.num_states)
        
        _, _, a, b = self.calc_taus_ss(vOld)
        
        #m
        d_vals[0] = -b[0]#-1 / tau.tm

        #hf jf if i2f
        d_vals[1] = -b[1]
        d_vals[2] = -(b[3]+a[1])
        d_vals[3] = -(a[3]+b[4])
        d_vals[4] = -a[4]

        #hs js is i2s
        d_vals[5] = -b[2]
        d_vals[6] = -(b[3]+a[2])
        d_vals[7] = -(a[3]+b[4])
        d_vals[8] = -a[4]

#        np.clip(d_vals, a_min=-1e15, a_max=None, out=d_vals)

        d_vals = np.diag(d_vals)
        #m (none)
        #hf jf if i2f
        d_vals[1,2] = a[1]
        d_vals[2,1] = b[1]
        d_vals[2,3] = a[3]
        d_vals[3,2] = b[3]
        d_vals[3,4] = a[4]
        d_vals[4,3] = b[4]
        
        #hs js is i2s
        d_vals[5,6] = a[2]
        d_vals[6,5] = b[2]
        d_vals[6,7] = a[3]
        d_vals[7,6] = b[3]
        d_vals[7,8] = a[4]
        d_vals[8,7] = b[4]
      
        return d_vals


    def ddtcalc(self, vals, vOld):
        d_vals = np.zeros_like(vals)
        
        tau, ss, a, b = self.calc_taus_ss(vOld)
        #m
        d_vals[0] = (1-vals[0])*a[0] - vals[0]*b[0]

        #hf jf if i2f
        d_vals[1] = vals[2]*a[1] - vals[1]*b[1]
        d_vals[2] = a[3]*vals[3]+b[1]*vals[1] - vals[2]*(b[3]+a[1])
        d_vals[3] = b[3]*vals[2]+a[4]*vals[4] - vals[3]*(a[3]+b[4])
        d_vals[4] = b[4]*vals[3] - vals[4]*a[4]
        
        #hs js is i2s
        d_vals[5] = vals[6]*a[2] - vals[5]*b[2]
        d_vals[6] = a[3]*vals[7]+b[2]*vals[5] - vals[6]*(b[3]+a[2])
        d_vals[7] = b[3]*vals[6]+a[4]*vals[8] - vals[7]*(a[3]+b[4])
        d_vals[8] = b[4]*vals[7] - vals[8]*a[4]        
        
#        if not np.isclose(np.sum(vals[1:]),2):
#            print(np.sum(vals[1:]))

#        np.clip(d_vals, a_min=-1e15, a_max=1e15, out=d_vals)
        return d_vals
    
    def getRevPot(self):
        return (self.RGAS * self.TEMP / self.FDAY) * np.log(self.naO / self.naI)

    def calcCurrent(self, vals, vOld, ret=[True]*3, setRecArray=True):
        vals = np.array(vals)
        if len(vals.shape) == 1:
            vals.shape = (self.num_states,-1)
        elif vals.shape[0] != self.num_states and vals.shape[-1] == self.num_states:
            vals = vals.T
            
        m, hf,jf,i1f,i2f, hs,js,i1s,i2s = vals

        self.recArray = self.recArray.append(pd.DataFrame(vals.T, columns=self.recArrayNames))
        
        ena = self.getRevPot()
        
        Ahf = self.Ahf_mult;
        Ahs = 1.0 - Ahf;
        
        h = Ahf*hf + Ahs*hs;
#        hp = Ahf * hf + Ahs *hsp;
        
#        fINap = (1.0 / (1.0 + self.KmCaMK / self.CaMKa));
#        oprob = self.m * self.m * self.m * ((1.0 - fINap) * h * self.j + fINap * hp * self.jp)

#        oprob = m**3 * ((1.0 - fINap) *  h  + fINap * hp * jp)
        oprob = m**3 * h

        
        INa = (self.GNa if self.retOptions['G'] else 1) *\
                (oprob if self.retOptions['Open'] else 1) *\
                ((vOld - ena) if self.retOptions['RevPot'] else 1);
        
#        fINaLp = (1.0 / (1.0 + self.KmCaMK / self.CaMKa));
#        loprob = mL * ((1.0 - fINaLp) * hL + fINaLp * hLp)

        INaL = 0#(self.GNaL if self.retOptions['G'] else 1)*\
            #(loprob if self.retOptions['Open'] else 1)*\
#            ((vOld - ena) if self.retOptions['RevPot'] else 1);
        
        return (INa if self.retOptions['INa'] else 0)+\
            (INaL if self.retOptions['INaL'] else 0)

    def update(self, vOld, dt):
        m,hf,hs,j,hsp,jp,mL,hL,hLp = self.state_vals
        tau, ss = self.calc_taus_ss(vOld)
        
        m = ss.mss - (ss.mss - m) * np.exp(-dt / tau.tm);
        hf = ss.hss - (ss.hss - hf) * np.exp(-dt / tau.thf);
        hs = ss.hss - (ss.hss - hs) * np.exp(-dt / tau.ths);
        j = ss.jss - (ss.jss - j) * np.exp(-dt / tau.tj);

        hsp = ss.hssp - (ss.hssp - hsp) * np.exp(-dt / tau.thsp);
        jp = ss.jss - (ss.jss - jp) * np.exp(-dt / tau.tjp);

        mL = ss.mLss - (ss.mLss - mL) * np.exp(-dt / tau.tmL);
        hL = ss.hLss - (ss.hLss - hL) * np.exp(-dt / tau.thL);
        hLp = ss.hLssp - (ss.hLssp - hLp) * np.exp(-dt / tau.thLp);


        self.state_vals = m,hf,hs,j,hsp,jp,mL,hL,hLp
        return self.calcCurrent(self.state_vals, vOld)
        



#10.1161/CIRCULATIONAHA.112.105320 P glynn
class Koval_ina:
    num_params = 22
    param_bounds = \
       [(-3,3)] +\
        [(-3,3)]*3 +\
        [(-3,3)] + [(-0.2,3)] + [(-3,3)] +\
        [(-0.2,3),(-3,3)] + [(-10,10)] +\
        [(-3,3)] + [(-10,10)] + [(-3,3)] +\
        [(-3,3)]*9
    
    RGAS = 8314.4
    FDAY = 96485

    def __init__(self, gNaFactor = 0, 
                 P1a1Factor=0, P2a1Factor=0, P1a4Factor=0, 
                 P1a5Factor=0, P2a5Factor=0, P1b1Factor=0, 
                 P2b1Factor=0, P1b2Factor=0, P2b2Factor=0, 
                 P1b3Factor=0, P2b3Factor=0, P1b5Factor=0, 
                 P2b5Factor=0, P1a6Factor=0, P1b6Factor=0, 
                 P1a7Factor=0, P1b7Factor=0, P1a8Factor=0, 
                 P1b8Factor=0, P1a9Factor=0, P1b9Factor=0, 
                 TEMP = 310.0, naO = 140.0, naI = 8.35504003):

        self.P1a1 = np.exp(P1a1Factor)*7.5207
        self.P2a1 = np.exp(P2a1Factor)*0.1027
        self.P1a4 = np.exp(P1a4Factor)*0.188495
        self.P1a5 = np.exp(P1a5Factor)*7.0e-7
        self.P2a5 = np.exp(P2a5Factor)*7.7
        self.P1b1 = np.exp(P1b1Factor)*0.1917
        self.P2b1 = np.exp(P2b1Factor)*20.3
        self.P1b2 = np.exp(P1b2Factor)*0.2
        self.P2b2 = P2b2Factor+2.5
        self.P1b3 = np.exp(P1b3Factor)*0.22
        self.P2b3 = P2b3Factor+7.5
        self.P1b5 = np.exp(P1b5Factor)*0.0108469
        self.P2b5 = np.exp(P2b5Factor)*2e-5
        self.P1a6 = np.exp(P1a6Factor)*1000.0
        self.P1b6 = np.exp(P1b6Factor)*6.0448e-3
        self.P1a7 = np.exp(P1a7Factor)*1.05263e-5
        self.P1b7 = np.exp(P1b7Factor)*0.02
        self.P1a8 = np.exp(P1a8Factor)*4.0933e-13
        self.P1b8 = np.exp(P1b8Factor)*9.5e-4
        self.P1a9 = np.exp(P1a9Factor)*8.2
        self.P1b9 = np.exp(P1b9Factor)*0.022
        self.gNa  = np.exp(gNaFactor)*7.35

        self.TEMP = TEMP
        self.RanolazineConc = 0
        self.naO = naO
        self.naI = naI

        self.recArrayNames = pd.Index(["C1","C2","C3",
                                       "IC2","IC3","IF",
                                       "IM1","IM2",
                                       "LC1","LC2","LC3",
                                       "O", "LO",
                                       "OB", "LOB"])
        self.state_vals = pd.Series([0.0003850597267,0.02639207662,0.7015088787,
                                     0.009845083654,0.2616851145,0.0001436395221,
                                     3.913769904e-05,3.381242427e-08,
                                     1.659002962e-13,1.137084204e-11,3.02240205e-10,
                                     9.754706096e-07,4.202747013e-16,
                                     0,0],
                                    index=self.recArrayNames)

        self.recArray = pd.DataFrame(columns=self.recArrayNames)

        self.retOptions = {'G': True, 'INa': True, 'INaL': True,\
                                 'Open': True, 'RevPot': True}
        self.lastVal = None


    def calc_alphas_betas(self, vOld):
        if self.lastVal is not None and np.array_equal(self.lastVal[0], vOld):
            return self.lastVal[1]
        exp = np.exp
        a = np.zeros(10)
        b = np.zeros(10)
        
        
        # P123a_max = 45
        # P123a_tau = 6#15
        # P123a_shift = -25
        # P123a_shift_diff = 4-10
        
        # P4a_max = 2.53835453
        # P4a_shift = -20
        # P4a_tau = 16.6
        
        # P5a_max = 600
        # P5a_shift = -160
        # P5a_tau = 6
        
        # a[1] = P123a_max/((exp(-(vOld-(P123a_shift-P123a_shift_diff))/P123a_tau)) + 1)
        # a[2] = P123a_max/((exp(-(vOld-(P123a_shift))/P123a_tau)) + 1)
        # a[3] = P123a_max/((exp(-(vOld-(P123a_shift+P123a_shift_diff))/P123a_tau)) + 1)
        
        # a[4] = P4a_max/(exp(-(vOld-P4a_shift)/P4a_tau) + 1)
        
        # a[5] = P5a_max/(exp((vOld-P5a_shift)/P5a_tau) + 1)
        
        # a[6] = a[4]/self.P1a6
        # a[7] = self.P1a7*a[4]
        # a[8] = self.P1a8
        # a[9] = self.RanolazineConc*self.P1a9

        # b[1] = 400/(exp((vOld+137-P123a_shift_diff)/self.P2b1)+1)
        # b[2] = 400/(exp((vOld+137)/self.P2b1)+1)
        # b[3] = 400/(exp((vOld+137+P123a_shift_diff)/self.P2b1)+1)
        
        # b[5] = self.P1b5 + self.P2b5*(vOld+7.0)
        # b[4] = (a[3]*a[4]*a[5])/(b[3]*b[5])
        # b[6] = self.P1b6*a[5]
        # b[7] = self.P1b7*a[5]
        # b[8] = self.P1b8
        # b[9] = self.P1b9
        
        
        a[1] = self.P1a1/((self.P2a1*exp(-(vOld+2.5)/17))
                        + 0.20*exp(-(vOld+2.5)/150))
        a[2] = self.P1a1/((self.P2a1*exp(-(vOld+2.5)/15))
                        + 0.23*exp(-(vOld+2.5)/150))
        a[3] = self.P1a1/((self.P2a1*exp(-(vOld+2.5)/12))
                        + 0.25*exp(-(vOld+2.5)/150))
        a[4] = 1.0/(self.P1a4*exp(-(vOld+7.0)/16.6) + 0.393956)
        a[5] = self.P1a5*exp(-(vOld+7)/self.P2a5) #self.P1a5*exp(-Vm/self.P2a5)
        a[6] = a[4]/self.P1a6
        a[7] = self.P1a7*a[4]
        a[8] = self.P1a8
        a[9] = self.RanolazineConc*self.P1a9

        b[1] = self.P1b1*exp(-(vOld+2.5)/self.P2b1)
        b[2] = self.P1b2*exp(-(vOld-self.P2b2)/self.P2b1)
        b[3] = self.P1b3*exp(-(vOld-self.P2b3)/self.P2b1)
        b[5] = self.P1b5 + self.P2b5*(vOld+7.0)
        b[4] = (a[3]*a[4]*a[5])/(b[3]*b[5])
        b[6] = self.P1b6*a[5]
        b[7] = self.P1b7*a[5]
        b[8] = self.P1b8
        b[9] = self.P1b9
        
#        if np.max(a) > 1e6 or np.max(b) > 1e6:
#            print('test')
        self.lastVal = (vOld, (a,b))
        
        return a, b

    def ddtcalc(self, vals, vOld):
        C1,C2,C3,\
        IC2,IC3,IF,\
        IM1,IM2,\
        LC1,LC2,LC3,\
        O,LO,\
        OB,LOB = vals
        
        d_vals = np.zeros_like(vals)
        a, b = self.calc_alphas_betas(vOld)
        
        dC1  = (a[5]*IF+b[3]*O+b[8]*LC1+a[2]*C2   -(b[5]+a[3]+a[8]+b[2])*C1)
        dC2  = (a[5]*IC2+b[2]*C1+b[8]*LC2+a[1]*C3 -(b[5]+a[2]+a[8]+b[1])*C2)
        dC3  = (a[5]*IC3+b[1]*C2+b[8]*LC3       -(b[5]+a[1]+a[8])*C3)
        dIC2 = (b[2]*IF+b[5]*C2+a[1]*IC3        -(a[2]+a[5]+b[1])*IC2)
        dIC3 = (b[1]*IC2+b[5]*C3              -(a[1]+a[5])*IC3)
        dIF  = (b[6]*IM1+a[4]*O+b[5]*C1+a[2]*IC2  -(a[6]+b[4]+a[5]+b[2])*IF)
        dIM1 = (b[7]*IM2+a[6]*IF              -(a[7]+b[6])*IM1)
        dIM2 = (a[7]*IM1                    -(b[7])*IM2)
        dLC1 = (a[8]*C1+b[3]*LO+a[2]*LC2        -(b[8]+a[3]+b[2])*LC1)
        dLC2 = (a[8]*C2+b[2]*LC1+a[1]*LC3       -(b[8]+a[2]+b[1])*LC2)
        dLC3 = (b[1]*LC2+a[8]*C3              -(a[1]+b[8])*LC3)
        dO   = (b[9]*OB+b[8]*LO+a[3]*C1+b[4]*IF   -(a[9]+a[8]+b[3]+a[4])*O)
        dLO  = (a[8]*O+b[9]*LOB+a[3]*LC1        -(b[8]+a[9]+b[3])*LO)
        dOB  = (a[9]*O                      -(b[9])*OB)
        dLOB = (a[9]*LO                     -(b[9])*LOB)

        d_vals = dC1,dC2,dC3,dIC2, dIC3, dIF, dIM1, dIM2, dLC1, dLC2, dLC3, dO, dLO, dOB, dLOB
        return d_vals

    def jac(self, vOld):
        a, b = self.calc_alphas_betas(vOld)
        J = np.array([
            [-(b[5]+a[3]+a[8]+b[2]),	a[2],	0,	0,	0,	a[5],	0,	0,	b[8],	0,	0,	b[3],	0,	0,	0],
            [b[2],	-(b[5]+a[2]+a[8]+b[1]),	a[1],	a[5],	0,	0,	0,	0,	0,	b[8],	0,	0,	0,	0,	0],
            [0,	b[1],	-(b[5]+a[1]+a[8]),	0,	a[5],	0,	0,	0,	0,	0,	b[8],	0,	0,	0,	0],
            [0,	b[5],	0,	-(a[2]+a[5]+b[1]),	a[1],	b[2],	0,	0,	0,	0,	0,	0,	0,	0,	0],
            [0,	0,	b[5],	b[1],	-(a[1]+a[5]),	0,	0,	0,	0,	0,	0,	0,	0,	0,	0],
            [b[5],	0,	0,	a[2],	0,	-(a[6]+b[4]+a[5]+b[2]),	b[6],	0,	0,	0,	0,	a[4],	0,	0,	0],
            [0,	0,	0,	0,	0,	a[6],	-(a[7]+b[6]),	b[7],	0,	0,	0,	0,	0,	0,	0],
            [0,	0,	0,	0,	0,	0,	a[7],	-(b[7]),	0,	0,	0,	0,	0,	0,	0],
            [a[8],	0,	0,	0,	0,	0,	0,	0,	-(b[8]+a[3]+b[2]),	a[2],	0,	0,	b[3],	0,	0],
            [0,	a[8],	0,	0,	0,	0,	0,	0,	b[2],	-(b[8]+a[2]+b[1]),	a[1],	0,	0,	0,	0],
            [0,	0,	a[8],	0,	0,	0,	0,	0,	0,	b[1],	-(a[1]+b[8]),	0,	0,	0,	0],
            [a[3],	0,	0,	0,	0,	b[4],	0,	0,	0,	0,	0,	-(a[9]+a[8]+b[3]+a[4]),	b[8],	b[9],	0],
            [0,	0,	0,	0,	0,	0,	0,	0,	a[3],	0,	0,	a[8],	-(b[8]+a[9]+b[3]),	0,	b[9]],
            [0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	a[9],	0,	-b[9],	0],
            [0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	a[9],	0,	-b[9]]
            ])
        return J       
        
    def getRevPot(self):
        log = np.log
        return self.RGAS * self.TEMP / self.FDAY * log((self.naO) / (self.naI))

    def calcCurrent(self, vals, vOld, ret=[True]*3, setRecArray=True):
        vals = np.array(vals)
        if len(vals.shape) == 1:
            vals.shape = (9,-1)
        elif vals.shape[0] != 9 and vals.shape[-1] == 9:
            vals = vals.T
            
        C1,C2,C3,\
        IC2,IC3,IF,\
        IM1,IM2,\
        LC1,LC2,LC3,\
        O,LO,\
        OB,LOB = vals
 
        if setRecArray:
            self.recArray = self.recArray.append(pd.DataFrame(vals.T, columns=self.recArrayNames))

        
        ena = self.getRevPot()
        oprob = (O if self.retOptions['INa'] else 0)+\
                (LO if self.retOptions['INaL'] else 0)

        INa = (self.gNa if self.retOptions['G'] else 1) *\
                (oprob if self.retOptions['Open'] else 1) *\
                ((vOld - ena) if self.retOptions['RevPot'] else 1);
        return INa

    def update(self, vOld, dt):
        d_vals = self.ddtcalc(self.state_vals, vOld)
        self.state_vals += dt*d_vals
        INa = self.calcCurrent(self.state_vals, vOld, setRecArray=True)
        return INa



