#!/usr/bin/env julia
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 14:35:48 2020

@author: grat05
"""

struct OHaraRudy_INa
    num_params = 29
    param_bounds = Base.ones((2,num_params)).*[-5,5]
    RGAS = 8314.0;
    FDAY = 96487.0;

    KmCaMK = 0.15
    CaMKa = 1e-5
    
    function OHaraRudy_INa(GNaFactor=0, GNaLFactor=0, \
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
                 TEMP = 310.0, naO = 140.0, naI = 7)
                 
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
    end

    def __init__(self, ):



    def updateIna(self, vOld, dt, ret=[True]*3):
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
end