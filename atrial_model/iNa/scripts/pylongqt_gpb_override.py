#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 12:40:05 2020

@author: grat05
"""


import PyLongQt as pylqt

import sys
import os
sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), 
                 os.pardir, os.pardir, os.pardir)))

from atrial_model.iNa import models

ina_model = models.OHaraRudy_wMark_INa


class GpbHumanAtrialSubmodel(pylqt.Cells.GpbAtrialOnal17):
    __name__ = "Human Atrial Hund 2020"
    def __init__(self):
        pylqt.Cells.GpbAtrialOnal17.__init__(self)
        self.model = None
        self.model_init = None
    def clone(self):
        new_model = GpbHumanAtrialSubmodel()
        new_model.model_init = self.model_init
        return new_model
    def setup(self):
        if self.model is None:
            if self.model_init is None:
                raise AttributeError("iNa channel model not set")
            TEMP = self.pars['TEMP']
            naO = self.pars['naO']
            naI = self.vars['naI']
            self.model = self.model_init(TEMP=TEMP, naO=naO, naI=naI)
            self.model.memoize = False
    def cell_type(self):
        return self.__name__
    def set_submodel(self, model_init):
        self.model_init = model_init
    def updateIna(self):
        self.setup()
        vOld = self.vOld
        dt = self.dt
        Fjunc = self.pars['Fjunc']
        Fsl = self.pars['Fsl']
        
        iNa = self.model.update(vOld, dt, record=False)
        
        iNajunc = Fjunc * iNa
        iNasl = Fsl * iNa
        self.vars['iNa'] = iNa
        self.vars['iNajunc'] = iNajunc
        self.vars['iNasl'] = iNasl
    def updateInaL(self):
        pass

cell_ep_model = GpbHumanAtrialSubmodel()
to_trace = cell_ep_model.variableSelection
to_trace.add('iNa')
cell_ep_model.variableSelection = to_trace
cell_ep_model.model_init = ina_model

proto = pylqt.Protocols.CurrentClamp()
proto.cell = cell_ep_model
proto.tMax = 20000
proto.writetime = 0
proto.runSim()