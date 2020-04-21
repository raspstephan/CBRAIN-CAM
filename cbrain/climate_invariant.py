
from cbrain.imports import *
from tensorflow.keras.layers import *
from tensorflow.keras import layers
import enum
import tensorflow_probability as tfp

################################### Tensorflow Versions ##############################################

################### Q2VRH and T2TNS Layers ########################################


def eliq(T):
    a_liq = np.float32(np.array([-0.976195544e-15,-0.952447341e-13,\
                                 0.640689451e-10,\
                      0.206739458e-7,0.302950461e-5,0.264847430e-3,\
                      0.142986287e-1,0.443987641,6.11239921]));
    c_liq = np.float32(-80.0)
    T0 = np.float32(273.16)
    return np.float32(100.0)*tfm.polyval(a_liq,tfm.maximum(c_liq,T-T0))

def eice(T):
    a_ice = np.float32(np.array([0.252751365e-14,0.146898966e-11,0.385852041e-9,\
                      0.602588177e-7,0.615021634e-5,0.420895665e-3,\
                      0.188439774e-1,0.503160820,6.11147274]));
    c_ice = np.float32(np.array([273.15,185,-100,0.00763685,0.000151069,7.48215e-07]))
    T0 = np.float32(273.16)
    return tf.where(T>c_ice[0],eliq(T),\
                   tf.where(T<=c_ice[1],np.float32(100.0)*(c_ice[3]+tfm.maximum(c_ice[2],T-T0)*\
                   (c_ice[4]+tfm.maximum(c_ice[2],T-T0)*c_ice[5])),\
                           np.float32(100.0)*tfm.polyval(a_ice,T-T0)))

def esat(T):
    T0 = np.float32(273.16)
    T00 = np.float32(253.16)
    omtmp = (T-T00)/(T0-T00)
    omega = tfm.maximum(np.float32(0.0),tfm.minimum(np.float32(1.0),omtmp))

    return tf.where(T>T0,eliq(T),tf.where(T<T00,eice(T),(omega*eliq(T)+(1-omega)*eice(T))))

def qv(T,RH,P0,PS,hyam,hybm):
    
    R = np.float32(287.0)
    Rv = np.float32(461.0)
    p = P0 * hyam + PS[:, None] * hybm # Total pressure (Pa)
    
    T = tf.cast(T,tf.float32)
    RH = tf.cast(RH,tf.float32)
    p = tf.cast(p,tf.float32)
    
    return R*esat(T)*RH/(Rv*p)
    # DEBUG 1
    # return esat(T)
    
def RH(T,qv,P0,PS,hyam,hybm):
    R = np.float32(287.0)
    Rv = np.float32(461.0)
    p = P0 * hyam + PS[:, None] * hybm # Total pressure (Pa)
    
    T = tf.cast(T,tf.float32)
    qv = tf.cast(qv,tf.float32)
    p = tf.cast(p,tf.float32)
    
    return Rv*p*qv/(R*esat(T))

def qsat(T,P0,PS,hyam,hybm):
    return qv(T,1,P0,PS,hyam,hybm)



def dP(PS):    
    S = PS.shape
    P = 1e5 * np.tile(hyai,(S[0],1))+np.transpose(np.tile(PS,(31,1)))*np.tile(hybi,(S[0],1))
    return P[:, 1:]-P[:, :-1]



        
###################################### QV2RH and T2TNS layer ##############################################  

class QV2RH(Layer):
    def __init__(self, inp_subQ, inp_divQ, inp_subRH, inp_divRH, hyam, hybm, **kwargs):
        """
        Call using ([input])
        Assumes
        prior: [QBP, 
        TBP, PS, SOLIN, SHFLX, LHFLX]
        Returns
        post(erior): [RHBP,
        TBP, PS, SOLIN, SHFLX, LHFLX]
        Arguments:
        inp_subQ = Normalization based on input with specific humidity (subtraction constant)
        inp_divQ = Normalization based on input with specific humidity (division constant)
        inp_subRH = Normalization based on input with relative humidity (subtraction constant)
        inp_divRH = Normalization based on input with relative humidity (division constant)
        hyam = Constant a for pressure based on mid-levels
        hybm = Constant b for pressure based on mid-levels
        """
        self.inp_subQ, self.inp_divQ, self.inp_subRH, self.inp_divRH, self.hyam, self.hybm = \
            np.array(inp_subQ), np.array(inp_divQ), np.array(inp_subRH), np.array(inp_divRH), \
        np.array(hyam), np.array(hybm)
        # Define variable indices here
        # Input
        self.QBP_idx = slice(0,30)
        self.TBP_idx = slice(30,60)
        self.PS_idx = 60
        self.SHFLX_idx = 62
        self.LHFLX_idx = 63

        super().__init__(**kwargs)

    def build(self, input_shape):
        super().build(input_shape)

    def get_config(self):
        config = {'inp_subQ': list(self.inp_subQ), 'inp_divQ': list(self.inp_divQ),
                  'inp_subRH': list(self.inp_subRH), 'inp_divRH': list(self.inp_divRH),
                  'hyam': list(self.hyam),'hybm': list(self.hybm)}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    def call(self, arrs):
        prior = arrs
        
        Tprior = prior[:,self.TBP_idx]*self.inp_divQ[self.TBP_idx]+self.inp_subQ[self.TBP_idx]
        qvprior = prior[:,self.QBP_idx]*self.inp_divQ[self.QBP_idx]+self.inp_subQ[self.QBP_idx]
        PSprior = prior[:,self.PS_idx]*self.inp_divQ[self.PS_idx]+self.inp_subQ[self.PS_idx]
        RHprior = (RH(Tprior,qvprior,P0,PSprior,self.hyam,self.hybm)-\
                    self.inp_subRH[self.QBP_idx])/self.inp_divRH[self.QBP_idx]
        
        post = tf.concat([tf.cast(RHprior,tf.float32),prior[:,30:]], axis=1)
        
        return post

    def compute_output_shape(self,input_shape):
        """Input shape + 1"""
        return (input_shape[0][0])


class T2TmTNS(Layer):
    def __init__(self, inp_subT, inp_divT, inp_subTNS, inp_divTNS, hyam, hybm, **kwargs):
        """
        From temperature to (temperature)-(near-surface temperature)
        Call using ([input])
        Assumes
        prior: [QBP, 
        TBP, 
        PS, SOLIN, SHFLX, LHFLX]
        Returns
        post(erior): [QBP,
        TfromNS, 
        PS, SOLIN, SHFLX, LHFLX]
        Arguments:
        inp_subT = Normalization based on input with temperature (subtraction constant)
        inp_divT = Normalization based on input with temperature (division constant)
        inp_subTNS = Normalization based on input with (temp - near-sur temp) (subtraction constant)
        inp_divTNS = Normalization based on input with (temp - near-sur temp) (division constant)
        hyam = Constant a for pressure based on mid-levels
        hybm = Constant b for pressure based on mid-levels
        """
        self.inp_subT, self.inp_divT, self.inp_subTNS, self.inp_divTNS, self.hyam, self.hybm = \
            np.array(inp_subT), np.array(inp_divT), np.array(inp_subTNS), np.array(inp_divTNS), \
        np.array(hyam), np.array(hybm)
        # Define variable indices here
        # Input
        self.QBP_idx = slice(0,30)
        self.TBP_idx = slice(30,60)
        self.PS_idx = 60
        self.SHFLX_idx = 62
        self.LHFLX_idx = 63

        super().__init__(**kwargs)

    def build(self, input_shape):
        super().build(input_shape)

    def get_config(self):
        config = {'inp_subT': list(self.inp_subT), 'inp_divT': list(self.inp_divT),
                  'inp_subTNS': list(self.inp_subTNS), 'inp_divTNS': list(self.inp_divTNS),
                  'hyam': list(self.hyam),'hybm': list(self.hybm)}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    def call(self, arrs):
        prior = arrs
        
        Tprior = prior[:,self.TBP_idx]*self.inp_divT[self.TBP_idx]+self.inp_subT[self.TBP_idx]
        
        Tile_dim = tf.constant([1,30],tf.int32)
        TNSprior = ((Tprior-tf.tile(tf.expand_dims(Tprior[:,-1],axis=1),Tile_dim))-\
                    self.inp_subTNS[self.TBP_idx])/\
        self.inp_divTNS[self.TBP_idx]
        
        post = tf.concat([prior[:,:30],tf.cast(TNSprior,tf.float32),prior[:,60:]], axis=1)
        
        return post

    def compute_output_shape(self,input_shape):
        """Input shape + 1"""
        return (input_shape[0][0])


######################################## Scale operation Layer  ############################################  

class OpType(enum.Enum):
    LH_SH=-1
    PWA=0
    LHFLX=1
    PWA_PARTIAL=2
    PWA_PARTIAL_2 = 3 #raise to 0.75
    

class ScaleOp(layers.Layer):
    #if index = -1 that means take shflx+lhflx
    def __init__(self,index,inp_subQ, inp_divQ,**kwargs):
        self.scaling_index = index
        self.inp_subQ, self.inp_divQ =  np.array(inp_subQ), np.array(inp_divQ)
        super(ScaleOp,self).__init__(**kwargs)
        
        
    def get_config(self):
        config = {'index':self.scaling_index,'inp_subQ': list(self.inp_subQ), 'inp_divQ': list(self.inp_divQ)}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    
    def call(self,inps):      
        inp,op = inps
        #for scaling using LHFLX+SHFLX
        if self.scaling_index==OpType.LH_SH.value:
            scaling_factor = (inp[:,62]*self.inp_divQ[62] + self.inp_subQ[62]) + (inp[:,63]*self.inp_divQ[63] + self.inp_subQ[63])
            op_updated = op[:,:60] * tf.expand_dims(scaling_factor,1)
        
        elif self.scaling_index==OpType.PWA.value:
            scaling_factor = inp[:,64]
            op_updated = op[:,:60] * tf.expand_dims(scaling_factor,1)
            
        elif self.scaling_index==OpType.LHFLX.value:
            scaling_factor = inp[:,63]*self.inp_divQ[63] + self.inp_subQ[63]
            op_updated = op[:,:60] * tf.expand_dims(scaling_factor,1)
       
        elif self.scaling_index==OpType.PWA_PARTIAL.value:
            scaling_factor = inp[:,64]
            con_moi = op[:,:30] * tf.expand_dims(scaling_factor,1) * 0.5
            con_heat = op[:,30:60] * tf.expand_dims(scaling_factor,1) * 0.5
            op_updated = tf.concat((con_moi,con_heat),axis=1)
        
        elif self.scaling_index==OpType.PWA_PARTIAL_2.value:
            scaling_factor = inp[:,64]
            con_moi = op[:,:30] * tf.expand_dims(scaling_factor**0.75,1)
            con_heat = op[:,30:60] / tf.expand_dims(scaling_factor**0.75,1)
            op_updated = tf.concat((con_moi,con_heat),axis=1)

        op_rest = op[:,60:]
        op = tf.concat((op_updated,op_rest),axis=1)
        return op


    

######################################## Reverse Interpolation Layer  ############################################  

class reverseInterpLayer(layers.Layer):
    '''
        returns the values of pressure and temperature in the original coordinate system
        
        input - batch_size X (tilde_dimen*2+4) --- 84 in this case
        output - batch_size X 64
        original lev_tilde = batch_size x 30
        x_ref_min - batch_size x 1
        x_ref_max - batch_size x 1
        y_ref - batch_size x interim_dim
    '''
    def __init__(self,interim_dim_size, **kwargs):
        self.interim_dim_size = interim_dim_size #40 for starting
        super(reverseInterpLayer,self).__init__(**kwargs)
    
    
        
    def get_config(self):
        config = {"interim_dim_size":self.interim_dim_size}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    

    def call(self,inputs):      
        X = inputs[0]
        X_original = inputs[1] #batch_size X 30, lev_tilde_before
        x_ref_min = tf.fill(value=0.0,dims=[tf.shape(X)[0],])
        x_ref_max = tf.fill(value=1.4,dims=[tf.shape(X)[0],])
        y_ref_pressure = X[:,:self.interim_dim_size]
        y_ref_temperature = X[:,self.interim_dim_size:2*self.interim_dim_size]
        y_pressure = tfp.math.batch_interp_regular_1d_grid(X_original,x_ref_min,x_ref_max,y_ref_pressure)
        y_temperature = tfp.math.batch_interp_regular_1d_grid(X_original,x_ref_min,x_ref_max,y_ref_temperature)
        y_tilde_before = tf.concat([y_pressure,y_temperature,X[:,2*self.interim_dim_size:]], axis=1)
        return y_tilde_before 
    



#################################################################################################################
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
################################### Numpy Versions ##############################################


################### Q2VRH and T2TNS Layers ########################################

class CrhClass:
    def __init__(self):
        pass
    
    def eliq(self,T):
        a_liq = np.array([-0.976195544e-15,-0.952447341e-13,0.640689451e-10,0.206739458e-7,0.302950461e-5,0.264847430e-3,0.142986287e-1,0.443987641,6.11239921]);
        c_liq = -80
        T0 = 273.16
        return 100*np.polyval(a_liq,np.maximum(c_liq,T-T0))

    def eice(self,T):
        a_ice = np.array([0.252751365e-14,0.146898966e-11,0.385852041e-9,0.602588177e-7,0.615021634e-5,0.420895665e-3,0.188439774e-1,0.503160820,6.11147274]);
        c_ice = np.array([273.15,185,-100,0.00763685,0.000151069,7.48215e-07])
        T0 = 273.16
        return (T>c_ice[0])*self.eliq(T)+\
    (T<=c_ice[0])*(T>c_ice[1])*100*np.polyval(a_ice,T-T0)+\
    (T<=c_ice[1])*100*(c_ice[3]+np.maximum(c_ice[2],T-T0)*(c_ice[4]+np.maximum(c_ice[2],T-T0)*c_ice[5]))    

    def esat(self,T):
        T0 = 273.16
        T00 = 253.16
        omega = np.maximum(0,np.minimum(1,(T-T00)/(T0-T00)))

        return (T>T0)*self.eliq(T)+(T<T00)*self.eice(T)+(T<=T0)*(T>=T00)*(omega*self.eliq(T)+(1-omega)*self.eice(T))

    def RH(self,T,qv,P0,PS,hyam,hybm):
        R = 287
        Rv = 461
        S = PS.shape
        p = 1e5 * np.tile(hyam,(S[0],1))+np.transpose(np.tile(PS,(30,1)))*np.tile(hybm,(S[0],1))

        return Rv*p*qv/(R*self.esat(T))

    def qv(self,T,RH,P0,PS,hyam,hybm):
        R = 287
        Rv = 461
        S = PS.shape
        p = 1e5 * np.tile(hyam,(S[0],1))+np.transpose(np.tile(PS,(30,1)))*np.tile(hybm,(S[0],1))

        return R*self.esat(T)*RH/(Rv*p)


    def qsat(self,T,P0,PS,hyam,hybm):
        return self.qv(T,1,P0,PS,hyam,hybm)



    def dP(self,PS):    
        S = PS.shape
        P = 1e5 * np.tile(hyai,(S[0],1))+np.transpose(np.tile(PS,(31,1)))*np.tile(hybi,(S[0],1))
        return P[:, 1:]-P[:, :-1]


class ThermLibNumpy:
    @staticmethod
    def eliqNumpy(T):
        a_liq = np.float32(np.array([-0.976195544e-15,-0.952447341e-13,\
                                     0.640689451e-10,\
                          0.206739458e-7,0.302950461e-5,0.264847430e-3,\
                          0.142986287e-1,0.443987641,6.11239921]));
        c_liq = np.float32(-80.0)
        T0 = np.float32(273.16)
        return np.float32(100.0)*np.polyval(a_liq,np.maximum(c_liq,T-T0))

    
    @staticmethod    
    def eiceNumpy(T):
        a_ice = np.float32(np.array([0.252751365e-14,0.146898966e-11,0.385852041e-9,\
                          0.602588177e-7,0.615021634e-5,0.420895665e-3,\
                          0.188439774e-1,0.503160820,6.11147274]));
        c_ice = np.float32(np.array([273.15,185,-100,0.00763685,0.000151069,7.48215e-07]))
        T0 = np.float32(273.16)
        return np.where(T>c_ice[0],ThermLibNumpy.eliqNumpy(T),\
                       np.where(T<=c_ice[1],np.float32(100.0)*(c_ice[3]+np.maximum(c_ice[2],T-T0)*\
                       (c_ice[4]+np.maximum(c_ice[2],T-T0)*c_ice[5])),\
                               np.float32(100.0)*np.polyval(a_ice,T-T0)))
    
    @staticmethod
    def esatNumpy(T):
        T0 = np.float32(273.16)
        T00 = np.float32(253.16)
        omtmp = (T-T00)/(T0-T00)
        omega = np.maximum(np.float32(0.0),np.minimum(np.float32(1.0),omtmp))

        return np.where(T>T0,ThermLibNumpy.eliqNumpy(T),np.where(T<T00,ThermLibNumpy.eiceNumpy(T),(omega*ThermLibNumpy.eliqNumpy(T)+(1-omega)*ThermLibNumpy.eiceNumpy(T))))

    @staticmethod
    def qvNumpy(T,RH,P0,PS,hyam,hybm):

        R = np.float32(287.0)
        Rv = np.float32(461.0)
        p = P0 * hyam + PS[:, None] * hybm # Total pressure (Pa)

        T = T.astype(np.float32)
        if type(RH) == int:
            RH = T**0
        RH = RH.astype(np.float32)
        p = p.astype(np.float32)

        return R*ThermLibNumpy.esatNumpy(T)*RH/(Rv*p)
        # DEBUG 1
        # return esat(T)
    
    @staticmethod
    def RHNumpy(T,qv,P0,PS,hyam,hybm):
        R = np.float32(287.0)
        Rv = np.float32(461.0)
        p = P0 * hyam + PS[:, None] * hybm # Total pressure (Pa)

        T = T.astype(np.float32)
        qv = qv.astype(np.float32)
        p = p.astype(np.float32)

        return Rv*p*qv/(R*ThermLibNumpy.esatNumpy(T))
    
    
    @staticmethod
    def qsatNumpy(T,P0,PS,hyam,hybm):
        return ThermLibNumpy.qvNumpy(T,1,P0,PS,hyam,hybm)

    
    @staticmethod
    def qsatsurfNumpy(TS,P0,PS):
        R = 287
        Rv = 461
        return R*ThermLibNumpy.esatNumpy(TS)/(Rv*PS)

    @staticmethod
    def dPNumpy(PS):    
        S = PS.shape
        P = 1e5 * np.tile(hyai,(S[0],1))+np.transpose(np.tile(PS,(31,1)))*np.tile(hybi,(S[0],1))
        return P[:, 1:]-P[:, :-1]


class QV2RHNumpy:
    def __init__(self, inp_sub, inp_div, inp_subRH, inp_divRH, hyam, hybm):
        self.inp_sub, self.inp_div, self.inp_subRH, self.inp_divRH, self.hyam, self.hybm = \
            np.array(inp_sub), np.array(inp_div), np.array(inp_subRH), np.array(inp_divRH), \
        np.array(hyam), np.array(hybm)
        # Define variable indices here
        # Input
        self.QBP_idx = slice(0,30)
        self.TBP_idx = slice(30,60)
        self.PS_idx = 60
        self.SHFLX_idx = 62
        self.LHFLX_idx = 63
        
# tgb - 4/13/2020 - Process function for TS
    def process(self,X):
        Tprior = X[:,self.TBP_idx]*self.inp_div[self.TBP_idx]+self.inp_sub[self.TBP_idx]
        qvprior = X[:,self.QBP_idx]*self.inp_div[self.QBP_idx]+self.inp_sub[self.QBP_idx]
        PSprior = X[:,self.PS_idx]*self.inp_div[self.PS_idx]+self.inp_sub[self.PS_idx]
        RHprior = (ThermLibNumpy.RHNumpy(Tprior,qvprior,P0,PSprior,self.hyam,self.hybm)-\
                    self.inp_subRH[self.QBP_idx])/self.inp_divRH[self.QBP_idx]
        
        X_result = np.concatenate([RHprior.astype(np.float32),X[:,30:]], axis=1)
        return X_result



class LhflxTransNumpy:
    def __init__(self, inp_sub, inp_div, hyam, hybm):
        self.inp_sub, self.inp_div, self.hyam, self.hybm = \
            np.array(inp_sub), np.array(inp_div),\
        np.array(hyam), np.array(hybm)
        # Define variable indices here
        # Input
        self.QBP_idx = slice(0,30)
        self.TBP_idx = slice(30,60)
        self.PS_idx = 60
        self.SHFLX_idx = 62
        self.LHFLX_idx = 63
        self.TS_idx = 64
   
    #     def process(self,X):
#         Tprior = X[:,self.TBP_idx]*self.inp_div[self.TBP_idx]+self.inp_sub[self.TBP_idx]
#         qvprior = X[:,self.QBP_idx]*self.inp_div[self.QBP_idx]+self.inp_sub[self.QBP_idx]
#         PSprior = X[:,self.PS_idx]*self.inp_div[self.PS_idx]+self.inp_sub[self.PS_idx]
#         TSprior = X[:,self.TS_idx]*self.inp_div[self.TS_idx]+self.inp_sub[self.TS_idx]
#         qsatsurf = ThermLibNumpy.qsatsurfNumpy(TSprior,1e5,PSprior)
#         X[:,self.LHFLX_idx] = X[:,self.LHFLX_idx]/(L_V*qsatsurf)
#         return X
    # tgb - 4/13/2020 - Equivalent for TNS
    def process(self,X):
        Tprior = X[:,self.TBP_idx]*self.inp_div[self.TBP_idx]+self.inp_sub[self.TBP_idx]
        qvprior = X[:,self.QBP_idx]*self.inp_div[self.QBP_idx]+self.inp_sub[self.QBP_idx]
        PSprior = X[:,self.PS_idx]*self.inp_div[self.PS_idx]+self.inp_sub[self.PS_idx]
#         TSprior = X[:,self.TS_idx]*self.inp_div[self.TS_idx]+self.inp_sub[self.TS_idx]
        qsat = ThermLibNumpy.qsatNumpy(Tprior,1e5,PSprior,self.hyam,self.hybm)
#         qsatsurf = ThermLibNumpy.qsatsurfNumpy(TSprior,1e5,PSprior)
#         X[:,self.LHFLX_idx] = X[:,self.LHFLX_idx]/(L_V*qsatsurf)
        X[:,self.LHFLX_idx] = X[:,self.LHFLX_idx]/(L_V*qsat[:,-1])
        return X
    
    
class T2TmTNSNumpy:
    def __init__(self, inp_sub, inp_div, inp_subTNS, inp_divTNS, hyam, hybm):
        self.inp_sub, self.inp_div, self.inp_subTNS, self.inp_divTNS, self.hyam, self.hybm = \
            np.array(inp_sub), np.array(inp_div), np.array(inp_subTNS), np.array(inp_divTNS), \
        np.array(hyam), np.array(hybm)
        # Define variable indices here
        # Input
        self.QBP_idx = slice(0,30)
        self.TBP_idx = slice(30,60)
        self.PS_idx = 60
        self.SHFLX_idx = 62
        self.LHFLX_idx = 63
        
    def process(self,X):
        Tprior = X[:,self.TBP_idx]*self.inp_div[self.TBP_idx]+self.inp_sub[self.TBP_idx]
        
        Tile_dim = [1,30]
        TNSprior = ((Tprior-np.tile(np.expand_dims(Tprior[:,-1],axis=1),Tile_dim))-\
                    self.inp_subTNS[self.TBP_idx])/\
        self.inp_divTNS[self.TBP_idx]
        
        post = np.concatenate([X[:,:30],TNSprior.astype(np.float32),X[:,60:]], axis=1)
        
        X_result = post
        return X_result
    

################################### Scaling layer Numpy #################################################
'''appends the scaling factor to the input array'''
'''takes non normalized inputs '''
class ScalingNumpy:
    def __init__(self,hyam,hybm):
        self.hyam = hyam
        self.hybm = hybm
    
    def __crhScaling(self,inp):
        qv0 = inp[:,:30]
        T = inp[:,30:60]
        ps = inp[:,60]
        dP0 = CrhClass().dP(ps)
        qsat0 = CrhClass().qsat(T,P0,ps,self.hyam,self.hybm)
        return np.sum(qv0*dP0,axis=1)/np.sum(qsat0*dP0,axis=1) 
        
    def __pwScaling(self,inp):
        qv0 = inp[:,:30]
        T = inp[:,30:60]
        ps = inp[:,60]
        dP0 = ThermLibNumpy.dPNumpy(ps)
        return np.sum(qv0*dP0/G,axis=1)
    
    def process(self,X):
        scalings = self.__pwScaling(X).reshape(-1,1)
        return scalings
    
    def crh(self,X):
        return self.__crhScaling(X)
    

################################### Level Transformation layer Numpy #################################################
    
'''this is the forward interpolation layer lev-levTilde takes the normalized input vectors'''
    
class InterpolationNumpy:
    def __init__(self,lev,is_continous,Tnot,lower_lim,interm_size):
        self.lev = lev
        self.lower_lim = lower_lim
        self.Tnot = Tnot
        self.is_continous = is_continous ## for discrete or continous transformation
        self.interm_size = interm_size
    
    @staticmethod 
    def levTildeDiscrete(X,lev,inp_sub, inp_div, batch_size=1024,interm_dim_size=40):
        '''can be used independently
            note: the input X should be raw transformed i.e without any other transformation(RH or QV) 
            or if given in that way then please provide appropriate inp_sub, inp_div
        ''' ## not being used in the process method
        X_denormalized = X*inp_div+inp_sub
        X_pressure = X[:,:30]
        X_temperature = X[:,30:60] #batchx30
        X_temperature_denomalized = X_denormalized[:,30:60]

        lev_stacked = np.repeat(np.array(lev).reshape(1,-1),batch_size,axis=0)
        imin = np.argmin(X_temperature_denomalized[:,6:],axis=1)+6
        lev_roof = np.array(lev[imin])
        lev_tilde = (lev_stacked[:,-1].reshape(-1,1)-lev_stacked[:])/(lev_stacked[:,-1].reshape(-1,1)-lev_roof.reshape(-1,1))#batchx30


        lev_tilde_after_single = np.linspace(1.4,0,num=interm_dim_size)

        X_temperature_after = []
        X_pressure_after = []

        for i in range(batch_size):
            X_temperature_after.append(np.interp(lev_tilde_after_single, np.flip(lev_tilde[i]), np.flip(X_temperature[i])))
            X_pressure_after.append(np.interp(lev_tilde_after_single, np.flip(lev_tilde[i]), np.flip(X_pressure[i])))

        X_temperature_after = np.array(X_temperature_after)
        X_pressure_after = np.array(X_pressure_after)

        X_result = np.hstack((X_pressure_after,X_temperature_after))
        X_result = np.hstack((X_result,X[:,60:64]))

        return  X_result, lev_tilde, lev_roof
    
    @staticmethod 
    def levTildeConti(X,lev,inp_sub,inp_div,batch_size=1024,interm_dim_size=40,Tnot=5):
        '''can be used independently
            note: the input X should be raw transformed i.e without any other transformation(RH or QV) 
            or if given in that way then please provide appropriate inp_sub, inp_div
        ''' ## not being used in the process method
        X_denormalized = X*inp_div+inp_sub
        X_pressure = X[:,:30]
        X_temperature = X[:,30:60] #batchx30
        X_temperature_denormalized = X_denormalized[:,30:60]
        lev_stacked = np.repeat(np.array(lev).reshape(1,-1),batch_size,axis=0)
        imin = np.argmin(X_temperature_denormalized[:,6:],axis=1)+6 #take one below this and the next one
        lev1 = np.array(lev[imin-1]) #batch_size dim
        lev2 = np.array(lev[imin+1])
        T1 = np.take_along_axis( X_temperature_denormalized, (imin-1).reshape(-1,1),axis=1).flatten() ## batch_size
        T2 = np.take_along_axis( X_temperature_denormalized, (imin+1).reshape(-1,1),axis=1).flatten() ## batch_size    
        deltaT = abs(T2-T1)
        alpha = (1.0/2)*(2 - np.exp(-1*deltaT/Tnot))
        lev_roof = alpha*lev1 + (1-alpha)*lev2

        lev_tilde = (lev_stacked[:,-1].reshape(-1,1)-lev_stacked[:])/(lev_stacked[:,-1].reshape(-1,1)-lev_roof.reshape(-1,1))#batchx30
        lev_tilde_after_single = np.linspace(1.4,0,num=interm_dim_size)

        X_temperature_after = []
        X_pressure_after = []

        for i in range(batch_size):
            X_temperature_after.append(np.interp(lev_tilde_after_single, np.flip(lev_tilde[i]), np.flip(X_temperature[i])))
            X_pressure_after.append(np.interp(lev_tilde_after_single, np.flip(lev_tilde[i]), np.flip(X_pressure[i])))

        X_temperature_after = np.array(X_temperature_after)
        X_pressure_after = np.array(X_pressure_after)

        X_result = np.hstack((X_pressure_after,X_temperature_after))
        X_result = np.hstack((X_result,X[:,60:64]))

        return  X_result, lev_tilde, lev_roof

    
    def process(self,X,X_result):

        batch_size = X.shape[0]
        X_temperature = X[:,30:60]
        lev_stacked = np.repeat(np.array(self.lev).reshape(1,-1),batch_size,axis=0)
        imin = np.argmin(X_temperature[:,self.lower_lim:],axis=1)+self.lower_lim
        if self.is_continous:
            lower = np.clip(imin-1,0,None)
            upper = np.clip(imin+1,None,29)
            lev1 = np.array(self.lev[lower]) #batch_size dim
            lev2 = np.array(self.lev[upper])
            T1 = np.take_along_axis( X_temperature, (lower).reshape(-1,1),axis=1).flatten() ## batch_size
            T2 = np.take_along_axis( X_temperature, (upper).reshape(-1,1),axis=1).flatten() ## batch_size    
            deltaT = abs(T2-T1)
            alpha = (1.0/2)*(2 - np.exp(-1*deltaT/self.Tnot))
            lev_roof = alpha*lev1 + (1-alpha)*lev2
        else:
            lev_roof = np.array(self.lev[imin])
            
        lev_tilde = (lev_stacked[:,-1].reshape(-1,1)-lev_stacked[:])/(lev_stacked[:,-1].reshape(-1,1)-lev_roof.reshape(-1,1))

        
        ## lef tilde after 
        X_temperature = X_result[:,30:60] #batchx30
        X_pressure = X_result[:,:30]
        lev_tilde_after_single = np.linspace(1.4,0,num=self.interm_size)
        X_temperature_after = []
        X_pressure_after = []

        for i in range(batch_size):
            X_temperature_after.append(np.interp(lev_tilde_after_single, np.flip(lev_tilde[i]), np.flip(X_temperature[i])))
            X_pressure_after.append(np.interp(lev_tilde_after_single, np.flip(lev_tilde[i]), np.flip(X_pressure[i])))
            
        X_temperature_after = np.array(X_temperature_after)
        X_pressure_after = np.array(X_pressure_after)
        X_processed = np.hstack((X_pressure_after,X_temperature_after,X_result[:,60:64],lev_tilde))
        
        return X_processed

#################################################################################################################