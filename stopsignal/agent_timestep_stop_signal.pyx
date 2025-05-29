import numpy as np
import pandas as pd
import warnings


def multitimestep_mutator(agent,popdata,numsteps):
    for i in range(numsteps):
        #print('antes', agent.AMPA_eff[7][4])
        timestep_mutator(agent,popdata)
        #print('despues', agent.AMPA_eff[7][4])
    #print('out', agent.AMPA_eff[7][4])
        
        

def get_fDA(DA,dpmn_type,dpmn_x_fda,dpmn_y_fda,dpmn_d2_DA_eps):
    fda = np.zeros(len(DA))
    if dpmn_type < 1.5: # D1
      
        mask_dpmn = np.array([ x<-y for x,y in zip(DA,dpmn_x_fda)])
        fda[mask_dpmn==True] = -dpmn_y_fda[mask_dpmn==True]
        fda[mask_dpmn==False] = (dpmn_y_fda[mask_dpmn==False]/dpmn_x_fda[mask_dpmn==False])*DA[mask_dpmn==False]

    elif dpmn_type > 1.5:
        mask_dpmn = np.array([ x>y for x,y in zip(DA,dpmn_x_fda)])
        fda[mask_dpmn==True] = dpmn_y_fda[mask_dpmn==True]*dpmn_d2_DA_eps[mask_dpmn==True] 
        fda[mask_dpmn==False] = (dpmn_y_fda[mask_dpmn==False]/dpmn_x_fda[mask_dpmn==False])*DA[mask_dpmn==False]*dpmn_d2_DA_eps[mask_dpmn==False]
    
    #print("fda",fda)
    return fda


def get_f5HT(t, srtn_a, srtn_b, srtn_c): 
    #if t == 0: 
        #f_srtn_5HT =  (0/(srtn_c + np.exp(-srtn_b * t)))
    #else: 
        #f_srtn_5HT =  (srtn_a/(srtn_c + np.exp(-srtn_b * (t))) + srtn_a #(t-mid)
    
    warnings.filterwarnings('ignore')
    
    mid = 15000 #10000
    mid = np.float128(mid)

    if t == 0: 
        f_srtn_5HT =  0./(srtn_c + np.exp(-srtn_b * (t-mid))) #(t-mid)
    else: 
        f_srtn_5HT =  srtn_a/(srtn_c + np.exp(-srtn_b * (t-mid))) #(t-mid)
    

    return f_srtn_5HT


def timestep_mutator(a,popdata):

    Npop = len(popdata)

    newspikes = []
    for i in range(Npop):
        newspikes.append([])

    # 1337-1345
    for popid in range(len(popdata)):
        a.dpmn_XPRE[popid] *= 0
        a.dpmn_XPOST[popid] *= 0

        a.srtn_XPRE[popid] *= 0 
        a.srtn_XPOST[popid] *= 0

        #print('here1.1')
        
    # I_ext = ExtS_AMPA * (V(t) - V_E) + ExtS_NMDA * (V(t) - V_E) + ExtS_GABA * (V(t) - V_I)
    # Where ExtS_AMPA, ExtS_NMDA, and ExtS_GABA are mean-reverting random walk
    # It depends on the FreqExt (external frequency), MeanExtEff (efficacy of the external connections), and MeanExtCon (number of external connections).
    
    for popid in range(len(popdata)):

    #print('here1.2')
      
        a.ExtMuS_AMPA[popid] = a.MeanExtEff_AMPA[popid] * a.FreqExt_AMPA[popid] * .001 * a.MeanExtCon_AMPA[popid] * a.Tau_AMPA[popid]
        a.ExtSigmaS_AMPA[popid] = a.MeanExtEff_AMPA[popid] * np.sqrt(a.Tau_AMPA[popid] * .5 * a.FreqExt_AMPA[popid] * .001 * a.MeanExtCon_AMPA[popid])
        a.ExtS_AMPA[popid] += a.dt / a.Tau_AMPA[popid] * (-a.ExtS_AMPA[popid] + a.ExtMuS_AMPA[popid]) + a.ExtSigmaS_AMPA[popid] * np.sqrt(a.dt * 2. / a.Tau_AMPA[popid]) * np.random.normal(size=len(a.Tau_AMPA[popid]))
        
        a.LS_AMPA[popid] *= np.exp(-a.dt / a.Tau_AMPA[popid])
        a.S_AMPA[popid] *= np.exp(-a.dt / a.Tau_AMPA[popid])

        if a.srtn_type[popid][0] > 0: 
            a.LS_AMPA_d1[popid] *= np.exp(-a.dt / a.Tau_AMPA[popid]) 
            #a.S_AMPA_d1[popid] *= np.exp(-a.dt / a.Tau_AMPA[popid])

    for src_popid in range(len(popdata)):
        for dest_popid in range(len(popdata)):
            if a.AMPA_con[src_popid][dest_popid] is not None:
                for src_neuron in a.spikes[src_popid]:
                    a.LS_AMPA[dest_popid] += a.AMPA_eff[src_popid][dest_popid][src_neuron] * a.AMPA_con[src_popid][dest_popid][src_neuron]
                    
                    if src_popid == 7:
                        if dest_popid == 4: 
                            if a.srtn_thalamus[src_popid][0] > 0:
                                a.WS_AMPA_d1[dest_popid] = a.AMPA_eff[src_popid][dest_popid][src_neuron] * a.AMPA_con[src_popid][dest_popid]
                                #print('WS_AMPA_d1', a.WS_AMPA_d1[dest_popid])
                                a.LS_AMPA_d1[dest_popid] += a.AMPA_eff[src_popid][dest_popid][src_neuron] * a.AMPA_con[src_popid][dest_popid][src_neuron]
                                
                                
                    
                    
                    
                    a.dpmn_XPRE[dest_popid] = np.maximum(a.dpmn_XPRE[dest_popid], a.dpmn_cortex[src_popid][src_neuron] * a.AMPA_con[src_popid][dest_popid][src_neuron] * np.sign(a.dpmn_type[dest_popid]))
                    a.srtn_XPRE[dest_popid] = np.maximum(a.srtn_XPRE[dest_popid], a.srtn_thalamus[src_popid][src_neuron] * a.AMPA_con[src_popid][dest_popid][src_neuron] * np.sign(a.srtn_type[dest_popid]))
                
                    
    for popid in range(len(popdata)):
        a.ExtMuS_GABA[popid] = a.MeanExtEff_GABA[popid] * a.FreqExt_GABA[popid] * .001 * a.MeanExtCon_GABA[popid] * a.Tau_GABA[popid]
        a.ExtSigmaS_GABA[popid] = a.MeanExtEff_GABA[popid] * np.sqrt(a.Tau_GABA[popid] * .5 * a.FreqExt_GABA[popid] * .001 * a.MeanExtCon_GABA[popid])
        a.ExtS_GABA[popid] += a.dt / a.Tau_GABA[popid] * (-a.ExtS_GABA[popid] + a.ExtMuS_GABA[popid]) + a.ExtSigmaS_GABA[popid] * np.sqrt(a.dt * 2. / a.Tau_GABA[popid]) * np.random.normal(size=len(a.Tau_AMPA[popid]))
        a.LS_GABA[popid] *= np.exp(-a.dt / a.Tau_GABA[popid])

    for src_popid in range(len(popdata)):
        for dest_popid in range(len(popdata)):
            if a.GABA_con[src_popid][dest_popid] is not None:
                for src_neuron in a.spikes[src_popid]:
                    a.LS_GABA[dest_popid] += a.GABA_eff[src_popid][dest_popid][src_neuron] * a.GABA_con[src_popid][dest_popid][src_neuron]

    for popid in range(len(popdata)):
        a.ExtMuS_NMDA[popid] = a.MeanExtEff_NMDA[popid] * a.FreqExt_NMDA[popid] * .001 * a.MeanExtCon_NMDA[popid] * a.Tau_NMDA[popid]
        a.ExtSigmaS_NMDA[popid] = a.MeanExtEff_NMDA[popid] * np.sqrt(a.Tau_NMDA[popid] * .5 * a.FreqExt_NMDA[popid] * .001 * a.MeanExtCon_NMDA[popid])
        a.ExtS_NMDA[popid] += a.dt / a.Tau_NMDA[popid] * (-a.ExtS_NMDA[popid] + a.ExtMuS_NMDA[popid]) + a.ExtSigmaS_NMDA[popid] * np.sqrt(a.dt * 2. / a.Tau_NMDA[popid]) * np.random.normal(size=len(a.Tau_AMPA[popid]))
        a.LS_NMDA[popid] *= np.exp(-a.dt / a.Tau_NMDA[popid])
        a.timesincelastspike[popid] += a.dt 
        a.Ptimesincelastspike[popid] += a.dt

    for src_popid in range(len(popdata)):
        for dest_popid in range(len(popdata)):
            if a.NMDA_con[src_popid][dest_popid] is not None:
                for src_neuron in a.spikes[src_popid]:
                    ALPHA = 0.6332
                    a.LastConductanceNMDA[src_popid][dest_popid][src_neuron] *= np.exp(-a.Ptimesincelastspike[src_popid][src_neuron]/a.Tau_NMDA[dest_popid])
                    a.LS_NMDA[dest_popid] += a.NMDA_eff[src_popid][dest_popid][src_neuron] * a.NMDA_con[src_popid][dest_popid][src_neuron] * ALPHA * (1 - a.LastConductanceNMDA[src_popid][dest_popid][src_neuron])
                    a.LastConductanceNMDA[src_popid][dest_popid][src_neuron] += ALPHA * (1 - a.LastConductanceNMDA[src_popid][dest_popid][src_neuron])

    for popid in range(len(popdata)):
        a.cond[popid] = (a.V[popid] < a.V_h[popid]).astype(int)
        # true (cond = 1)
        a.h[popid] = a.h[popid] + a.cond[popid] * a.dt * (1 - a.h[popid]) / a.tauhp[popid]
        # false (cond = 0)
        a.h[popid] = a.h[popid] + (1 - a.cond[popid]) * a.dt * (-a.h[popid]) / a.tauhm[popid]
        # mix
        a.g_rb[popid] = a.g_T[popid] * a.h[popid] * (1 - a.cond[popid])

    for popid in range(len(popdata)):
        # 0 = 1st continue, 1 = proceed
        a.cond[popid] = (a.V[popid] <= a.Threshold[popid]).astype(int)
        a.V[popid] -= (a.V[popid] - a.ResetPot[popid]) * (1 - a.cond[popid])
        # 0 = 1st or 2nd continue, 1 = proceed
        a.cond[popid] = a.cond[popid] * (a.RefrState[popid] == 0).astype(int)
        a.RefrState[popid] -= np.sign(a.RefrState[popid]) * (1 - a.cond[popid])

        a.g_adr[popid] = a.g_adr_max[popid] / (1 + np.exp((a.V[popid]-a.Vadr_h[popid]) / a.Vadr_s[popid]))

        a.dv[popid] = a.V[popid] + 55
        a.tau_n[popid] = a.tau_k_max[popid] / (np.exp(-1 * a.dv[popid] / 30) + np.exp(a.dv[popid] / 30))
        a.n_inif[popid] = 1 / (1 + np.exp(-(a.V[popid] - a.Vk_h[popid]) / a.Vk_s[popid]))
        a.n_k[popid] = a.n_k[popid] + a.cond[popid] * -a.dt / a.tau_n[popid] * (a.n_k[popid] - a.n_inif[popid])
        a.g_k[popid] = a.g_k_max[popid] * a.n_k[popid]

        a.V[popid] = a.V[popid] + a.cond[popid] * -a.dt * (1 / a.Taum[popid] * (a.V[popid] - a.RestPot[popid]) + a.Ca[popid] * a.g_ahp[popid] / a.C[popid] * 0.001 * (a.V[popid] - a.Vk[popid]) + a.g_adr[popid] / a.C[popid] * (a.V[popid] - a.ADRRevPot[popid]) + a.g_k[popid] / a.C[popid] * (a.V[popid] - a.ADRRevPot[popid]) + a.g_rb[popid] / a.C[popid] * (a.V[popid] - a.V_T[popid]))
        a.Ca[popid] = a.Ca[popid] - a.cond[popid] * a.Ca[popid] * a.dt / a.Tau_ca[popid]

        a.Vaux[popid] = np.minimum(a.V[popid],a.Threshold[popid])

        a.V[popid] = a.V[popid] + a.cond[popid] * a.dt * (a.RevPot_NMDA[popid] - a.Vaux[popid]) * .001 * (a.LS_NMDA[popid] + a.ExtS_NMDA[popid]) / a.C[popid] / (1. + np.exp(-0.062 * a.Vaux[popid] / 3.57))
        
        a.V[popid] = a.V[popid] + a.cond[popid] * a.dt * (a.RevPot_AMPA[popid] - a.Vaux[popid]) * .001 * (a.LS_AMPA[popid] + a.ExtS_AMPA[popid]) / a.C[popid]
        
        a.temp = a.V[popid]
        #a.temp_s = a.LS_AMPA[popid] + a.ExtS_AMPA[popid]

        
       
        if a.srtn_type[popid][0] > 0:  #popid = dSPN
            a.d1_EPSC = (a.Vaux[popid] - a.RevPot_AMPA[popid]) * (a.LS_AMPA_d1[popid])
            #a.d1_EPSP = (a.V[popid] - a.RevPot_AMPA[popid]) * (a.WS_AMPA_d1[popid]) 
            a.d1_EPSP_1 =  a.d1_EPSC / (a.WS_AMPA_d1[popid]) 
            a.d1_EPSP_2 =  (a.Vaux[popid] - a.RevPot_AMPA[popid]) * (a.WS_AMPA_d1[popid]) 
            #print('a.d1_EPSC', a.d1_EPSC)
            #print('a.WS_AMPA_d1[popid]', a.WS_AMPA_d1[popid])

        #if a.srtn_thalamus[popid][0] > 0:  #popid = Th
            #a.th_EPSC = (a.V[popid] - a.RevPot_AMPA[popid]) * .001 * (a.LS_AMPA_d1[popid])
            #a.th_EPSP = (a.V[popid] - a.RevPot_AMPA[popid]) * .001 * (a.S_AMPA_d1[popid])

        

        a.V[popid] = a.V[popid] + a.cond[popid] * a.dt * (a.RevPot_GABA[popid] - a.Vaux[popid]) * .001 * (a.LS_GABA[popid] + a.ExtS_GABA[popid]) / a.C[popid]


    for popid in range(len(popdata)):
        newspikes[popid] = list(np.nonzero(a.V[popid] > a.Threshold[popid])[0])
        for neuron in newspikes[popid]:
            a.V[popid][neuron] = 0
            a.Ca[popid][neuron] += a.alpha_ca[popid][neuron]
            a.RefrState[popid][neuron] = 10
            a.Ptimesincelastspike[popid][neuron] = a.timesincelastspike[popid][neuron]
            a.timesincelastspike[popid][neuron] = 0
            a.dpmn_XPOST[popid][neuron] = 1
            #print('post')
            a.srtn_XPOST[popid][neuron] = 1 #New - 5HT implementation? 

            #print('here3')

    a.spikes = newspikes

    for popid in range(len(popdata)):
        if a.dpmn_type[popid][0] > 0:
            a.dpmn_DAp[popid] -= (a.dt * a.dpmn_DAp[popid]) / a.dpmn_tauDOP[popid]
            a.dpmn_APRE[popid] += a.dt * (a.dpmn_dPRE[popid] * a.dpmn_XPRE[popid] - a.dpmn_APRE[popid]) / a.dpmn_tauPRE[popid]
            a.dpmn_APOST[popid] += a.dt * (a.dpmn_dPOST[popid] * a.dpmn_XPOST[popid] - a.dpmn_APOST[popid]) / a.dpmn_tauPOST[popid]

            a.dpmn_E[popid] += a.dt * (a.dpmn_XPOST[popid] * a.dpmn_APRE[popid] - a.dpmn_XPRE[popid] * a.dpmn_APOST[popid] - a.dpmn_E[popid]) / a.dpmn_tauE[popid]

            DA = a.dpmn_m[popid] * (a.dpmn_DAp[popid] + a.dpmn_DAt[popid])
            
            if a.dpmn_type[popid][0] < 1.5:
                fDA = a.dpmn_fDA_D1 = get_fDA(DA,a.dpmn_type[popid][0],a.dpmn_x_fda[popid],a.dpmn_y_fda[popid],a.dpmn_d2_DA_eps[popid])
            elif a.dpmn_type[popid][0] > 1.5:
                fDA = a.dpmn_fDA_D2 = get_fDA(DA,a.dpmn_type[popid][0],a.dpmn_x_fda[popid],a.dpmn_y_fda[popid],a.dpmn_d2_DA_eps[popid])
            #print("fDA",fDA)
            
            for src_popid in range(len(popdata)):
                if a.dpmn_cortex[src_popid][0] > 0:
                    if a.AMPA_con[src_popid][popid] is not None:
                        
                        update = a.dt * a.AMPA_con[src_popid][popid] * a.dpmn_alphaw[popid] * fDA * a.dpmn_E[popid]
                        
                        update = np.maximum(update,-1)

                        update = np.minimum(update,1)
                        ind_pos = np.greater(update,0).astype(int)

                        a.AMPA_eff[src_popid][popid] += (update * ind_pos * (a.dpmn_wmax[popid] - a.AMPA_eff[src_popid][popid]))
                       
                        ind_neg = np.less(update,0).astype(int)
                        print(ind_neg)
                        a.AMPA_eff[src_popid][popid] += (update * ind_neg * (a.AMPA_eff[src_popid][popid] - 0.001)) # w_min = 0.01
                        
                
        #New - 5HT implementation from here 
        if a.srtn_type[popid][0] > 0: 

            #f_5HT = get_f5HT(a.phasetimer_0, a.srtn_a[popid], a.srtn_b[popid], a.srtn_c[popid])        
            a.srtn_5HT[popid] = get_f5HT(a.phasetimer_0, a.srtn_a[popid], a.srtn_b[popid], a.srtn_c[popid])              
            a.srtn_fK5HT[popid] = - a.srtn_5HT[popid] +  a.srtn_a[popid]/a.srtn_c[popid]
                       
            #a.srtn_5HT[popid] -= (a.dt * a.srtn_5HT[popid]) / a.srtn_tau5HT[popid]
            a.srtn_APRE[popid] += a.dt * (a.srtn_dPRE[popid] * a.srtn_XPRE[popid] - a.srtn_APRE[popid]) / a.srtn_tauPRE[popid]
            a.srtn_APOST[popid] += a.dt * (a.srtn_dPOST[popid]*100 * a.srtn_XPOST[popid] - a.srtn_APOST[popid]) / a.srtn_tauPOST[popid]

    
            a.srtn_E[popid] += a.dt * (a.srtn_lambda1[popid] * a.srtn_XPOST[popid] * a.srtn_APRE[popid] - a.srtn_lambda2[popid] * a.srtn_XPRE[popid] * a.srtn_APOST[popid] - a.srtn_E[popid]) / a.srtn_tauE[popid]
                   

            for src_popid in range(len(popdata)):
                
                if a.srtn_thalamus[src_popid][0] > 0:
                    
                    #print('src_popid', src_popid)
                    
                    if a.AMPA_con[src_popid][popid] is not None:

                        if a.phase == 0: 
                        
                            #a.srtn_alphaw[popid] = 0  # Tim's suggestion 
                            
                            #set update = 0.5 and check weights update - Cati's suggestion 
                            
                            update = a.dt * a.AMPA_con[src_popid][popid] * a.srtn_alphaw[popid] * (a.srtn_E[popid]) *  a.srtn_fK5HT[popid]
                            
                            a.AMPA_eff[src_popid][popid] += ((update * (a.srtn_wmax[popid] - a.AMPA_eff[src_popid][popid]) * (a.AMPA_eff[src_popid][popid] - a.srtn_wmin[popid])) + a.dt * a.srtn_gamma[popid] * (a.srtn_wbase[popid] - a.AMPA_eff[src_popid][popid])) 
                            
                            
                        
    for popid in range(len(popdata)):
        a.rollingbuffer[popid][a.bufferpointer] = len(a.spikes[popid])
    a.bufferpointer += 1
    if a.bufferpointer >= a.bufferlength:
        a.bufferpointer = 0

    #print('despues',  a.AMPA_eff[7][4])
        
