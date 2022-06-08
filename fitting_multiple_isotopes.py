import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#from fitting_routines import fit, plot, prepare_model
import satlas as sat
import glob
import Isotopes


#define model building function
def buildmodel(isotope, spec_y):
    models = []
    model1 = sat.hfsmodel.HFSModel(
                I=isotope["I"],
                J=isotope["J"],
                ABC=isotope["ABC"],
                centroid=isotope['centroid'],
                fwhm=[100,50], #Gaussian~100 for experiment, Laurentz~50 from natural linewidth
                scale=(spec_y.max()-spec_y.min())/2,
                background_params=[spec_y.min()/2],
                shape="voigt",
                sidepeak_params={"Offset":50, "N":0, "Poisson":0},
                use_racah=False)
    model1.set_variation({"Cl": False, "Cu": False})
    models = np.append(models,model1)

    if 'I1' in isotope:
        model2 = sat.hfsmodel.HFSModel(
                    I=isotope["I1"],
                    J=isotope["J1"],
                    ABC=isotope["ABC1"],
                    centroid=isotope['centroid1'],
                    fwhm=[100,50], #Gaussian~100 for experiment, Laurentz~50 from natural linewidth
                    scale=(spec_y.max()-spec_y.min())/2,
                    background_params=[spec_y.min()/2],
                    shape="voigt",
                    sidepeak_params={"Offset":50, "N":0, "Poisson":0},
                    use_racah=False)
        model2.set_variation({"Cl": False, "Cu": False})
        models = np.append(models,model2)
        if 'I2' in isotope:
            model3 = sat.hfsmodel.HFSModel(
                    I=isotope["I2"],
                    J=isotope["J2"],
                    ABC=isotope["ABC2"],
                    centroid=isotope['centroid2'],
                    fwhm=[100,50], #Gaussian~100 for experiment, Laurentz~50 from natural linewidth
                    scale=(spec_y.max()-spec_y.min())/2,
                    background_params=[spec_y.min()/2],
                    shape="voigt",
                    sidepeak_params={"Offset":50, "N":0, "Poisson":0},
                    use_racah=False)
            model3.set_variation({"Cl": False, "Cu": False})
            models = np.append(models,model3)
    
    return models

def get_data(file,iso_num):
    filename = file[5:-4]
    run_num = filename[-3:]
    data = pd.read_csv(file, sep='\t', header=0)
    data = data.fillna(0)
    Vdac = data['dac_volts']
    counts = data['scaler_0'] + data['scaler_1'] + data['scaler_2'] + data['scaler_3']
    dcounts = np.sqrt(counts)
    Volt = 40000 - 1000*Vdac #V
    mass = Isotopes.isotopes[iso_num]["mass"]*931.49432*1e6 #eV/c2
    v0 = (299.792458/214.1754759)*1e9 #MHz
    vref = (299.792458/214.349)*1e9 #MHz
    beta = np.sqrt(1-((mass/(Volt+mass))**2))
    v = v0*np.sqrt((1-beta)/(1+beta))-vref
    v = np.array(v)
    counts = np.array(counts)
    return [filename, run_num, v, counts, dcounts]

#define files and get data
files_123 = ['data/123Te_trs_run086.txt']
files_125 = ['data/125Te_trs_run250.txt']
data_123 = get_data(files_123[0],123)
data_125 = get_data(files_125[0],125)

#build individual model
counts_ref_123 = data_123[-2]
models_123 = buildmodel(Isotopes.isotopes[123], counts_ref_123)
counts_ref_125 = data_125[-2]
models_125 = buildmodel(Isotopes.isotopes[125], counts_ref_125)
model_123 = models_123[0]
for i in models_123[1:]:
    model_123 = model_123 + i
model_125 = models_125[0]
for i in models_125[1:]:
    model_125 = model_125 + i

#build combined model
#model_123.shared = ["FWHMG", "FWHML", "Background0"]
#model_125.shared = ["FWHMG", "FWHML", "Background0"]
model = sat.linkedmodel.LinkedModel([model_123,model_125])
model.params.add('bratio', value=Isotopes.isotopes[123]['ABC1'][3]/Isotopes.isotopes[123]['ABC1'][2])
model.params.add('spin_1_2_aratio', value=-1)
model.params.add('spin_11_2_aratio', value=-1)
model.set_expr({'s0_s1_Bu':'bratio * s0_s1_Bl', 's1_s1_Bu':'bratio * s1_s1_Bl',
                's0_s0_Au':'spin_1_2_aratio*s0_s0_Al', 's1_s0_Au':'spin_1_2_aratio*s1_s0_Al',
                's0_s1_Au':'spin_11_2_aratio*s0_s1_Al', 's1_s1_Au':'spin_11_2_aratio*s1_s1_Al',
                's0_s1_FWHMG':'s0_s0_FWHMG', 's0_s1_FWHML':'s0_s0_FWHML', 's0_s1_Background0':'s0_s0_Background0',
                's1_s1_FWHMG':'s1_s0_FWHMG', 's1_s1_FWHML':'s1_s0_FWHML', 's1_s1_Background0':'s1_s0_Background0'})



#fit
v_ref_123 = data_123[-3]
v_ref_125 = data_125[-3]
sat.fitting.chisquare_spectroscopic_fit(model, [v_ref_123,v_ref_125], [counts_ref_123,counts_ref_125])
model.display_chisquare_fit(show_correl=False, min_correl=0.01)

#export numerical output
output = model.get_result_dict()
df = pd.DataFrame(output,index=['value', 'error'])
df.to_csv('fitting/params/simfit_123_'+data_123[1]+'_125_'+data_125[1]+'params.csv')

#plot 123Te
display_v_123 = np.linspace(v_ref_123.min(), v_ref_123.max(), 1000)
fig, axs = plt.subplots(2)
axs[0].errorbar(data_123[2], data_123[3], yerr=data_123[4], fmt='.k')
axs[0].plot(display_v_123,model_123(display_v_123), color='red', label='combined')
if 'I1' in Isotopes.isotopes[123]:
    axs[0].plot(display_v_123,models_123[0](display_v_123), color='blue', label='ground')
    axs[0].plot(display_v_123,models_123[1](display_v_123), color='green', label='isomer1')
    if 'I2' in Isotopes.isotopes[123]:
        axs[0].plot(display_v_123,models_123[2](display_v_123), color='grey', label='isomer2')
axs[0].set(title='123Te_'+data_123[1], ylabel='counts')
axs[0].legend()
#plot 125Te
display_v_125 = np.linspace(v_ref_125.min(), v_ref_125.max(), 1000)
axs[1].errorbar(data_125[2], data_125[3], yerr=data_125[4], fmt='.k')
axs[1].plot(display_v_125,model_125(display_v_125), color='red', label='combined')
if 'I1' in Isotopes.isotopes[125]:
    axs[1].plot(display_v_125,models_125[0](display_v_125), color='blue', label='ground')
    axs[1].plot(display_v_125,models_125[1](display_v_125), color='green', label='isomer1')
    if 'I2' in Isotopes.isotopes[125]:
        axs[1].plot(display_v_125,models_123[2](display_v_125), color='grey', label='isomer2')
axs[1].set(title='125Te_'+data_125[1], xlabel='frequency(MHz)', ylabel='counts')
axs[1].legend()

axs[0].set_xlim([-4000,2000])
axs[1].set_xlim([-4000,2000])
fig.set_size_inches(18.5, 10.5)
fig.savefig('fitting/plots/simfit_123_'+data_123[1]+'_125_'+data_125[1]+'.png',dpi=100)
plt.show()

