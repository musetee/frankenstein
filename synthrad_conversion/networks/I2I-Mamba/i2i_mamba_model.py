from synthrad_conversion.networks.model_registry import register_model
import numpy as np

@register_model('ddpm2d_seg2med_multimodal')
class DDPM2DSeg2Med_multimodal_Runner:
    def __init__(self, opt, paths, train_loader, val_loader, train_patient_IDs, test_patient_IDs):
        self.model = i2i_mamba(
            opt, paths,
            train_loader=train_loader,
            val_loader=val_loader,
            train_patient_IDs=train_patient_IDs,
            test_patient_IDs=test_patient_IDs
        )
        self.opt = opt

    def train(self):
        self.model.train()

    def test(self):
        self.model.val()

from options.train_options import TrainOptions
class i2i_mamba:
    def __init__(self, paths, train_loader, val_loader) -> None:
        opt = TrainOptions().parse()
        if opt.model=='cycle_gan':
            L1_avg=np.zeros([2,opt.niter + opt.niter_decay,len(dataset_val)])      
        else:
            L1_avg=np.zeros([opt.niter + opt.niter_decay,len(dataset_val)])      
        
        model = create_model(opt)
        visualizer = Visualizer(opt)
        total_steps = 0