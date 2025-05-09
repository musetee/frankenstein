from synthrad_conversion.networks.model_registry import MODEL_REGISTRY

def launch_model(model_name, opt, paths, train_loader, val_loader, mode,
                 train_patient_IDs=None, test_patient_IDs=None):
    if model_name not in MODEL_REGISTRY:
        raise NotImplementedError(f"Model {model_name} is not registered.")

    runner = MODEL_REGISTRY[model_name](
        opt, paths, train_loader, val_loader,
        train_patient_IDs=train_patient_IDs,
        test_patient_IDs=test_patient_IDs
    )

    if mode == 'train':
        runner.train()
    elif mode == 'test':
        runner.test()
    elif mode == 'analyse':
        runner.analyse()