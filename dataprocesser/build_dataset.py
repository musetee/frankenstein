class BaseDataLoader:
    def __init__(self, configs, paths=None, dimension=2, **kwargs): 
        self.configs=configs
        self.paths=paths
        self.init_parameters_and_transforms()
        self.get_loader()
        #print('all files in dataset:',len(self.source_file_list))
        

        self.rotation_level = kwargs.get('rotation_level', 0) # Default to no rotation (0)
        self.zoom_level = kwargs.get('zoom_level', 1.0)  # Default to no zoom (1.0)
        self.flip = kwargs.get('flip', 0)  # Default to no flip

        self.create_dataset(dimension=dimension)

        ifsave = None if paths is None else True
        self.finalcheck(ifsave=ifsave,ifcheck=False,iftest_volumes_pixdim=False)

    def get_loader(self):
        self.source_file_list = []
        self.train_ds=[]
        self.val_ds=[]