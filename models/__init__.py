def create_model(opt):
    model = None
    if opt.model == 'cycle_gan':
        assert(opt.dataset_mode == 'unaligned')
        from .cycle_gan_model import CycleGANModel
        # from .Dec_model import CycleGqANModel
        model = CycleGANModel()
    elif opt.model == 'G_dec':
        assert(opt.dataset_mode == 'unaligned')
        from .cycle_gan_model_pre import CycleGANModel
        # from .Dec_model import CycleGqANModel
        model = CycleGANModel()
    elif opt.model == 'test':
        assert(opt.dataset_mode == 'single')
        from .test_model import TestModel
        model = TestModel()
    else:
        raise NotImplementedError('model [%s] not implemented.' % opt.model)
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model
