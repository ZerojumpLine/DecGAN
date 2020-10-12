def create_model(opt):
    model = None
    if opt.model == 'dec_gan':
        assert(opt.dataset_mode == 'unaligned')
        from .dec_gan_model import DecGANModel
        model = DecGANModel()
    elif opt.model == 'G_dec':
        assert(opt.dataset_mode == 'unaligned')
        from .dec_model import DecModel
        model = DecModel()
    elif opt.model == 'test':
        assert(opt.dataset_mode == 'single')
        from .test_model import TestModel
        model = TestModel()
    else:
        raise NotImplementedError('model [%s] not implemented.' % opt.model)
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model
