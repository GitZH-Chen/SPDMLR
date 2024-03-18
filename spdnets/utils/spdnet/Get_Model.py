from spdnets.models.spdnet import SPDNet

def get_model(args):
    model = SPDNet(args)
    print(model)
    return model.double()