import torch

def get_desired_parameter_names(model, desired_layer_type):
    """
    Returns the names of the model parameters that are inside a desired layer type.
    """
    result = []
    for name, module in model.named_modules():
        if isinstance(module, tuple(desired_layer_type)):
            tmp_results = [f'{name}.{n}' for n in module._parameters.keys()]
            result += tmp_results
    return result

def get_parameter_name_grouped(model):
    ...



def distance_between_models(model1, model2, distance_type):
    def distance_between_models_l2(model1, model2):
        """
        Returns the l2 distance between two models.
        """
        distance = 0
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            distance += torch.norm(p1 - p2)
        return distance.item()
    
    def distance_between_models_cosine(model1, model2):
        """
        Returns the cosine distance between two models.
        """
        distance = 0
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            distance += 1 - torch.cosine_similarity(p1.flatten(), p2.flatten())
        return distance.item()

    """
    Returns the distance between two models.
    """
    if distance_type == 'l2':
        return distance_between_models_l2(model1, model2)
    elif distance_type == 'cosine':
        return distance_between_models_cosine(model1, model2)
    else:
        raise ValueError(f'Distance type {distance_type} not supported.')
    


