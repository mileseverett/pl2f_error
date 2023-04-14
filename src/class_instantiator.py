from class_importer import import_class

def instantiate_class_from_config(config: dict):
    class_path = config['class_path']
    init_args = config.get('init_args', {})

    cls = import_class(class_path)
    instance = cls(**init_args)

    return instance
