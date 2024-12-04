import os


def get_project_root():
    """Get absolute path to project root directory"""
    current_path = os.path.abspath(__file__)
    # Go up 3 levels: utils -> src -> school_chatbot
    return os.path.dirname(os.path.dirname(os.path.dirname(current_path)))


def get_config_path():
    """Get absolute path to config.yaml"""
    return os.path.join(get_project_root(), "config", "config.yaml")
