import os

def get_files_folder(folder_path) -> list[str]:
    return [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.cvs')]


def get_urls():
    return [
        #"https://eu.davidaustinroses.com/blogs/rose-care/how-to-prune-english-shrub-and-climbing-roses-to-maximise-flowering",
        "https://eu.davidaustinroses.com/blogs/news/a-guide-to-pruning",  # roses
        "https://www.cvlibs.net/datasets/kitti"]

def get_urls2() -> list[str]:
    return [
    #"https://eu.davidaustinroses.com/blogs/rose-care/how-to-prune-english-shrub-and-climbing-roses-to-maximise-flowering", #roses
    #"https://eu.davidaustinroses.com/blogs/news/a-guide-to-pruning", #roses
    
    #"https://github.com/VisDrone/VisDrone-Dataset",
    #"https://paperswithcode.com/dataset/visdrone",
    "https://openaccess.thecvf.com/content_ECCVW_2018/papers/11133/Zhu_VisDrone-DET2018_The_Vision_Meets_Drone_Object_Detection_in_Image_Challenge_ECCVW_2018_paper.pdf",
    "https://github.com/VisDrone/VisDrone2018-MOT-toolkit",
    "https://en.wikipedia.org/wiki/Object_detection",
    "https://en.wikipedia.org/wiki/Computer_vision",
    "https://en.wikipedia.org/wiki/Convolutional_neural_network",
    "https://en.wikipedia.org/wiki/Unmanned_aerial_vehicle",
    "https://www.faa.gov/uas",
    "https://www.tensorflow.org",
    "https://pytorch.org",
    "https://keras.io",
    "https://arxiv.org/abs/1804.06985",
    "https://arxiv.org/abs/2202.11983",
    "https://motchallenge.net",
    "https://www.cvlibs.net/datasets/kitti",
    "https://www.dronedeploy.com",
    "https://www.dji.com",
    "https://arxiv.org",
    "https://openaccess.thecvf.com",
    "https://roboflow.com",
    "https://www.kaggle.com",
    "https://paperswithcode.com",
    "https://github.com"]