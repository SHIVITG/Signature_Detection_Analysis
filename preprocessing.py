import os

orig_groups = []
forg_groups = []
path_hindi = "input_data/BHSig260/Hindi/"
path_bengali = "input_data/BHSig260/Bengali/"

def fetch_groups():
    print("Report: Dictionary of forged and original images creating")
    print("")
    create_signature_groups(path_hindi)
    create_signature_groups(path_bengali)
    print("")
    print("Report: Dictionary of forged and original images created")
    print("")
    print("Details of orig_groups and forg_groups:-")
    print("The length of orig_groups:",len(orig_groups))
    print("The length of forg_groups:",len(forg_groups))
    return orig_groups, forg_groups

def create_signature_groups(path):
    dir_list = next(os.walk(path))[1]
    dir_list.sort()
    for directory in dir_list:
        images = os.listdir(path+directory)
        images.sort()
        images = [path+directory+'/'+x for x in images]
        forg_groups.append(images[:30]) # First 30 signatures in each folder are forrged
        orig_groups.append(images[30:])
    print("Images from {} appended sucessfully:".format(path))
    
def main():
    print("My main method")

    
if __name__ == "__main__":
    fetch_groups()