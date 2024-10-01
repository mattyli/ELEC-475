from os import listdir
from PIL import Image

def main():
    
    for filename in listdir('images/'):
        if filename.endswith('.jpg'):
            try:
                with Image.open('images/'+filename) as img:
                    img.verify() # verify that it is, in fact an image
            except (IOError, SyntaxError) as e:
                print('Bad file:', filename) # print out the names of corrupt files

if __name__ == '__main__':
    main()