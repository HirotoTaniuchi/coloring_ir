from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


def get_lines_from_text( fntype:str='train.txt',
                         pdata:Path=Path('/home/usrs/taniuchi/dataset/ir_seg_dataset/') ) -> list():
    # Read text files
    fn = pdata / fntype  # Absolute path for train.txt
    with fn.open() as _f:  
        lines = _f.readlines()  # Get all lines
    lines = [ l.strip() for l in lines]  # Remove \n from each line
    lines = [ l for l in lines if not 'flip' in l]  # Take line if flip is not included
    return lines


def main():
    # Set folders
    pdata = Path('/home/usrs/taniuchi/dataset/ir_seg_dataset/')
    pimg = pdata / 'images'
    plab = pdata / 'labels'

    # Get label filenames
    paths = plab.glob('*.png')  # Absolute path for label filenames
    fns = [p.name for p in paths]  # Only filenames, e.g., XXXX.png

    # Read text files
    fn_train = pdata / 'train.txt'  # Absolute path for train.txt
    with fn_train.open() as _f:  
        lines = _f.readlines()  # Get all lines
    lines = [ l.strip() for l in lines]  # Remove \n from each line
    lines = [ l for l in lines if not 'flip' in l]  # Take line if flip is not included
    #print(lines)  

    lines2 = get_lines_from_text( 'train.txt', pdata )

    # Show image and label
    fn = pimg / (lines2[0] + '.png')
    im = Image.open(fn)
    breakpoint()
    img = np.asarray(im)

    fig, axs = plt.subplots(1,2)
    axs[0].imshow(img[:,:,0:3])
    axs[1].imshow(img[:,:,3])
    plt.show()

    
if __name__=='__main__':
    main()
