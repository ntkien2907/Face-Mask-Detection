from utils import resize_all_images, remove_file_with_extension

mydir = 'data/'

resize_all_images(dir=mydir, new_size=(128,128))

remove_file_with_extension(path=mydir+'incorrectly_mask/', extension='.jpg')
remove_file_with_extension(path=mydir+'with_mask/'       , extension='.jpg')
remove_file_with_extension(path=mydir+'without_mask/'    , extension='.jpg')
