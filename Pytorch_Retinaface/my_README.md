reference -- https://github.com/biubug6/Pytorch_Retinaface
to download pretrained models - https://drive.google.com/open?id=1oZRSG0ZegbVkVwUd8wUIQx8W7yfZ_ki1
retina face detection cropping command
change images_path according to the requirement (images_path = glob.glob(testset_folder+'/*/*') - crop image in the subfolders)
python ours_test_widerface.py -m weights/mobilenet0.25_Final.pth --dataset_folder folder path
