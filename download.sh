echo "Downloading pretrained model..."
wget http://nsvqa.csail.mit.edu/assets/pretrained.zip
unzip pretrained.zip
rm pretrained.zip

mkdir mask_rcnn
cd mask_rcnn
echo "Downloading pretrained weights for Mask R-CNN backbone..."
wget http://nsvqa.csail.mit.edu/assets/backbones.zip
unzip backbones.zip
rm backbones.zip

cd ..  # root