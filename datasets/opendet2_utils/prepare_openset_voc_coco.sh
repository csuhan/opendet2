DATA_DIR=datasets/voc_coco
COCO_DIR=datasets/coco
VOC07_DIR=datasets/VOC2007
VOC12_DIR=datasets/VOC2012

# make neccesary dirs
rm $DATA_DIR -rf
echo "make dirs"
mkdir -p $DATA_DIR
mkdir -p $DATA_DIR/Annotations
# mkdir -p DATA_DIR/JPEGImages
mkdir -p $DATA_DIR/ImageSets
mkdir -p $DATA_DIR/ImageSets/Main

# cp data
# make use you have $COCO_DIR, VOC07_DIR and VOC12_DIR
echo "copy coco images"
cp $COCO_DIR/train2017 $DATA_DIR/JPEGImages -r
cp $COCO_DIR/val2017/* $DATA_DIR/JPEGImages/

echo "convert coco annotation to voc"
python datasets/opendet2_utils/convert_coco_to_voc.py --dir $DATA_DIR --ann_path $COCO_DIR/annotations/instances_train2017.json
python datasets/opendet2_utils/convert_coco_to_voc.py --dir $DATA_DIR --ann_path $COCO_DIR/annotations/instances_val2017.json


# generate imageset
echo "generate coco sub imagesets"
# class incremental settings
# 20-40
python datasets/opendet2_utils/prepare_openset_voc_coco_cls_specific.py --dir $DATA_DIR --in_split instances_train2017 --out_split instances_train2017_cls_spe_20_40 --start_class 20 --end_class 40 --pre_num_sample 8000 --post_num_sample 5000
# 40-60
python datasets/opendet2_utils/prepare_openset_voc_coco_cls_specific.py --dir $DATA_DIR --in_split instances_train2017 --out_split instances_train2017_cls_spe_20_60 --start_class 20 --end_class 60 --pre_num_sample 16000 --post_num_sample 10000
# 60-80
python datasets/opendet2_utils/prepare_openset_voc_coco_cls_specific.py --dir $DATA_DIR --in_split instances_train2017 --out_split instances_train2017_cls_spe_20_80 --start_class 20 --end_class 80 --pre_num_sample 24000 --post_num_sample 15000

# image incremental settings
# 2500
python datasets/opendet2_utils/prepare_openset_voc_coco_cls_agnostic.py --dir $DATA_DIR --in_split instances_train2017 --out_split instances_train2017_cls_agn_2500 --start_class 20 --end_class 80  --post_num_sample 2500
# 5000
python datasets/opendet2_utils/prepare_openset_voc_coco_cls_agnostic.py --dir $DATA_DIR --in_split instances_train2017 --out_split instances_train2017_cls_agn_5000 --start_class 20 --end_class 80  --post_num_sample 5000
# 10000
python datasets/opendet2_utils/prepare_openset_voc_coco_cls_agnostic.py --dir $DATA_DIR --in_split instances_train2017 --out_split instances_train2017_cls_agn_10000 --start_class 20 --end_class 80  --post_num_sample 10000
# 20000
python datasets/opendet2_utils/prepare_openset_voc_coco_cls_agnostic.py --dir $DATA_DIR --in_split instances_train2017 --out_split instances_train2017_cls_agn_20000 --start_class 20 --end_class 80  --post_num_sample 20000


echo "copy voc images"
cp $VOC07_DIR/JPEGImages/* $DATA_DIR/JPEGImages/
cp $VOC12_DIR/JPEGImages/* $DATA_DIR/JPEGImages/

echo "copy voc annotation"
cp $VOC07_DIR/Annotations/* $DATA_DIR/Annotations/
cp $VOC12_DIR/Annotations/* $DATA_DIR/Annotations/

echo "copy voc imagesets"
cp $VOC07_DIR/ImageSets/Main/train.txt $DATA_DIR/ImageSets/Main/voc07train.txt
cp $VOC07_DIR/ImageSets/Main/val.txt $DATA_DIR/ImageSets/Main/voc07val.txt
cp $VOC07_DIR/ImageSets/Main/test.txt $DATA_DIR/ImageSets/Main/voc07test.txt
cp $VOC12_DIR/ImageSets/Main/trainval.txt $DATA_DIR/ImageSets/Main/voc12trainval.txt

echo "generate voc_coco_val imagesets"
cat $DATA_DIR/ImageSets/Main/voc07val.txt > $DATA_DIR/ImageSets/Main/voc_coco_val.txt
cat $DATA_DIR/ImageSets/Main/instances_val2017.txt >> $DATA_DIR/ImageSets/Main/voc_coco_val.txt

echo "generate voc_coco_20_40_test imagesets"
cat $DATA_DIR/ImageSets/Main/voc07test.txt > $DATA_DIR/ImageSets/Main/voc_coco_20_40_test.txt
cat $DATA_DIR/ImageSets/Main/instances_train2017_cls_spe_20_40.txt >> $DATA_DIR/ImageSets/Main/voc_coco_20_40_test.txt

echo "generate voc_coco_40_60_test imagesets"
cat $DATA_DIR/ImageSets/Main/voc07test.txt > $DATA_DIR/ImageSets/Main/voc_coco_20_60_test.txt
cat $DATA_DIR/ImageSets/Main/instances_train2017_cls_spe_20_60.txt >> $DATA_DIR/ImageSets/Main/voc_coco_20_60_test.txt

echo "generate voc_coco_60_80_test imagesets"
cat $DATA_DIR/ImageSets/Main/voc07test.txt > $DATA_DIR/ImageSets/Main/voc_coco_20_80_test.txt
cat $DATA_DIR/ImageSets/Main/instances_train2017_cls_spe_20_80.txt >> $DATA_DIR/ImageSets/Main/voc_coco_20_80_test.txt

echo "generate voc_coco_2500_test imagesets"
cat $DATA_DIR/ImageSets/Main/voc07test.txt > $DATA_DIR/ImageSets/Main/voc_coco_2500_test.txt
cat $DATA_DIR/ImageSets/Main/instances_train2017_cls_agn_2500.txt >> $DATA_DIR/ImageSets/Main/voc_coco_2500_test.txt

echo "generate voc_coco_5000_test imagesets"
cat $DATA_DIR/ImageSets/Main/voc07test.txt > $DATA_DIR/ImageSets/Main/voc_coco_5000_test.txt
cat $DATA_DIR/ImageSets/Main/instances_train2017_cls_agn_5000.txt >> $DATA_DIR/ImageSets/Main/voc_coco_5000_test.txt

echo "generate voc_coco_10000_test imagesets"
cat $DATA_DIR/ImageSets/Main/voc07test.txt > $DATA_DIR/ImageSets/Main/voc_coco_10000_test.txt
cat $DATA_DIR/ImageSets/Main/instances_train2017_cls_agn_10000.txt >> $DATA_DIR/ImageSets/Main/voc_coco_10000_test.txt

echo "generate voc_coco_20000_test imagesets"
cat $DATA_DIR/ImageSets/Main/voc07test.txt > $DATA_DIR/ImageSets/Main/voc_coco_20000_test.txt
cat $DATA_DIR/ImageSets/Main/instances_train2017_cls_agn_20000.txt >> $DATA_DIR/ImageSets/Main/voc_coco_20000_test.txt

