# dms-mask-extraction

Adding code for dms mask extraction previous works done y Aman k, Madhevan A, Vinsent P

All the weight files are found under the  account google cloud bucket

In server aadhar-document-extractor
$export PYTHONPATH=$PYTHONPATH:/home/ofsdms/Lochan/Mask_RCNN/
$cd Lochan

1. python3 ~/dms-mask-extraction/Lochan/deploy_v3_aadhar-document-extractor.py

2. python3 ~/dms-mask-extraction/Lochan/deploy_v3_aadhar-text-strip-extractor.py 

3. python3 ~/dms-mask-extraction/Lochan/doc-classifier_deploy.py

4. python3 ~/dms-mask-extraction/Lochan/Orientation_API.py

5. python3 ~/dms-mask-extraction/Lochan/deploy_v3_mrcnn-deployment-final.py

6. python3 ~/dms-mask-extraction/Lochan/deploy_v3_pan-text-strip-extractor.py




