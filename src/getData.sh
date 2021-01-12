#!/bin/bash
gsutil cp -r gs://uga-dsp/project3/neurofinder.00.00.test.zip .

for l in $(ls *zip)
do
    unzip $l
done
exit 0