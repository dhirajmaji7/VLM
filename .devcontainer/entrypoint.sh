#!/bin/bash
echo "==============================="
echo "🚀 ENTRYPOINT SCRIPT RUNNING!"
echo "==============================="

# pip install --upgrade pip==23.1.2
# pip install scipy==1.7.3
# pip install scikit-image==0.20.0
# cd /home/dmaji/perception/mmdetection3d
# pip install -v -e .
pip install pandas pyarrow

git config --global user.email "dhirajmaji7@gmail.com"
git config --global user.name "Dhiraj Maji"

exec /bin/bash