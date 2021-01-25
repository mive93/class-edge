echo "Downloading caches and precision matrices"
wget https://cloud.hipert.unimore.it/s/EqFqKqQMk6okeEy/download -O ../data/camera_caches.zip
unzip -d ../data/ ../data/camera_caches.zip
rm ../data/camera_caches.zip