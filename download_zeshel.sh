fileid="1ZcKZ1is0VEkY9kNfPxIG19qEIqHE5LIO"
filename="zeshel.tar.bz2"

curl "https://drive.usercontent.google.com/download?id=${fileid}&confirm=xxx" -o ${filename}
tar -xf $filename
