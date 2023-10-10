mkdir pretrained_models
cd pretrained_models


# Download zip dataset from Google Drive
filename='model_best.pth'
fileid='1U1_EDes8Lhu5kciMLLSS82gWKObnRX9H'
wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=${fileid}' -O- | sed -rn 's/.confirm=([0-9A-Za-z_]+)./\1\n/p')&id=${fileid}" -O ${filename} && rm -rf /tmp/cookies.txt