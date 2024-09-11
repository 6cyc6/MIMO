if [ -z $MIMO_DIR ]; then echo 'Please source "mimo_env.sh" first'
else
  wget --no-check-certificate -r 'https://drive.google.com/uc?export=download&id=1buAWXe-E8qTZTuUTpA4EiM76y8C32py0' -O model_weights.tar
  tar -xvf model_weights.tar -C $MIMO_DIR
  rm model_weights.tar
fi
