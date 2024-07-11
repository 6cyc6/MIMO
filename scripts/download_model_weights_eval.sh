if [ -z $MFIM_DIR ]; then echo 'Please source "mimo_env.sh" first'
else
  wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=1buAWXe-E8qTZTuUTpA4EiM76y8C32py0' -O model_weights.tar
  tar -xvf model_weights.tar -C $MFIM_DIR/eval/ndf
  tar -xvf model_weights.tar -C $MFIM_DIR/eval/rndf
  rm model_weights.tar
  echo 'Finished!'
fi
