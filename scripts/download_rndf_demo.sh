if [ -z $MIMO_DIR ]; then echo 'Please source "mimo_env.sh" first'
else
  gdown 'https://docs.google.com/uc?export=download&id=13rgepmqdN3pvn2OaAyvJND84zouP2Xue' -O data.tgz
  tar -xvzf data.tgz -C $MIMO_DIR/eval/rndf
  rm data.tgz
  echo 'Finished!'
fi
