if [ -z $MIMO_DIR ]; then echo 'Please source "mimo_env.sh" first'
else
  gdown 'https://docs.google.com/uc?export=download&id=1rj4FNV5-cr7gHd0jjfvxyaEarD901aF1' -O data.tgz
  tar -xvzf data.tgz -C $MIMO_DIR/eval/ndf
  rm data.tgz
  echo 'Finished!'
fi
