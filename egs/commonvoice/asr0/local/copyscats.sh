

do_scatter=true
do_prev=false

utils/parse_options.sh || exit 1

echo "copying data -> prev ${do_prev}"
if [ ${do_prev} = true ]; then
  cp dump/test/deltafalse/data.json dump/test/deltafalse/prev.json
  cp dump/train_nodev/deltafalse/data.json dump/train_nodev/deltafalse/prev.json
  cp dump/train_dev/deltafalse/data.json dump/train_dev/deltafalse/prev.json
fi

if [ ${do_scatter} = true ]; then
  echo "prepring scatter = ${do_scatter}"
  cp dump/test/deltafalse/scat.json dump/test/deltafalse/data.json
  cp dump/train_nodev/deltafalse/scat.json dump/train_nodev/deltafalse/data.json
  cp dump/train_dev/deltafalse/scat.json dump/train_dev/deltafalse/data.json
else
  echo "bypassing scatter ${do_scatter}"
  cp dump/test/deltafalse/prev.json dump/test/deltafalse/data.json
  cp dump/train_nodev/deltafalse/prev.json dump/train_nodev/deltafalse/data.json
  cp dump/train_dev/deltafalse/prev.json dump/train_dev/deltafalse/data.json
fi
