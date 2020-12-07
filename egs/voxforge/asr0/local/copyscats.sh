

do_scatter=true
do_prev=false


utils/parse_options.sh || exit 1

echo "copying data -> prev ${do_prev}"
if [ ${do_prev} = true ]; then
  cp dump/$4/deltafalse/$1.json dump/$4/deltafalse/prev.json
  cp dump/$2/deltafalse/$1.json dump/$2/deltafalse/prev.json
  cp dump/$3/deltafalse/$1.json dump/$3/deltafalse/prev.json
fi

if [ ${do_scatter} = true ]; then
  echo "prepring scatter = ${do_scatter}"
  cp dump/$4/deltafalse/scat.json dump/$4/deltafalse/$1.json
  cp dump/$2/deltafalse/scat.json dump/$2/deltafalse/$1.json
  cp dump/$3/deltafalse/scat.json dump/$3/deltafalse/$1.json
else
  echo "bypassing scatter ${do_scatter}"
  cp dump/$4/deltafalse/prev.json dump/$4/deltafalse/$1.json
  cp dump/$2/deltafalse/prev.json dump/$2/deltafalse/$1.json
  cp dump/$3/deltafalse/prev.json dump/$3/deltafalse/$1.json
fi
