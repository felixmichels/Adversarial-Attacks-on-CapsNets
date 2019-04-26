#/bin/sh

dirs=('logdir' 'data' 'ckpt' 'datasets')

for i in ${dirs[@]}; do
	mkdir -p "$i"
	if [[ ! -z "$2" ]]; then
		rm -r "$i/$1/$2"
	fi
done
