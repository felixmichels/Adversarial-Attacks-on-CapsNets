#/bin/sh

dirs=('logdir' 'data' 'ckpt' 'datasets')

for i in ${dirs[@]}; do
	mkdir -p "$i"
	if [[ ! -z "$1" ]]; then
		rm -r "$i/$1"
	fi
done
