shapenetroot=$1
cwd=$(pwd)
binvoxcmd=$cwd'/binvox'
echo $binvoxcmd
#for syn in {'02691156','03001627','02958343'}; do
for syn in {'00000001','00000002'}; do
	cd $shapenetroot'/'$syn
	for modeldir in */ ; do
		cd $modeldir
		$binvoxcmd -d 64 -cb -e model.obj
		cd ../
	done
	cd ../../
done
