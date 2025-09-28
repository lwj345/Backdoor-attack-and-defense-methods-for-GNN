#cora
# python train_gcond_transduct.py --dataset cora --nlayers=2 --sgc=1 --lr_feat=1e-4 --gpu_id=0  --lr_adj=1e-4 --r=0.25  --seed=1 --epoch=1000 --save=1 --lr_trigger 0.5 --trigger_size 4 --selector 'cluster' > logs/cora/cora-gcond-r025-cluster-20240606-for-save.txt &
# python train_gcond_transduct.py --dataset cora --nlayers=2 --sgc=1 --lr_feat=1e-4 --gpu_id=1  --lr_adj=1e-4 --r=0.5  --seed=1 --epoch=1000 --save=1 --lr_trigger 0.5 --trigger_size 4 --selector 'cluster' > logs/cora/cora-gcond-r05-cluster-20240606-for-save.txt &
# python train_gcond_transduct.py --dataset cora --nlayers=2 --sgc=1 --lr_feat=1e-4 --gpu_id=2  --lr_adj=1e-4 --r=1.0  --seed=1 --epoch=1000 --save=1 --lr_trigger 0.5 --trigger_size 4 --selector 'cluster' > logs/cora/cora-gcond-r1-cluster-20240606-for-save.txt &

#citeseer###########
# python train_gcond_transduct.py --dataset citeseer --nlayers=2 --sgc=1 --lr_feat=3e-4 --gpu_id=2 --lr_adj=1e-4 --r=0.25  --seed=1 --epoch=750 --save=1 --lr_trigger 0.1 --trigger_size 4 --selector 'cluster' > logs/citeseer/citeseer-gcond-r025-cluster-20240609-for-save-lr_feat3e-4.txt &
# python train_gcond_transduct.py --dataset citeseer --nlayers=2 --sgc=1 --lr_feat=1e-4 --gpu_id=4  --lr_adj=1e-4 --r=0.5  --seed=1 --epoch=1000 --save=1 --lr_trigger 0.1 --trigger_size 4 --selector 'cluster' > logs/citeseer/citeseer-gcond-r05-cluster-20240606-for-save.txt &
# python train_gcond_transduct.py --dataset citeseer --nlayers=2 --sgc=1 --lr_feat=7e-5 --gpu_id=5 --lr_adj=1e-4 --r=1.0  --seed=1 --epoch=200 --save=1 --lr_trigger 0.1 --trigger_size 4 --selector 'cluster' > logs/citeseer/citeseer-gcond-r1-cluster-20240609-for-save-lr_feat7e-5.txt &

# flickr
# python train_gcond_induct.py --dataset flickr --sgc=2 --nlayers=2 --lr_feat=0.005 --lr_adj=0.01  --r=0.01 --gpu_id=7 --seed=1 --epochs=1000  --inner=1 --outer=10 --save=1 --lr_trigger 0.1 --trigger_size 4 --selector 'cluster' > logs/flickr/flickr-gcond-r001-cluter-20240606-for-save.txt &
# python train_gcond_induct.py --dataset flickr --sgc=2 --nlayers=2 --lr_feat=0.007 --lr_adj=0.01  --r=0.005 --gpu_id=0 --seed=1 --epochs=1000  --inner=1 --outer=10 --save=1 --lr_trigger 0.01 --trigger_size 4 --selector 'cluster' > logs/flickr/flickr-gcond-r0005-cluter-20240606-for-save-lr_trigger001-lr_feat00007.txt &
# python train_gcond_induct.py --dataset flickr --sgc=2 --nlayers=2 --lr_feat=0.005 --lr_adj=0.01  --r=0.001 --gpu_id=2 --seed=1 --epochs=1000  --inner=1 --outer=10 --save=1 --lr_trigger 0.1 --trigger_size 4 --selector 'cluster' > logs/flickr/flickr-gcond-r0001-cluter-20240606-for-save.txt &


#reddit
# #0.0005
# python train_gcond_induct.py --dataset reddit --sgc=1 --nlayers=2 --lr_feat=0.1 --lr_adj=0.1  --r=0.0005 --gpu_id=3 --seed=1 --epochs=1000  --inner=1 --outer=10 --save=1 --lr_trigger 0.01 --trigger_size 4 --selector 'cluster' > logs/reddit/reddit-gcond-r00005-cluter-20240609-lr_trigger001-lr_feat01-for-save.txt &
# python train_gcond_induct.py --dataset reddit --sgc=1 --nlayers=2 --lr_feat=0.2 --lr_adj=0.1  --r=0.0005 --gpu_id=1 --seed=1 --epochs=1000  --inner=1 --outer=10 --save=1 --lr_trigger 0.01 --trigger_size 4 --selector 'cluster' > logs/reddit/reddit-gcond-r00005-cluter-20240609-lr_trigger001-lr_feat02-for-save-seed1.txt &
# # #0.001
# python train_gcond_induct.py --dataset reddit --sgc=1 --nlayers=2 --lr_feat=0.1 --lr_adj=0.1  --r=0.001 --gpu_id=3 --seed=1 --epochs=1000  --inner=1 --outer=10 --save=1 --lr_trigger 0.01 --trigger_size 4 --selector 'cluster' > logs/reddit/reddit-gcond-r0001-cluter-20240609-lr_trigger001-lr_feat01-for-save-seed1.txt &
# # #0.002
# python train_gcond_induct.py --dataset reddit --sgc=1 --nlayers=2 --lr_feat=0.1 --lr_adj=0.1  --r=0.002 --gpu_id=3 --seed=1 --epochs=1000  --inner=1 --outer=10 --save=1 --lr_trigger 0.008 --trigger_size 4 --selector 'cluster' > logs/reddit/reddit-gcond-r0002-cluter-20240609-lr_trigger0008-lr_feat01-for-save-seed1.txt &
