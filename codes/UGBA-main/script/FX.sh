#!/usr/bin/env bash

set -euo pipefail

# Create FX directory structure mirroring logs/
mkdir -p FX/{Cora,Pubmed,Flickr,Arxiv}

# echo "[Cora] Starting None/Prune/Isolate runs..."
# # Cora - None
# nohup python -u run_adaptive.py \
#   --dataset=Cora --prune_thr=0.1 --homo_loss_weight=50 \
#   --vs_number=10 --test_model=GCN --defense_mode=none \
#   --selection_method=cluster_degree --homo_boost_thrd=0.5 \
#   --epochs=200 --trojan_epochs=400 \
#   > "FX/Cora/*Cora_Ours_None_gcn2gcn_clu_vs10.log" 2>&1 &

# # Cora - Prune
# nohup python -u run_adaptive.py \
#   --dataset=Cora --prune_thr=0.1 --homo_loss_weight=50 \
#   --vs_number=10 --test_model=GCN --defense_mode=prune \
#   --selection_method=cluster_degree --homo_boost_thrd=0.5 \
#   --epochs=200 --trojan_epochs=400 \
#   > "FX/Cora/*Cora_Ours_Prune_gcn2gcn_clu_vs10.log" 2>&1 &

# # Cora - Isolate (= Prune+LD)
# nohup python -u run_adaptive.py \
#   --dataset=Cora --prune_thr=0.1 --homo_loss_weight=50 \
#   --vs_number=10 --test_model=GCN --defense_mode=isolate \
#   --selection_method=cluster_degree --homo_boost_thrd=0.5 \
#   --epochs=200 --trojan_epochs=400 \
#   > "FX/Cora/*Cora_Ours_Isolate_gcn2gcn_clu_vs10.log" 2>&1 &

# echo "[Pubmed] Starting None/Prune/Isolate runs..."
# #Pubmed - None
# nohup python -u run_adaptive.py \
#   --dataset=Pubmed --prune_thr=0.2 --homo_loss_weight=100 \
#   --vs_number=40 --test_model=GCN --defense_mode=none \
#   --selection_method=cluster_degree --homo_boost_thrd=0.5 \
#   --epochs=200 --trojan_epochs=2000 \
#   > "FX/Pubmed/*Pubmed_Ours_None_gcn2gcn_clu_vs40_homo200_epoch1500.log" 2>&1 &

# # Pubmed - Prune
# nohup python -u run_adaptive.py \
#   --dataset=Pubmed --prune_thr=0.2 --homo_loss_weight=100 \
#   --vs_number=40 --test_model=GCN --defense_mode=prune \
#   --selection_method=cluster_degree --homo_boost_thrd=0.5 \
#   --epochs=200 --trojan_epochs=2000 \
#   > "FX/Pubmed/*Pubmed_Ours_Prune_gcn2gcn_clu_vs40_homo50.log" 2>&1 &

# # Pubmed - Isolate
# nohup python -u run_adaptive.py \
#   --dataset=Pubmed --prune_thr=0.2 --homo_loss_weight=100 \
#   --vs_number=40 --test_model=GCN --defense_mode=isolate \
#   --selection_method=cluster_degree --homo_boost_thrd=0.5 \
#   --epochs=200 --trojan_epochs=2000 \
#   > "FX/Pubmed/*Pubmed_Ours_Isolate_gcn2gcn_clu_vs40.log" 2>&1 &

echo "[Flickr] Starting None/Prune/Isolate runs..."
# Flickr - None
nohup python -u run_adaptive.py \
  --dataset=Flickr --prune_thr=0.4 --hidden=64 \
  --homo_loss_weight=100 --vs_number=80 --test_model=GCN \
  --defense_mode=none --selection_method=cluster_degree \
  --homo_boost_thrd=0.8 --epochs=200 --trojan_epochs=400 \
  > "FX/Flickr/*Flickr_Ours_None_gcn2gcn_clu_vs80_epoch1000.log" 2>&1 &

# # Flickr - Prune
# nohup python -u run_adaptive.py \
#   --dataset=Flickr --prune_thr=0.4 --hidden=64 \
#   --homo_loss_weight=100 --vs_number=80 --test_model=GCN \
#   --defense_mode=prune --selection_method=cluster_degree \
#   --homo_boost_thrd=0.8 --epochs=200 --trojan_epochs=400 \
#   > "FX/Flickr/*Flickr_Ours_Prune_gcn2gcn_clu_vs80.log" 2>&1 &

# # Flickr - Isolate
# nohup python -u run_adaptive.py \
#   --dataset=Flickr --prune_thr=0.4 --hidden=64 \
#   --homo_loss_weight=100 --vs_number=80 --test_model=GCN \
#   --defense_mode=isolate --selection_method=cluster_degree \
#   --homo_boost_thrd=0.8 --epochs=200 --trojan_epochs=400 \
#   > "FX/Flickr/*Flickr_Ours_Isolate_gcn2gcn_clu_vs80.log" 2>&1 &

# echo "[Arxiv] Starting None/Prune/Isolate runs..."
# # OGBN-Arxiv - None
# nohup python -u run_adaptive.py \
#   --dataset=ogbn-arxiv --prune_thr=0.8 --homo_loss_weight=200 \
#   --vs_number=160 --test_model=GCN --defense_mode=none \
#   --selection_method=cluster_degree --homo_boost_thrd=0.8 \
#   --epochs=800 --trojan_epochs=800 \
#   > "FX/Arxiv/*Arxiv_Ours_None_gcn2gcn_clu_vs160.log" 2>&1 &

# # # OGBN-Arxiv - Prune
# nohup python -u run_adaptive.py \
#   --dataset=ogbn-arxiv --prune_thr=0.8 --homo_loss_weight=200 \
#   --vs_number=160 --test_model=GCN --defense_mode=prune \
#   --selection_method=cluster_degree --homo_boost_thrd=0.8 \
#   --epochs=800 --trojan_epochs=800 \
#   > "FX/Arxiv/*Arxiv_Ours_Prune_gcn2gcn_clu_vs160.log" 2>&1 &

# # OGBN-Arxiv - Isolate
# nohup python -u run_adaptive.py \
#   --dataset=ogbn-arxiv --prune_thr=0.8 --homo_loss_weight=200 \
#   --vs_number=160 --test_model=GCN --defense_mode=isolate \
#   --selection_method=cluster_degree --homo_boost_thrd=0.8 \
#   --epochs=800 --trojan_epochs=800 \
#   > "FX/Arxiv/*Arxiv_Ours_Isolate_gcn2gcn_clu_vs160.log" 2>&1 &

# echo "All runs started in background. Use: tail -f FX/*/*.log"

