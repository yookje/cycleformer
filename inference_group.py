import os
import glob
import time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
import torch.distributed as dist

from omegaconf import OmegaConf

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from torch.utils.data import DataLoader, Dataset

#modified for pickle
import re
from dataset_pickle import TSPDataset, collate_fn, make_tgt_mask
#from dataset_inference_pickle import TSPDataset as TSPDataset_Val
#from dataset_inference_pickle import collate_fn as collate_fn_val

#import pickle
from dataset_inference import TSPDataset as TSPDataset_Val
from dataset_inference import collate_fn as collate_fn_val

from model import make_model, subsequent_mask
from loss import SimpleLossCompute, LabelSmoothing

def rate(step, model_size, factor, warmup):
    """
    we have to default the step to 1 for LambdaLR function
    to avoid zero raising to negative power.
    """
    if step == 0:
        step = 1
    return factor * (model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5)))

from discord_webhook import DiscordWebhook, DiscordEmbed

def end_discord(v_num=None,  node_size = None, G = None, G_hitratio = None, G_optgap = None,  elapsed_time2 = None):
    url = "https://discord.com/api/webhooks/1334379289889472542/wltQ21o5e9e3S5TQ7xMYLv71GLmRcXeihEWTY6tGW_cR64zZvGrP3SI9cmHaRDcxeAsW"
    webhook = DiscordWebhook(url=url)

    embed = DiscordEmbed(title="Inference End", description=f"version_", color="03b2f8")
    embed.set_author(name="Jieun Yook")
    embed.set_timestamp()
    
    embed.add_embed_field(name=f"node_size : {node_size}", value="G = "+str(G), inline = False)

    embed.add_embed_field(name=f"hitratio", value=str((G_hitratio,4)), inline = False)
    embed.add_embed_field(name=f"optgap", value=str((G_optgap,4)), inline = False)
    embed.add_embed_field(name="total test time (s)", value=str((elapsed_time2,4)), inline = False)

    webhook.add_embed(embed)
    response = webhook.execute()

class TSPModel(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.model = make_model(
            src_sz=cfg.node_size, 
            enc_num_layers = cfg.enc_num_layers,
            dec_num_layers = cfg.dec_num_layers,
            d_model=cfg.d_model, 
            d_ff=cfg.d_ff, 
            h=cfg.h, 
            dropout=cfg.dropout,
            encoder_pe = "2D",
            decoder_pe = "circular",
            decoder_lut = "memory",
        )
        self.automatic_optimization = False
        
        criterion = LabelSmoothing(size=cfg.node_size, smoothing=cfg.smoothing)
        
        self.loss_compute = SimpleLossCompute(self.model.generator, criterion, cfg.node_size)
        
        self.set_cfg(cfg)
        self.train_outputs = []
        
        self.val_optimal_tour_distances = []
        self.val_predicted_tour_distances = []
        
        self.test_corrects = []
        self.test_optimal_tour_distances = []
        self.test_predicted_tour_distances = []
        
    def set_cfg(self, cfg):
        self.cfg = cfg
        self.save_hyperparameters(cfg)  # save config file with pytorch lightening

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.cfg.lr*len(self.cfg.gpus), betas=self.cfg.betas, eps=self.cfg.eps)
        lr_scheduler = LambdaLR(
            optimizer=optimizer,
            lr_lambda=lambda step: rate(step, model_size=self.cfg.d_model, factor=self.cfg.factor, warmup=self.cfg.warmup),
        )
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}

    def train_dataloader(self):
        train_dataset = TSPDataset(self.cfg.train_data_path)
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size = self.cfg.train_batch_size, 
            shuffle = True, 
            collate_fn = collate_fn,
            pin_memory=True
        )
        return train_dataloader

    def val_dataloader(self):
        self.val_dataset = TSPDataset_Val(self.cfg.val_data_path)
        val_dataloader = DataLoader(
            self.val_dataset, 
            batch_size = self.cfg.val_batch_size, 
            shuffle = False, 
            collate_fn = collate_fn_val,
            pin_memory=True
        )
        return val_dataloader
    
    def training_step(self, batch):
        src = batch["src"]
        tgt = batch["tgt"]
        visited_mask = batch["visited_mask"]
        tgt_y = batch["tgt_y"]
        ntokens = batch["ntokens"]
        tgt_mask = batch["tgt_mask"]

        opt = self.optimizers() # manual backprop
        opt.zero_grad() # manual backprop
        
        self.model.train()
        out = self.model(src, tgt, tgt_mask) # [B, V, E]
        
        if self.cfg.comparison_matrix == "memory":
            comparison_matrix = self.model.memory
        elif self.cfg.comparison_matrix == "encoder_lut":
            comparison_matrix = self.model.encoder_lut
        elif self.cfg.comparison_matrix == "decoder_lut":
            comparison_matrix = self.model.decoder_lut
        else:
            assert False
        
        loss = self.loss_compute(out, tgt_y, visited_mask, ntokens, comparison_matrix) # check! 
        

        training_step_outputs = [l.item() for l in loss]
        self.train_outputs.extend(training_step_outputs)

        loss = loss.mean()
        self.manual_backward(loss) # manual backprop
        
        opt.step() # manual backprop

        if self.trainer.is_global_zero:
            self.log(
                name = "train_loss",
                value = loss,
                prog_bar = True,
            )
        
        return {"loss": loss}

    def on_train_epoch_start(self) -> None:
        if self.trainer.is_global_zero:
            self.train_start_time = time.time()
    
    def on_train_epoch_end(self):
        outputs = self.all_gather(sum(self.train_outputs))
        lengths = self.all_gather(len(self.train_outputs))
        self.train_outputs.clear()
        
        lr_scheduler = self.lr_schedulers() # manual backprop
        lr_scheduler.step() # manual backprop
        
        if self.trainer.is_global_zero:
            train_loss = outputs.sum() / lengths.sum()
            train_time = time.time() - self.train_start_time
            self.print(
                f"##############Train: Epoch {self.current_epoch}###################",
                "train_loss={:.03f}, ".format(train_loss),
                "train time={:.03f}".format(train_time),
                f"##################################################################\n",
            )
            
    def validation_step(self, batch, batch_idx):
        src = batch["src"]
        tgt = batch["tgt"]
        visited_mask = batch["visited_mask"]
        ntokens = batch["ntokens"]
        tgt_mask = batch["tgt_mask"]
        tsp_tours = batch["tsp_tours"]
        #for fuzzy
        src_fuzzy = batch["src_fuzzy"]

        
        batch_size = tsp_tours.shape[0]
        self.model.eval()
        with torch.no_grad():
            memory = self.model.encode(src, src_fuzzy)
            ys = tgt.clone()
            visited_mask = visited_mask.clone()
            for i in range(self.cfg.node_size - 1):
                # memory, tgt, tgt_mask
                tgt_mask = subsequent_mask(ys.size(1)).type(torch.bool).to(src.device)
                out = self.model.decode(memory, src_fuzzy , ys, tgt_mask)
                
                if self.cfg.comparison_matrix == "memory":
                    comparison_matrix = self.model.memory
                elif self.cfg.comparison_matrix == "encoder_lut":
                    comparison_matrix = self.model.encoder_lut
                elif self.cfg.comparison_matrix == "decoder_lut":
                    comparison_matrix = self.model.decoder_lut
                else:
                    assert False
                    
                prob = self.model.generator(out[:, -1].unsqueeze(1), visited_mask, comparison_matrix)
                
                _, next_word = torch.max(prob, dim=-1)
                next_word = next_word.squeeze(-1)
                
                visited_mask[torch.arange(batch_size), 0, next_word] = True
                
                ys = torch.cat([ys, next_word.unsqueeze(-1)], dim=1)
        
        
        optimal_tour_distance = self.get_tour_distance(src, tsp_tours)
        predicted_tour_distance = self.get_tour_distance(src, ys)
        
        result = {
            "optimal_tour_distance": optimal_tour_distance.tolist(),
            "predicted_tour_distance": predicted_tour_distance.tolist(),
            }    
        
        self.val_optimal_tour_distances.extend(result["optimal_tour_distance"])
        self.val_predicted_tour_distances.extend(result["predicted_tour_distance"])
        
        return result
    
    def get_tour_distance(self, graph, tour):
        # graph.shape = [B, N, 2]
        # tour.shape = [B, N]

        shp = graph.shape
        gathering_index = tour.unsqueeze(-1).expand(*shp)
        ordered_seq = graph.gather(dim = 1, index = gathering_index)
        rolled_seq = ordered_seq.roll(dims = 1, shifts = -1)
        segment_lengths = ((ordered_seq - rolled_seq) ** 2).sum(-1).sqrt() # [B, N]
        group_travel_distances = segment_lengths.sum(-1)
        return group_travel_distances

    def on_validation_epoch_start(self) -> None:
        if self.trainer.is_global_zero:
            self.validation_start_time = time.time()

    def on_validation_epoch_end(self):
        optimal_tour_distances = self.all_gather(sum(self.val_optimal_tour_distances))
        predicted_tour_distances = self.all_gather(sum(self.val_predicted_tour_distances))
        
        self.val_optimal_tour_distances.clear()
        self.val_predicted_tour_distances.clear()
        
        total = self.cfg.node_size * len(self.val_dataset)
        opt_gaps = (predicted_tour_distances - optimal_tour_distances) / optimal_tour_distances
        mean_opt_gap = opt_gaps.mean().item() * 100
       

        self.log(
            name = "opt_gap",
            value = mean_opt_gap,
            prog_bar = True,
            sync_dist=True
        )
        
        if self.trainer.is_global_zero:
            validation_time = time.time() - self.validation_start_time
            self.print(
                f"##############Validation: Epoch {self.current_epoch}##############",
                "validation time={:.03f}".format(validation_time),
                f"\ntotal={total}",
                f"\nmean_opt_gap = {mean_opt_gap}  %",
                f"##################################################################\n",
            )
            
    def test_dataloader(self):
        self.test_dataset = TSPDataset_Val(self.cfg.val_data_path)
        test_dataloader = DataLoader(
            self.test_dataset, 
            batch_size = self.cfg.test_batch_size, 
            shuffle = False, 
            collate_fn = collate_fn_val,
            pin_memory=True
        )
        return test_dataloader
    
    def points2adj(self, points):
            """
            return distance matrix
            Args:
            points: b, n, 2
            Returns: b, n, n
            """
            assert points.dim() == 3
            return torch.sum((points.unsqueeze(2) - points.unsqueeze(1)) ** 2, dim=-1) ** 0.5
      

    def test_step(self, batch, batch_idx):
        src = batch["src"]
        tgt = batch["tgt"]
        visited_mask = batch["visited_mask"]
        ntokens = batch["ntokens"]
        tgt_mask = batch["tgt_mask"]
        tsp_tours = batch["tsp_tours"]
        
        #for fuzzy
        src_fuzzy = batch["src_fuzzy"]

        
        batch_size = tsp_tours.shape[0]
        node_size = tsp_tours.shape[1]
        src_original = src.clone()
        fuzzy_emb_size = src_fuzzy.shape[-1]

        G = self.cfg.G

        src = src.unsqueeze(1).repeat(1, G, 1, 1).reshape(batch_size * G, node_size, 2) # [B * G, N, 2]
        #modified for fuzzy
        src_fuzzy = src_fuzzy.unsqueeze(1).repeat(1, G, 1, 1).reshape(batch_size * G, node_size, fuzzy_emb_size)  #[B*G, N, emb size of fuzzy]
        tgt = torch.arange(G).to(src.device).unsqueeze(0).repeat(batch_size, 1).reshape(batch_size * G, 1) # [B * G, 1]
        
        visited_mask = torch.zeros(batch_size, G, 1, node_size, dtype = torch.bool, device = src.device) # [B, G, 1, N]
        visited_mask[:, torch.arange(G), :, torch.arange(G)] = True
        visited_mask = visited_mask.reshape(batch_size * G, 1, node_size)
        
        self.model.eval()
        with torch.no_grad():
            #modified for fuzzy
            memory = self.model.encode(src, src_fuzzy)
            ys = tgt.clone()

            """
            #modified for given subseq
            dist_mat = self.points2adj(src)
            num_given_seq = 2

            BG = batch_size * G
            nn = torch.zeros(BG, num_given_seq - 1, device=ys.device, dtype=ys.dtype)
           
            # dist_mat에서 ys의 첫 번째 열에 해당하는 인덱스를 선택
            distances = dist_mat[torch.arange(BG), ys[:, 0]] #distances = [BG, N]

            _, nn_result = torch.topk(distances, k=num_given_seq, dim=-1, largest=False)
            nn[:, 0] = nn_result[:, -1]
            
            ys = torch.cat([ys, nn], dim=1)

            
            visited_mask[torch.arange(batch_size * G), 0, nn.squeeze()] = True
            """
    
            #for ablations
            outs=[]
            visited_mask = visited_mask.clone()
            #for i in range(self.cfg.node_size - 2):
            for i in range(self.cfg.node_size - 1):
                
                # memory, tgt, tgt_mask
                tgt_mask = subsequent_mask(ys.size(1)).type(torch.bool).to(src.device)
                
                #modified for nn
                #out = self.model.decode(memory, src_fuzzy, ys, tgt_mask, i+1)
                #modified for fuzzy
                out = self.model.decode(memory, src_fuzzy, ys, tgt_mask, i)

                #for ablation
                outs.append(out.to(torch.device('cpu')))
                
                if self.cfg.comparison_matrix == "memory":
                    comparison_matrix = self.model.memory
                elif self.cfg.comparison_matrix == "encoder_lut":
                    comparison_matrix = self.model.encoder_lut
                elif self.cfg.comparison_matrix == "decoder_lut":
                    comparison_matrix = self.model.decoder_lut
                else:
                    assert False
                
                prob = self.model.generator(out[:, -1].unsqueeze(1), visited_mask, comparison_matrix)
        
                
                _, next_word = torch.max(prob, dim=-1)
                next_word = next_word.squeeze(-1)
                
                visited_mask[torch.arange(batch_size * G), 0, next_word] = True
                
                ys = torch.cat([ys, next_word.unsqueeze(-1)], dim=1) # [B * G, N]

        import numpy as np
        if hasattr(self.model, "enc_attention_map"):
            attention_map = self.model.enc_attention_map.cpu().detach().numpy()  # numpy로 변환
            save_dir = "./attention_maps"
            os.makedirs(save_dir, exist_ok=True)
            np.save(os.path.join(save_dir, f"attention_map_batch_{batch_idx}.npy"), attention_map)
            print(f"Attention map saved for batch {batch_idx}, shape: {attention_map.shape}")
     
                
        predicted_tour_distances = self.get_tour_distance(src, ys) # [B * G, 1]
        predicted_tour_distances = predicted_tour_distances.reshape(batch_size, G)
       
        
        indices = torch.argmin(predicted_tour_distances, dim = -1) # [B]
      
        ys = ys.reshape(batch_size, G, node_size)
        ys = ys[torch.arange(batch_size), indices, :]
    

        from collections import Counter
        correct = []
        for idx in range(batch_size):
            rolled_ys = ys[idx].roll(shifts = -1)
            rolled_tour = tsp_tours[idx].roll(shifts = -1)
            edges_pred = set(tuple(sorted((ys[idx][i].item(), rolled_ys[i].item()))) for i in range(len(rolled_ys)))
            edges_tour = set(tuple(sorted((tsp_tours[idx][i].item(), rolled_tour[i].item()))) for i in range(len(rolled_tour)))
            counter_pred = Counter(edges_pred)
            counter_tour = Counter(edges_tour)

            common_edges = counter_pred & counter_tour
            correct.append(sum(common_edges.values()))
        
        correct = torch.tensor(correct)
        optimal_tour_distance = self.get_tour_distance(src_original, tsp_tours)
        predicted_tour_distance = self.get_tour_distance(src_original, ys)

        """
        print("\n ys ", ys[0,:])
        print("\n opt ", tsp_tours[0,:])
        print("++++++++++++++++++++++")
        """
        
        result = {
            "correct": correct.tolist(),
            "optimal_tour_distance": optimal_tour_distance.tolist(),
            "predicted_tour_distance": predicted_tour_distance.tolist(),
            }    
        
        self.test_corrects.extend(result["correct"])
        self.test_optimal_tour_distances.extend(result["optimal_tour_distance"])
        self.test_predicted_tour_distances.extend(result["predicted_tour_distance"])
        
        return result
        
        
        
        #HERE
        """
        filename="blank"
        if self.trainer.is_global_zero:
            
            for batch_idx in range(batch_size):
                for group_idx in range(G):
                    with open(filename, "a") as f:
                        f.write(" ".join(str(x.item()) + str(" ") + str(y.item()) for x, y in src_original[batch_idx]))

                        f.write( str(" ") + str("optimal") + str(" "))
                        tmp_tour = tsp_tours[batch_idx].tolist()
                        producted_tour = [x + 1 for x in tmp_tour]
                        del tmp_tour
                        f.write( " ".join((map(str, producted_tour))))
                        
                        
                        
                        f.write(str(" ") + str("predicted") + str(" "))
                        tmp_tour = ys[batch_idx][group_idx].tolist()
                        producted_tour = [x + 1 for x in tmp_tour]
                        del tmp_tour
                        f.write( " ".join((map(str, producted_tour))))



                        f.write(str(" ") + str("optgap") + str(" "))
                        f.write( "".join(str(((predicted_tour_distances[batch_idx][group_idx].item() - optimal_tour_distance[batch_idx].item()) / optimal_tour_distance[batch_idx].item()) * 100)))

                        f.write(str(" ") + str("hitratio") + str(" "))
                        f.write("".join(str((correct[batch_idx][group_idx].item() / node_size) * 100)))
                    
                        f.write("\n")

        return 0
        """
    
    def on_test_epoch_end(self):
        
        local_corrects = sum(self.test_corrects)
        local_optimal = sum(self.test_optimal_tour_distances)
        local_predicted = sum(self.test_predicted_tour_distances)
        
        self.test_corrects.clear()
        self.test_optimal_tour_distances.clear()
        self.test_predicted_tour_distances.clear()
        
        corrects = self.all_gather(local_corrects)
        optimal_tour_distances = self.all_gather(local_optimal)
        predicted_tour_distances = self.all_gather(local_predicted)
        
       

        if self.trainer.is_global_zero:
            correct = corrects.sum().item()
            total = self.cfg.node_size * len(self.test_dataset)
            hit_ratio = (correct / total) * 100
            
            opt_gaps = (predicted_tour_distances - optimal_tour_distances) / optimal_tour_distances
            mean_opt_gap = opt_gaps.mean().item() * 100
            self.hit_ratio = hit_ratio
            #modified
            self.optgap = mean_opt_gap
            self.hit_ratio = hit_ratio
           
            self.print(
                f"\ncorrect={correct}",
                f"\ntotal={total}",
                f"\nnode prediction(hit ratio) = {hit_ratio} %",
                f"\nmean_opt_gap = {mean_opt_gap}  %",
            )
            

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./config.yaml", help="Path to the configuration YAML file.")
    parser.add_argument("--ckpt", default="/root/FuzzyFormer/logs/lightning_logs/version_599/checkpoints/TSP20-epoch=00-opt_gap=115.3660.ckpt", help="Path to ckpt.")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_arguments()
    cfg = OmegaConf.load(args.config)
    pl.seed_everything(cfg.seed)
    
 
    if cfg.resume_checkpoint == None:
        #tsp_model = TSPModel.load_from_checkpoint(args.ckpt, strict = False)
        tsp_model = TSPModel.load_from_checkpoint(args.ckpt, strict = False, map_location=torch.device('cuda:2'))
        #match = re.search(r'version_(\w+)', args.ckpt)
        #v_num = match.group(1) #version정보 추출
    else : 
        tsp_model = TSPModel.load_from_checkpoint(cfg.resume_checkpoint, strict = False, map_location=torch.device('cuda:1'))
        #tsp_model = TSPModel.load_from_checkpoint(args.ckpt, strict = False)
        #match = re.search(r'version_(\w+)', cfg.resume_checkpoint)
        #v_num = match.group(1) #version정보 추출
    

    tsp_model.set_cfg(cfg)
    
    # build trainer
    trainer = pl.Trainer(
        default_root_dir="./",
        devices=cfg.gpus,
        accelerator="cuda",
        precision="16-mixed",
        max_epochs=cfg.max_epochs,
        reload_dataloaders_every_n_epochs=0,
        num_sanity_val_steps=0,
    )
    
    print("\n Gpu  ", cfg.gpus)
    s2 = time.time()
    trainer.test(tsp_model)
    e2 = time.time()
    elapsed_time2 = e2 - s2
    print("\n elapsed _time ", elapsed_time2)

    v_num=cfg.node_size
   

    end_discord(v_num, cfg.node_size, cfg.G,  tsp_model.optgap,elapsed_time2)
        
    