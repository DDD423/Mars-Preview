"""A randomly initialized byte-character Transformer, not a pretrained LLM."""
from dataclasses import asdict, dataclass
import copy
import math
import torch
from torch import nn
from .language import FILE_TOOLS, IDS, TOKENS, Grammar

TYPE_IDS={"NAME":1,"PATH":2,"TEXT":3,"VALUE":4}


def encode_request(request):
    kinds={s["slot"]:TYPE_IDS[s["type"]] for s in request["slots"]}
    available=request.get("enabled_tools")
    enabled=set(FILE_TOOLS if available is None else available)
    rows=[[0,0,0,0,9+i,5 if tool in enabled else 0] for i,tool in enumerate(FILE_TOOLS)]
    for part in request["segments"]:
        if part["kind"]=="slot":
            slot=int(part["id"][1:-1]); rows.append([0,0,0,0,slot,kinds[slot]])
        else:
            for char in part["text"]:
                bs=char.encode("utf-8"); rows.append([i*257+b+1 for i,b in enumerate(bs)]+[0]*(4-len(bs))+[0,0])
    rows.append([0,0,0,0,31,0])
    if len(rows)>300: raise ValueError("Translator input exceeds 300 encoded positions")
    return rows


def batch_inputs(requests, device):
    rows=[encode_request(r) for r in requests]; width=max(map(len,rows))
    x=torch.zeros(len(rows),width,6,dtype=torch.long)
    for i,row in enumerate(rows): x[i,:len(row)]=torch.tensor(row)
    return x.to(device)


@dataclass
class Config:
    embedding_dim:int=64
    hidden_dim:int=192
    heads:int=6
    encoder_layers:int=3
    decoder_layers:int=3
    feedforward_dim:int=768
    dropout:float=.1


class TranslatorModel(nn.Module):
    def __init__(self, config=None):
        super().__init__(); self.config=config or Config(); c=self.config
        self.byte_embedding=nn.Embedding(1028,c.embedding_dim,padding_idx=0)
        self.project=nn.Linear(c.embedding_dim,c.hidden_dim,bias=False)
        self.special=nn.Embedding(32,c.hidden_dim,padding_idx=0)
        self.types=nn.Embedding(6,c.hidden_dim,padding_idx=0)
        self.position=nn.Embedding(300,c.hidden_dim)
        self.output_embedding=nn.Embedding(len(TOKENS),c.hidden_dim,padding_idx=0)
        self.output_position=nn.Embedding(193,c.hidden_dim)
        enc=nn.TransformerEncoderLayer(c.hidden_dim,c.heads,c.feedforward_dim,c.dropout,batch_first=True,norm_first=True)
        dec=nn.TransformerDecoderLayer(c.hidden_dim,c.heads,c.feedforward_dim,c.dropout,batch_first=True,norm_first=True)
        self.encoder=nn.TransformerEncoder(enc,c.encoder_layers,norm=nn.LayerNorm(c.hidden_dim),enable_nested_tensor=False)
        self.decoder=nn.TransformerDecoder(dec,c.decoder_layers,norm=nn.LayerNorm(c.hidden_dim))
        self.head=nn.Linear(c.hidden_dim,len(TOKENS))
        self.dropout=nn.Dropout(c.dropout)

    def encode(self,x):
        pad=~x.any(-1)
        e=self.project(self.byte_embedding(x[:,:,:4]).sum(2))+self.special(x[:,:,4])+self.types(x[:,:,5])
        e=e+self.position(torch.arange(x.shape[1],device=x.device))
        return self.encoder(self.dropout(e),src_key_padding_mask=pad),pad

    def decode(self,y,memory,pad):
        pos=torch.arange(y.shape[1],device=y.device)
        emb=self.output_embedding(y)+self.output_position(pos)
        causal=torch.ones(y.shape[1],y.shape[1],device=y.device,dtype=torch.bool).triu(1)
        out=self.decoder(self.dropout(emb),memory,tgt_mask=causal,tgt_key_padding_mask=y.eq(0),memory_key_padding_mask=pad)
        return self.head(out)

    def forward(self,x,y):
        memory,pad=self.encode(x); return self.decode(y,memory,pad)

    def metadata(self):
        return {"config":asdict(self.config),"parameters":sum(p.numel() for p in self.parameters())}

    @torch.inference_mode()
    def generate(self,requests,device="cpu",beam=4,max_tokens=192):
        """Batched beam decoding. Only typed output grammar constrains choices."""
        self.eval(); memory,pad=self.encode(batch_inputs(requests,device))
        active=[]; completed=[[] for _ in requests]
        for i,r in enumerate(requests): active.append({"row":i,"tokens":[IDS["BOS"]],"grammar":Grammar(r),"score":0.,"logs":[]})
        for _ in range(max_tokens):
            if not active: break
            indices=torch.tensor([a["row"] for a in active],device=device)
            y=torch.tensor([a["tokens"] for a in active],device=device)
            logits=self.decode(y,memory[indices],pad[indices])[:,-1].float()
            mask=torch.full_like(logits,-1e9)
            choices=[]
            for i,a in enumerate(active):
                allowed=a["grammar"].allowed(); choices.append(len(allowed)); mask[i,allowed]=0
            logp=(logits+mask).log_softmax(-1)
            values,ids=logp.topk(min(beam,len(TOKENS)),dim=-1)
            values,ids=values.cpu().tolist(),ids.cpu().tolist()
            candidates=[[] for _ in requests]
            for i,a in enumerate(active):
                if not choices[i]: continue
                for value,token in zip(values[i],ids[i]):
                    if value < -1e8: continue
                    g=copy.deepcopy(a["grammar"]); g.advance(token)
                    b={"row":a["row"],"tokens":a["tokens"]+[token],"grammar":g,"score":a["score"]+value,
                       "logs":a["logs"]+([value] if choices[i]>1 else [])}
                    if g.state=="done": completed[a["row"]].append(b)
                    else: candidates[a["row"]].append(b)
            active=[]
            for i,items in enumerate(candidates):
                # Length normalization avoids preferring a short refusal solely by token count.
                rank=lambda a:a["score"]/max(1,len(a["logs"]))**.7
                completed[i]=sorted(completed[i],key=rank,reverse=True)[:beam]
                active+=sorted(items,key=rank,reverse=True)[:beam]
            # Stop rows whose completed beams dominate surviving beams.
            remaining=[]
            for a in active:
                done=completed[a["row"]]
                if len(done)>=beam and a["score"]/max(1,len(a["logs"]))**.7 < done[-1]["score"]/max(1,len(done[-1]["logs"]))**.7: continue
                remaining.append(a)
            active=remaining
        output=[]
        for i,done in enumerate(completed):
            if not done:
                output.append({"plan":None,"tokens":[],"features":[-20.,0.,4.]}); continue
            best=done[0]; mean=sum(best["logs"])/max(1,len(best["logs"]))
            second=done[1]["score"]/max(1,len(done[1]["logs"]))**.7 if len(done)>1 else best["score"]-10
            gap=best["score"]/max(1,len(best["logs"]))**.7-second
            output.append({"plan":best["grammar"].plan(requests[i].get("context_id","ctx_eval")),"tokens":best["tokens"],
                           "features":[mean,gap,float(len(best["grammar"].steps))]})
        return output
