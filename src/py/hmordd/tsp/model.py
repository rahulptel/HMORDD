
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(
            self,
            d_in,
            d_hid,
            d_out,
            bias=True,
            ln_eps=1e-5,
            act="relu",
            dropout=0.0,
            normalize=False,
    ):
        super(MLP, self).__init__()
        self.d_in = d_in
        self.d_hid = d_hid
        self.d_out = d_out

        self.normalize = normalize
        if self.normalize:
            self.ln = nn.LayerNorm(d_in)
        self.linear1 = nn.Linear(d_in, d_hid, bias=bias)
        self.linear2 = nn.Linear(d_hid, d_out, bias=bias)
        self.act = nn.ReLU() if act == "relu" else nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.ln(x) if self.normalize else x
        x = self.act(self.linear1(x))
        x = self.dropout(x) if self.dropout.p > 0 else x
        x = self.act(self.linear2(x))

        return x


class MultiHeadSelfAttentionWithEdge(nn.Module):
    """
    Based on: Global Self-Attention as a Replacement for Graph Convolution
    https://arxiv.org/pdf/2108.03348
    """

    def __init__(self, cfg, is_last_block=False):
        super(MultiHeadSelfAttentionWithEdge, self).__init__()
        assert cfg.d_emb % cfg.n_heads == 0

        self.d_emb = cfg.d_emb
        self.d_k = cfg.d_emb // cfg.n_heads
        self.n_heads = cfg.n_heads
        self.is_last_block = is_last_block
        self.drop_attn = nn.Dropout(cfg.dropout_attn)
        self.drop_proj_n = nn.Dropout(cfg.dropout_proj)

        # Node Q, K, V params
        self.W_qkv = nn.Linear(cfg.d_emb, 3 * cfg.d_emb, bias=cfg.bias_mha)
        self.O_n = nn.Linear(cfg.n_heads * self.d_k, cfg.d_emb, bias=cfg.bias_mha)

        # Edge bias and gating parameters
        self.W_g = nn.Linear(cfg.d_emb, cfg.n_heads, bias=cfg.bias_mha)
        self.W_e = nn.Linear(cfg.d_emb, cfg.n_heads, bias=cfg.bias_mha)

        # Output mapping params
        if is_last_block:
            self.O_e = None
        else:
            self.O_e = nn.Linear(cfg.n_heads, cfg.d_emb, bias=cfg.bias_mha)
            self.drop_proj_e = nn.Dropout(cfg.dropout_proj)

    def forward(self, n, e):
        """
        n : batch_size x n_nodes x d_emb
        e : batch_size x n_nodes x n_nodes x d_emb
        """
        assert e is not None
        B = n.shape[0]

        # Compute QKV and reshape
        # 3 x batch_size x n_heads x n_nodes x d_k
        QKV = (
            self.W_qkv(n)
            .reshape(B, -1, 3, self.n_heads, self.d_k)
            .permute(2, 0, 3, 1, 4)
        )

        # batch_size x n_heads x n_nodes x d_k
        Q, K, V = QKV[0], QKV[1], QKV[2]

        # Compute edge bias and gate
        # batch_size x n_nodes x n_nodes x n_heads
        E, G = self.W_e(e), torch.sigmoid(self.W_g(e))
        # batch_size x n_heads x n_nodes x n_nodes
        E, G = E.permute(0, 3, 1, 2), G.permute(0, 3, 1, 2)
        # batch_size x n_heads x n_nodes
        dynamic_centrality = torch.log(1 + G.sum(-1))

        # Compute implicit attention
        # batch_size x n_heads x n_nodes x n_nodes
        _A_raw = torch.einsum("ijkl,ijlm->ijkm", [Q, K.transpose(-2, -1)])
        _A_raw = _A_raw * (self.d_k ** (-0.5))
        _A_raw = torch.clamp(_A_raw, -5, 5)
        # Add explicit edge bias
        _E = _A_raw + E
        _A = torch.softmax(_E, dim=-1)
        # Apply explicit edge gating to V
        # batch_size x n_heads x n_nodes x d_k
        _V = self.drop_attn(_A) @ V
        _V = torch.einsum("ijkl,ijk->ijkl", [_V, dynamic_centrality])
        n = self.drop_proj_n(self.O_n(_V.transpose(1, 2).reshape(B, -1, self.d_emb)))
        e = (
            None
            if self.O_e is None
            else self.drop_proj_e(self.O_e(_E.permute(0, 2, 3, 1)))
        )

        return n, e


class GTEncoderLayer(nn.Module):
    def __init__(self, cfg, is_last_block=False):
        super(GTEncoderLayer, self).__init__()
        self.is_last_block = is_last_block
        # MHA with edge information
        self.ln_n1 = nn.LayerNorm(cfg.d_emb)
        self.ln_e1 = nn.LayerNorm(cfg.d_emb)
        self.mha = MultiHeadSelfAttentionWithEdge(cfg, is_last_block=self.is_last_block)
        # FF
        self.ln_n2 = nn.LayerNorm(cfg.d_emb)
        self.mlp_node = MLP(
            cfg.d_emb,
            cfg.h2i_ratio * cfg.d_emb,
            cfg.d_emb,
            bias=cfg.bias_mlp,
            normalize=False,
            dropout=0.0,
        )
        self.dropout_mlp_n = nn.Dropout(cfg.dropout_mlp)

        if not is_last_block:
            # self.dropout_mha_e = nn.Dropout(dropout_mha)
            self.ln_e2 = nn.LayerNorm(cfg.d_emb)
            self.mlp_edge = MLP(
                cfg.d_emb,
                cfg.h2i_ratio * cfg.d_emb,
                cfg.d_emb,
                bias=cfg.bias_mlp,
                normalize=False,
                dropout=0.0,
            )
            self.dropout_mlp_e = nn.Dropout(cfg.dropout_mlp)

    def forward(self, n, e):
        n_norm = self.ln_n1(n)
        e_norm = self.ln_e1(e)
        n_, e_ = self.mha(n_norm, e_norm)
        n = n + n_

        n = n + self.dropout_mlp_n(self.mlp_node(self.ln_n2(n)))
        if not self.is_last_block:
            e = e + e_
            e = e + self.dropout_mlp_e(self.mlp_edge(self.ln_e2(e)))

        return n, e


class GTEncoder(nn.Module):
    def __init__(self, cfg):
        super(GTEncoder, self).__init__()
        self.encoder_blocks = nn.ModuleList()
        for i in range(cfg.n_layers):
            is_last_block = i == cfg.n_layers - 1
            self.encoder_blocks.append(GTEncoderLayer(cfg, is_last_block=is_last_block))

    def forward(self, n, e):
        for block in self.encoder_blocks:
            n, e = block(n, e)

        return n


class TokenEmbedGraph(nn.Module):
    """
    DeepSet-based node and edge embeddings
    """

    def __init__(self, cfg, n_node_feat=7):
        super(TokenEmbedGraph, self).__init__()
        self.linear1 = nn.Linear(n_node_feat, 2 * cfg.d_emb)
        self.linear2 = nn.Linear(2 * cfg.d_emb, cfg.d_emb)
        self.linear3 = nn.Linear(1, cfg.d_emb)
        self.linear4 = nn.Linear(cfg.d_emb, cfg.d_emb)
        self.act = nn.ReLU() if cfg.act == "relu" else nn.GELU()

    def forward(self, n, e):
        n = self.act(self.linear1(n))  # B x n_objs x n_vars x (2 * d_emb)
        n = n.sum(1)  # B x n_vars x (2 * d_emb)
        n = self.act(self.linear2(n))  # B x n_vars x d_emb

        e = e.unsqueeze(-1)
        e = self.act(self.linear3(e))  # B x n_objs x n_vars x n_vars x d_emb
        e = e.sum(1)  # B x n_vars x n_vars x d_emb
        e = self.act(self.linear4(e))  # B x n_vars x n_vars x d_emb

        return n, e


class ParetoNodePredictor(nn.Module):
    # NOT_VISITED = 0
    # VISITED = 1
    # LAST_VISITED = 2
    NODE_VISIT_TYPES = 3
    N_LAYER_INDEX = 1
    N_CLASSES = 2

    def __init__(self, cfg):
        super(ParetoNodePredictor, self).__init__()
        self.concat_emb = cfg.concat_emb
        self.act = nn.ReLU() if cfg.act == "relu" else nn.GELU()
        self.token_encoder = TokenEmbedGraph(cfg)
        self.graph_encoder = GTEncoder(cfg)
        self.visit_encoder = nn.Embedding(self.NODE_VISIT_TYPES, cfg.d_emb)
        self.node_visit_encoder1 = nn.Sequential(
            nn.Linear(cfg.d_emb, cfg.h2i_ratio * cfg.d_emb),
            self.act,
        )
        self.node_visit_encoder2 = nn.Sequential(
            nn.Linear(cfg.h2i_ratio * cfg.d_emb, cfg.d_emb),
            self.act,
        )
        self.layer_encoder = nn.Sequential(
            nn.Linear(self.N_LAYER_INDEX, cfg.d_emb),
            self.act,
        )
        if self.concat_emb:
            self.pareto_predictor = nn.Sequential(
                nn.Linear(3 * cfg.d_emb, cfg.h2i_ratio * cfg.d_emb),
                self.act,
                nn.Linear(cfg.h2i_ratio * cfg.d_emb, self.N_CLASSES),
            )
        else:
            self.pareto_predictor = nn.Sequential(
                nn.Linear(cfg.d_emb, cfg.h2i_ratio * cfg.d_emb),
                self.act,
                nn.Linear(cfg.h2i_ratio * cfg.d_emb, self.N_CLASSES),
            )

    def forward(self, n, e, l, s):
        n, e = self.token_encoder(n, e)
        n = self.graph_encoder(n, e)  # B x n_vars x d_emb
        B, n_vars, d_emb = n.shape

        last_visit = s[:, -1]
        visit_mask = s[:, :-1]
        visit_mask[torch.arange(B), last_visit.long()] = 2
        visit_enc = self.visit_encoder(visit_mask.long())

        # B x d_emb
        node_visit = self.node_visit_encoder2(
            self.node_visit_encoder1((n + visit_enc)).sum(1)
        )
        customer_enc = n[torch.arange(B), last_visit.long()]
        l_enc = self.layer_encoder(((n_vars - l) / n_vars).unsqueeze(-1))

        if self.concat_emb:
            return self.pareto_predictor(
                torch.cat((node_visit, customer_enc, l_enc), dim=-1)
            )
        else:
            return self.pareto_predictor(node_visit + customer_enc + l_enc)

    def configure_optimizer(self, cfg):
        if cfg.wd > 0:
            # Ref: https://github.com/karpathy/nanoGPT/blob/master/model.py
            # start with all of the candidate parameters
            param_dict = {pn: p for pn, p in self.named_parameters()}
            # filter out those that do not require grad
            param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
            # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
            # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
            decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
            nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
            optim_groups = [
                {"params": decay_params, "weight_decay": cfg.wd},
                {"params": nodecay_params, "weight_decay": 0.0},
            ]
            num_decay_params = sum(p.numel() for p in decay_params)
            num_nodecay_params = sum(p.numel() for p in nodecay_params)
            print(
                f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
            )
            print(
                f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
            )
            optimizer_cls = getattr(torch.optim, cfg.type)
            optimizer = optimizer_cls(
                optim_groups,
                lr=cfg.lr,
                betas=(cfg.beta1, cfg.beta2),
            )
        else:
            optimizer_cls = getattr(torch.optim, cfg.type)
            optimizer = optimizer_cls(
                self.parameters(),
                lr=cfg.lr,
                betas=(cfg.beta1, cfg.beta2),
            )
        print(f"using optimizer: {cfg.type}")
        print()

        return optimizer