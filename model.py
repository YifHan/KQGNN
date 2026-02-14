import torch.nn as nn
import torch.nn.functional as F
import torch
import dgl

class CrossAttentionLayer(nn.Module):
    def __init__(self, image_dim, block_dim, out_dim):
        super(CrossAttentionLayer, self).__init__()
        self.image_fc = nn.Linear(image_dim, out_dim, bias=False)
        self.block_fc = nn.Linear(block_dim, out_dim, bias=False)
        self.Wv = nn.Parameter(torch.Tensor(out_dim, out_dim))
        self.bv = nn.Parameter(torch.Tensor(out_dim))
        self.We = nn.Parameter(torch.Tensor(out_dim, out_dim))
        self.be = nn.Parameter(torch.Tensor(out_dim))
        self.sigmoid = nn.Sigmoid()
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.image_fc.weight)
        nn.init.xavier_normal_(self.block_fc.weight)
        nn.init.xavier_normal_(self.Wv)
        nn.init.xavier_normal_(self.We)
        nn.init.zeros_(self.bv)
        nn.init.zeros_(self.be)

    def forward(self, image_features, block_features):
        print(f"image_features shape: {image_features.shape}")
        print(f"block_features shape: {block_features.shape}")

        # Image featur
        fi = self.image_fc(image_features)
        # Text feature
        ti = self.block_fc(block_features)

        alpha_v = self.sigmoid(torch.matmul(fi, self.Wv) + self.bv)
        alpha_e = self.sigmoid(torch.matmul(ti, self.We) + self.be) 

        # Feature fusion
        fused_features = alpha_v * fi + alpha_e * ti
        return fused_features


class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, use_residual=False):
        super(GATLayer, self).__init__()
        self.text_fc = nn.Linear(in_dim, out_dim, bias=False)
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)
        self.use_residual = use_residual
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)

    def edge_attention(self, edges):
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        a = self.attn_fc(z2)
        return {'e': F.leaky_relu(a)}

    def message_func(self, edges):
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': h}

    def forward(self, blocks, layer_id):
        if hasattr(blocks[layer_id], 'srcdata'):
            h = blocks[layer_id].srcdata['features']
        else:
            h = blocks

        z = self.fc(h)
        if hasattr(blocks[layer_id], 'srcdata'):
            blocks[layer_id].srcdata['z'] = z
            z_dst = z[:blocks[layer_id].number_of_dst_nodes()]
            blocks[layer_id].dstdata['z'] = z_dst
            blocks[layer_id].apply_edges(self.edge_attention)
            blocks[layer_id].update_all(self.message_func, self.reduce_func)

            if self.use_residual:
                return z_dst + blocks[layer_id].dstdata['h']
            return blocks[layer_id].dstdata['h']
        else:
            return z


class MultiHeadGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, merge='cat', use_residual=False):
        super(MultiHeadGATLayer, self).__init__()
        self.heads = nn.ModuleList(
            [GATLayer(in_dim, out_dim, use_residual) for _ in range(num_heads)]
        )
        self.merge = merge

    def forward(self, blocks, layer_id):
        head_outs = [attn_head(blocks, layer_id) for attn_head in self.heads]
        if self.merge == 'cat':
            return torch.cat(head_outs, dim=1)
        else:
            return torch.mean(torch.stack(head_outs))


class GAT(nn.Module):
    def __init__(self, image_dim, block_dim, hidden_dim, out_dim, num_heads, use_residual=False):
        super(GAT, self).__init__()
        self.cross_attention = CrossAttentionLayer(image_dim, block_dim, hidden_dim * num_heads)
        self.layer1 = MultiHeadGATLayer(hidden_dim * num_heads, hidden_dim, num_heads,
                                        merge='cat', use_residual=use_residual)
        self.layer2 = MultiHeadGATLayer(hidden_dim * num_heads, out_dim, 1,
                                        merge='cat', use_residual=use_residual)

    def forward(self, blocks, image_features, text_features,
                key_img_features=None, key_text_features=None, flag=None):
        print(f"flag: {flag}")
        print(f"image_features shape: {image_features.shape}")
        print(f"text_features shape: {text_features.shape}")
        if key_img_features is not None and key_text_features is not None:
            print(f"key_img_features shape: {key_img_features.shape}")
            print(f"key_text_features shape: {key_text_features.shape}")

        fused_features = self.cross_attention(image_features, text_features)
        if flag == '1':
            if key_img_features is not None and key_text_features is not None:
                # key_fused_features = self.cross_attention(key_img_features, key_text_features)
                # fused_features = torch.cat([fused_features, key_fused_features], dim=0)
                print(f"Combined fused_features shape: {fused_features.shape}")

            device = fused_features.device
            num_src_nodes = blocks[0].num_src_nodes()
            num_dst_nodes = blocks[0].num_dst_nodes()
            num_new_nodes = fused_features.size(0) - num_src_nodes

            if num_new_nodes > 0:
                new_src_nodes = torch.arange(num_src_nodes, num_src_nodes + num_new_nodes, device=device)
                new_dst_nodes = torch.arange(num_dst_nodes, num_dst_nodes + num_new_nodes, device=device)

                new_edges = blocks[0].edges()
                new_edges_src = torch.cat([new_edges[0].to(device), new_src_nodes])
                new_edges_dst = torch.cat([new_edges[1].to(device), new_dst_nodes])
                max_dst_id = max(new_edges_dst.max().item(), num_dst_nodes)

                new_block = dgl.create_block(
                    (new_edges_src, new_edges_dst),
                    num_src_nodes=fused_features.size(0),
                    num_dst_nodes=max_dst_id + 1
                )

                for k, v in blocks[0].srcdata.items():
                    zeros = torch.zeros(num_new_nodes, device=device) if v.dim() == 1 \
                        else torch.zeros(num_new_nodes, v.size(1), device=device)
                    new_block.srcdata[k] = torch.cat([v.to(device), zeros], dim=0)

                for k, v in blocks[0].dstdata.items():
                    zeros = torch.zeros(num_new_nodes, device=device) if v.dim() == 1 \
                        else torch.zeros(num_new_nodes, v.size(1), device=device)
                    new_block.dstdata[k] = torch.cat([v.to(device), zeros], dim=0)

                blocks[0] = new_block

        blocks[0].srcdata['features'] = fused_features

        h = self.layer1(blocks, 0)
        h = F.elu(h)

        if h.size(0) > blocks[1].num_src_nodes():
            h = h[:blocks[1].num_src_nodes()]
            
        blocks[1].srcdata['features'] = h

        h = self.layer2(blocks, 1)

        if h.size(0) > blocks[1].num_dst_nodes():
            h = h[:blocks[1].num_dst_nodes()]

        return h