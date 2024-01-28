import torch
from .layers import EmbeddingLayer_wo_offset, MultiLayerPerceptron, Attention


class MDDPO(torch.nn.Module):
    def __init__(self, categorical_field_dims, numerical_num, embed_dim, bottom_mlp_dims, tower_mlp_dims, task_num, expert_num, dropout):
        super().__init__()
        self.embedding = EmbeddingLayer_wo_offset(10614790, embed_dim)
        # self.numerical_layer = torch.nn.Linear(numerical_num, embed_dim)
        self.embed_output_dim = len(categorical_field_dims) * embed_dim
        self.task_num = task_num
        self.expert_num = expert_num

        self.ctr_info = torch.nn.Sequential(
            torch.nn.Linear(bottom_mlp_dims[-1], bottom_mlp_dims[-1]),
            torch.nn.ReLU()
        )
        self.ctcvr_info = torch.nn.Sequential(
            torch.nn.Linear(bottom_mlp_dims[-1], bottom_mlp_dims[-1]),
            torch.nn.ReLU()
        ) 
        self.ctr_attention = Attention(dim=bottom_mlp_dims[-1])
        self.ctcvr_attention = Attention(dim=bottom_mlp_dims[-1])
        self.cvr_info = torch.nn.Sequential(
            torch.nn.Linear(2*bottom_mlp_dims[-1], bottom_mlp_dims[-1]),
            torch.nn.ReLU()
        )


        self.expert = torch.nn.ModuleList([MultiLayerPerceptron(self.embed_output_dim, bottom_mlp_dims, dropout, output_layer=False) for i in range(expert_num)])
        self.tower = torch.nn.ModuleList([MultiLayerPerceptron(bottom_mlp_dims[-1], tower_mlp_dims, dropout) for i in range(task_num)])
        self.gate = torch.nn.ModuleList([torch.nn.Sequential(torch.nn.Linear(self.embed_output_dim, expert_num), torch.nn.Softmax(dim=1)) for i in range(task_num)])

    def forward(self, categorical_x, numerical_x):
        """
        :param 
        categorical_x: Long tensor of size ``(batch_size, categorical_field_dims)``
        numerical_x: Long tensor of size ``(batch_size, numerical_num)``
        """
        categorical_emb = self.embedding(categorical_x)
        # numerical_emb = self.numerical_layer(numerical_x).unsqueeze(1)
        emb = categorical_emb.view(-1, self.embed_output_dim)
        gate_value = [self.gate[i](emb).unsqueeze(1) for i in range(self.task_num)]
        fea = torch.cat([self.expert[i](emb).unsqueeze(1) for i in range(self.expert_num)], dim = 1)
        task_fea = [torch.bmm(gate_value[i], fea).squeeze(1) for i in range(self.task_num)]

        ctr_info = self.ctr_info(task_fea[0]).unsqueeze(1)
        cvr_ori = task_fea[1].unsqueeze(1)
        new_cvr1 = self.ctr_attention(torch.cat([ctr_info, cvr_ori], dim=1))

        ctcvr_info = self.ctcvr_info(task_fea[2]).unsqueeze(1)
        cvr_ori = task_fea[1].unsqueeze(1)
        new_cvr2 = self.ctcvr_attention(torch.cat([ctcvr_info, cvr_ori], dim=1))
        
        task_fea[1] = self.cvr_info(torch.cat([new_cvr1, new_cvr2], dim=1))
        
        results = [torch.sigmoid(self.tower[i](task_fea[i]).squeeze(1)) for i in range(self.task_num)]
        return results