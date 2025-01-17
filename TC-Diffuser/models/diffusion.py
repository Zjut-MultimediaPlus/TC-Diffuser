import torch
import torch.nn.functional as F
from torch.nn import Module, Parameter, ModuleList
import numpy as np
import torch.nn as nn
from .common import *
import pdb

class VarianceSchedule(Module):
#var_sched = VarianceSchedule(num_steps=100,beta_T=5e-2,mode='linear')
    def __init__(self, num_steps, mode='linear',beta_1=1e-4, beta_T=5e-2,cosine_s=8e-3):
        super().__init__()
        assert mode in ('linear', 'cosine')
        self.num_steps = num_steps
        self.beta_1 = beta_1
        self.beta_T = beta_T
        self.mode = mode

        if mode == 'linear':
            betas = torch.linspace(beta_1, beta_T, steps=num_steps)
        elif mode == 'cosine':
            timesteps = (
            torch.arange(num_steps + 1) / num_steps + cosine_s
            )
            alphas = timesteps / (1 + cosine_s) * math.pi / 2
            alphas = torch.cos(alphas).pow(2)
            alphas = alphas / alphas[0]
            betas = 1 - alphas[1:] / alphas[:-1]
            betas = betas.clamp(max=0.999)

        betas = torch.cat([torch.zeros([1]), betas], dim=0)     # Padding

        alphas = 1 - betas
        log_alphas = torch.log(alphas)
        for i in range(1, log_alphas.size(0)):  # 1 to T
            log_alphas[i] += log_alphas[i - 1]
        alpha_bars = log_alphas.exp()

        sigmas_flex = torch.sqrt(betas)
        sigmas_inflex = torch.zeros_like(sigmas_flex)
        for i in range(1, sigmas_flex.size(0)):
            sigmas_inflex[i] = ((1 - alpha_bars[i-1]) / (1 - alpha_bars[i])) * betas[i]
        sigmas_inflex = torch.sqrt(sigmas_inflex)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alpha_bars', alpha_bars)
        self.register_buffer('sigmas_flex', sigmas_flex)
        self.register_buffer('sigmas_inflex', sigmas_inflex)

    def uniform_sample_t(self, batch_size):
        ts = np.random.choice(np.arange(1, self.num_steps+1), batch_size)
        return ts.tolist()
    def get_sigmas(self, t, flexibility):
        assert 0 <= flexibility and flexibility <= 1
        sigmas = self.sigmas_flex[t] * flexibility + self.sigmas_inflex[t] * (1 - flexibility)
        return sigmas #

class DiffusionTraj(Module):

    def __init__(self, net, var_sched:VarianceSchedule):
        super().__init__()
        # self.net(c0 * x_0 + c1 * e_rand, beta=beta, context=context)
        self.net = net  # TransformerConcatLinear net = self.diffnet(point_dim=4, context_dim=config.encoder_dim, tf_layer=config.tf_layer, residual=False)
        self.var_sched = var_sched #var_sched = VarianceSchedule(num_steps=100,beta_T=5e-2,mode='linear')

    #loss = self.diffusion.get_loss(y_t.cuda(), x_gph,encoded_age)
    def get_loss(self, x_0, context, encoded_age,encoded_env_data, t=None): #x_0是y_t 就是12时刻的一个未知的部分 torch.Size([256, 12, 2])
        batch_size, T ,point_dim = x_0.size()
        if t == None:
            t = self.var_sched.uniform_sample_t(batch_size)#随机选一个步长
        alpha_bar = self.var_sched.alpha_bars[t] #torch.Size([256])
        beta = self.var_sched.betas[t].cuda() #torch.Size([256])
        c0 = torch.sqrt(alpha_bar).view(-1, 1, 1).cuda()       # (B, 1, 1)
        c1 = torch.sqrt(1 - alpha_bar).view(-1, 1, 1).cuda()   # (B, 1, 1) torch.Size([256, 1, 1])
        e_rand = torch.randn_like(x_0).cuda()  # torch.Size([256, 4, 4])

        e_theta = self.net(c0 * x_0 + c1 * e_rand, beta=beta, context=context,
                           encoded_age = encoded_age,encoded_env_data=encoded_env_data)#torch.Size([256, 4, 4])

        # Wt = [1, 1, 1, 1, 1, 1, 1, 1]

        first_add = 1.3
        if (T == 4):
            Wt = [first_add, 1, 1, 1]
        elif (T == 3):
            Wt = [first_add, 1, 1]
        elif (T == 2):
            Wt = [first_add, 1]
        elif (T == 1):
            Wt = [first_add]

        loss = 0
        for i in range(e_theta.size(1)): #T
            loss_i = Wt[i]*F.mse_loss(e_theta[:,i,:].view(-1, point_dim), e_rand[:,i,:].view(-1, point_dim), reduction='mean')
            loss+=loss_i
        loss = loss/e_theta.size(1)

        return loss # 是sum? 区分时刻


    def sample(self, num_points, context,
               encoded_age,encoded_env_data,
               sample, bestof, point_dim=4, flexibility=0.0, ret_traj=False, sampling="ddpm", step=100):
        traj_list = []
        for i in range(sample): #采样次数/要生成的样本数？ sample=6
            batch_size = context.size(0) #个体的个数
            if bestof: #true
                x_T = torch.randn([batch_size, num_points, point_dim]).to(context.device) #从正态分布中得到的标准分布
            else:
                x_T = torch.zeros([batch_size, num_points, point_dim]).to(context.device)
            traj = {self.var_sched.num_steps: x_T} # x_T：torch.Size([3, 4, 2])
            stride = step
            #stride = int(100/stride)
            #traj[t]：【B，4，4】 单条轨迹迭代
            for t in range(self.var_sched.num_steps, 0, -stride):  #从self.var_sched.num_steps(100)到0 步长为step 100
                z = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)
                alpha = self.var_sched.alphas[t]
                alpha_bar = self.var_sched.alpha_bars[t]
                alpha_bar_next = self.var_sched.alpha_bars[t-stride]
                sigma = self.var_sched.get_sigmas(t, flexibility)
                c0 = 1.0 / torch.sqrt(alpha)
                c1 = (1 - alpha) / torch.sqrt(1 - alpha_bar)

                #traj中获取当前轨迹x_t
                x_t = traj[t]#torch.Size([3, 4, 2])#目标-预测轨迹 4个时刻的 两个值--4个值
                beta = self.var_sched.betas[[t]*batch_size]
                #将x_t，beta和context传递给net函数以获取输出e_theta
                # 这句？def forward(self, x, beta, context
                #                 , encoded_age
                e_theta = self.net(x_t, beta=beta, context=context
                                   ,encoded_age = encoded_age,encoded_env_data=encoded_env_data)  #torch.Size([B, 4, 4])
                if sampling == "ddpm":#true
                    #如果sampling为"ddpm"，使用DDPM采样方法计算下一个点x_next。 下一个点是什么意思
                    x_next = c0 * (x_t - c1 * e_theta) + sigma * z   #torch.Size([B, 4, 4])
                elif sampling == "ddim":
                    x0_t = (x_t - e_theta * (1 - alpha_bar).sqrt()) / alpha_bar.sqrt()
                    x_next = alpha_bar_next.sqrt() * x0_t + (1 - alpha_bar_next).sqrt() * e_theta
                else:
                    pdb.set_trace()
                #将生成的点x_next存储在traj中，供下一次迭代使用
                traj[t-stride] = x_next.detach()     # Stop gradient and save trajectory.
                #将上一个输出x_t移动到CPU内存
                traj[t] = traj[t].cpu()         # Move previous output to CPU memory.
                if not ret_traj:#果ret_traj为假，则删除时间步t处的中间轨迹
                   del traj[t]
            #为真，则将完整的轨迹traj附加到traj_list中。 中间过程吧？
            if ret_traj:
                traj_list.append(traj) #一条轨迹 迭代的过程
            else:#false
                traj_list.append(traj[0])#否则，只将最终生成的点traj[0]附加到traj_list中。 这边只有一条轨迹
        return torch.stack(traj_list) #【sample,B,4,4】

class TrajNet(Module):

    def __init__(self, point_dim, context_dim, residual):
        super().__init__()
        self.act = F.leaky_relu
        self.residual = residual
        self.layers = ModuleList([  ################这句可能是突破口
            ConcatSquashLinear(2, 128, context_dim+3),
            ConcatSquashLinear(128, 256, context_dim+3),
            ConcatSquashLinear(256, 512, context_dim+3),
            ConcatSquashLinear(512, 256, context_dim+3),
            ConcatSquashLinear(256, 128, context_dim+3),
            ConcatSquashLinear(128, 2, context_dim+3),

        ])

    def forward(self, x, beta, context):
        """
        Args:
            x:  Point clouds at some timestep t, (B, N, d).
            beta:     Time. (B, ).
            context:  Shape latents. (B, F).
        """
        batch_size = x.size(0)
        beta = beta.view(batch_size, 1, 1)          # (B, 1, 1)
        context = context.view(batch_size, 1, -1)   # (B, 1, F)

        time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)  # (B, 1, 3)
        ctx_emb = torch.cat([time_emb, context], dim=-1)    # (B, 1, F+3)

        out = x
        #pdb.set_trace()
        for i, layer in enumerate(self.layers):
            out = layer(ctx=ctx_emb, x=out)
            if i < len(self.layers) - 1:
                out = self.act(out)

        if self.residual:
            return x + out
        else:
            return out

class TransformerConcatLinear(Module):
    def __init__(self, point_dim, context_dim, tf_layer, residual):
        super().__init__()
        self.residual = residual # False
        self.pos_emb = PositionalEncoding(d_model=2*context_dim, dropout=0.1, max_len=24)
        self.concat1_2 = ConcatSquashLinear(context_dim, 2 * context_dim, context_dim + 3)
        self.layer = nn.TransformerEncoderLayer(d_model=2*context_dim, nhead=4, dim_feedforward=4*context_dim)
        self.transformer_encoder = nn.TransformerEncoder(self.layer, num_layers=tf_layer)
        self.concat3 = ConcatSquashLinear(2*context_dim,context_dim,context_dim+3)
        self.concat4 = ConcatSquashLinear(context_dim,context_dim//2,context_dim+3)
        self.linear = ConcatSquashLinear(context_dim//2, point_dim, context_dim+3)
        self.concat5 = ConcatSquashLinear(point_dim, point_dim, point_dim)
        self.linear_true = Linear(128, point_dim)

        self.concat_env_age_x_traj = ConcatSquashLinear(2, 128, 32)
        self.concat_env_age_x_inten = ConcatSquashLinear(1, 64, 36)
        self.concat_env_age_x_wind = ConcatSquashLinear(1, 64, 20)

        self.concat1 = ConcatSquashLinear(point_dim, 2 * context_dim,
                                          context_dim + 3)  # context_dim:384?  #ctx_emb:torch.Size([B, 1, 387]) x:torch.Size([B, 4, 4])
        #_type1
        # self.concat_env_age_x_traj_type1 = ConcatSquashLinear(2, 2, 32)
        # self.concat_env_age_x_inten_type1 = ConcatSquashLinear(1, 1, 36)
        # self.concat_env_age_x_wind_type1 = ConcatSquashLinear(1, 1, 20)  # x的输入维度,输出的维度，ctx的输入维度

        # self.concat1_type2 = ConcatSquashLinear(point_dim, 2 * context_dim, 347)
        # self.concat3_type2 = ConcatSquashLinear(2 * context_dim, context_dim, 347)
        # self.concat4_type2 = ConcatSquashLinear(context_dim, context_dim // 2, 347)
        # self.linear_type2 = ConcatSquashLinear(context_dim // 2, point_dim, 347)

        #=======================================================
        # Missing key(s) in state_dict: ,
        #========================================================
    # self.net(c0 * x_0 + c1 * e_rand, beta=beta, context=context,
    #                            encoded_age = encoded_age)

    def ptask_then_pshare(self, x, beta, context
                , encoded_age,encoded_env_data):
        batch_size = x.size(0)
        beta = beta.view(batch_size, 1, 1)  # (B, 1, 1) torch.Size([B, 1, 1])
        context = context.view(batch_size, 1, -1)  # (B, 1, F) torch.Size([B, 1, 256])--torch.Size([B, 1, 384])

        encoded_age = encoded_age.view(batch_size, 1, -1)  # (B,1,4)
        env_traj = encoded_env_data[0].to(x.device).view(batch_size, 1, -1)
        env_inten = encoded_env_data[1].to(x.device).view(batch_size, 1, -1)
        env_wind = encoded_env_data[2].to(x.device).view(batch_size, 1, -1)

        # 轨迹
        ctx_emb_env_age_traj = env_traj
        ctx_emb_env_age_traj = ctx_emb_env_age_traj / 100
        # 强度
        ctx_emb_env_age_inten = torch.cat([env_inten, encoded_age], dim=-1)
        ctx_emb_env_age_inten = ctx_emb_env_age_inten / 100
        # 风速
        ctx_emb_env_age_wind = torch.cat([env_wind, encoded_age], dim=-1)  # (B, 1, 20)
        ctx_emb_env_age_wind = ctx_emb_env_age_wind / 100

        x_traj = self.concat_env_age_x_traj(ctx_emb_env_age_traj,
                                            x[:, :, 0:2])  # (B, 1, 84) (B,4,4)  输出torch.Size([256, 4, 256])
        x_inten = self.concat_env_age_x_inten(ctx_emb_env_age_inten, x[:, :, 2:3])
        x_wind = self.concat_env_age_x_wind(ctx_emb_env_age_wind, x[:, :, 3:4])
        x = torch.cat((x_traj, x_inten, x_wind), dim=2)

        time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)  # (B, 1, 3)
        ctx_emb = torch.cat([time_emb, context], dim=-1)  # (B, 1, F+3) torch.Size([B, 1, 259])

        x = self.concat1_2(ctx_emb, x)  # [B, 4, 512]
        # 【B,4,512】
        final_emb = x.permute(1, 0, 2)  # torch.Size([4, B, 512])
        final_emb = self.pos_emb(final_emb)  # torch.Size([4, B, 512])
        trans = self.transformer_encoder(final_emb).permute(1, 0, 2)  # torch.Size([B, 4, 512])
        trans = self.concat3(ctx_emb, trans)  # torch.Size([B, 4, 256])
        trans = self.concat4(ctx_emb, trans)  # torch.Size([B, 4, 128])
        trans = self.linear(ctx_emb, trans)  # torch.Size([B, 4, 4])  #直接从512缩小到4
        return trans
    def pshare_then_ptask(self, x, beta, context
                , encoded_age,encoded_env_data):
        batch_size = x.size(0)
        beta = beta.view(batch_size, 1, 1)  # (B, 1, 1) torch.Size([B, 1, 1])
        context = context.view(batch_size, 1, -1)  # (B, 1, F) torch.Size([B, 1, 256])--torch.Size([B, 1, 384])

        encoded_age = encoded_age.view(batch_size, 1, -1)  # (B,1,4)
        env_traj = encoded_env_data[0].to(x.device).view(batch_size, 1, -1)
        env_inten = encoded_env_data[1].to(x.device).view(batch_size, 1, -1)
        env_wind = encoded_env_data[2].to(x.device).view(batch_size, 1, -1)

        # 轨迹
        ctx_emb_env_age_traj = env_traj
        ctx_emb_env_age_traj = ctx_emb_env_age_traj / 100
        # 强度
        ctx_emb_env_age_inten = torch.cat([env_inten, encoded_age], dim=-1)
        ctx_emb_env_age_inten = ctx_emb_env_age_inten / 100
        # 风速
        ctx_emb_env_age_wind = torch.cat([env_wind, encoded_age], dim=-1)  # (B, 1, 20)
        ctx_emb_env_age_wind = ctx_emb_env_age_wind / 100

        time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)  # (B, 1, 3)
        ctx_emb = torch.cat([time_emb, context], dim=-1)  # (B, 1, F+3) torch.Size([B, 1, 259])

        x = self.concat1(ctx_emb, x)  # [B, 4, 512]
        # 【B,4,512】
        final_emb = x.permute(1, 0, 2)  # torch.Size([4, B, 512])
        final_emb = self.pos_emb(final_emb)  # torch.Size([4, B, 512])
        trans = self.transformer_encoder(final_emb).permute(1, 0, 2)  # torch.Size([B, 4, 512])
        trans = self.concat3(ctx_emb, trans)  # torch.Size([B, 4, 256])
        trans = self.concat4(ctx_emb, trans)  # torch.Size([B, 4, 128])
        trans = self.linear(ctx_emb, trans)  # torch.Size([B, 4, 4])  #

        x_traj = self.concat_env_age_x_traj_type1(ctx_emb_env_age_traj,
                                            trans[:, :, 0:2])  # (B, 1, 84) (B,4,4)  输出torch.Size([256, 4, 256])
        x_inten = self.concat_env_age_x_inten_type1(ctx_emb_env_age_inten, trans[:, :, 2:3])
        x_wind = self.concat_env_age_x_wind_type1(ctx_emb_env_age_wind, trans[:, :, 3:4])
        x = torch.cat((x_traj, x_inten, x_wind), dim=2)
        return x
    def pshare_and_ptask(self, x, beta, context
                , encoded_age,encoded_env_data):
        batch_size = x.size(0)
        beta = beta.view(batch_size, 1, 1)  # (B, 1, 1) torch.Size([B, 1, 1])
        context = context.view(batch_size, 1, -1)  # (B, 1, F) torch.Size([B, 1, 256])--torch.Size([B, 1, 384])

        encoded_age = encoded_age.view(batch_size, 1, -1)  # (B,1,4)
        env_traj = encoded_env_data[0].to(x.device).view(batch_size, 1, -1)
        env_inten = encoded_env_data[1].to(x.device).view(batch_size, 1, -1)
        env_wind = encoded_env_data[2].to(x.device).view(batch_size, 1, -1)

        # 轨迹
        ctx_emb_env_age_traj = env_traj
        ctx_emb_env_age_traj = ctx_emb_env_age_traj / 100
        # 强度
        ctx_emb_env_age_inten = torch.cat([env_inten, encoded_age], dim=-1)
        ctx_emb_env_age_inten = ctx_emb_env_age_inten / 100
        # 风速
        ctx_emb_env_age_wind = torch.cat([env_wind, encoded_age], dim=-1)  # (B, 1, 20)
        ctx_emb_env_age_wind = ctx_emb_env_age_wind / 100

        p_task = torch.cat((ctx_emb_env_age_traj, ctx_emb_env_age_inten, ctx_emb_env_age_wind), dim=2)

        time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)  # (B, 1, 3)
        ctx_emb = torch.cat([time_emb, context, p_task], dim=-1)  # (B, 1, F+3) torch.Size([B, 1, 347])

        x = self.concat1_type2(ctx_emb, x)  # [B, 4, 512]
        # 【B,4,512】
        final_emb = x.permute(1, 0, 2)  # torch.Size([4, B, 512])
        final_emb = self.pos_emb(final_emb)  # torch.Size([4, B, 512])
        trans = self.transformer_encoder(final_emb).permute(1, 0, 2)  # torch.Size([B, 4, 512])
        trans = self.concat3_type2(ctx_emb, trans)  # torch.Size([B, 4, 256])
        trans = self.concat4_type2(ctx_emb, trans)  # torch.Size([B, 4, 128])
        trans = self.linear_type2(ctx_emb, trans)  # torch.Size([B, 4, 4])  #直接从512缩小到4
        return trans
    def forward(self, x, beta, context
                , encoded_age,encoded_env_data
                ):
        trans = self.ptask_then_pshare(x, beta, context
                , encoded_age,encoded_env_data)
        return trans


        #1
        # return self.linear_true(trans)


class TransformerLinear(Module):

    def __init__(self, point_dim, context_dim, residual):
        super().__init__()
        self.residual = residual

        self.pos_emb = PositionalEncoding(d_model=128, dropout=0.1, max_len=24)
        self.y_up = nn.Linear(2, 128)
        self.ctx_up = nn.Linear(context_dim+3, 128)
        self.layer = nn.TransformerEncoderLayer(d_model=128, nhead=2, dim_feedforward=512)
        self.transformer_encoder = nn.TransformerEncoder(self.layer, num_layers=3)
        self.linear = nn.Linear(128, point_dim)

    def forward(self, x, beta, context):

        batch_size = x.size(0)
        beta = beta.view(batch_size, 1, 1)          # (B, 1, 1)
        context = context.view(batch_size, 1, -1)   # (B, 1, F)

        time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)  # (B, 1, 3)
        ctx_emb = torch.cat([time_emb, context], dim=-1)    # (B, 1, F+3)

        ctx_emb = self.ctx_up(ctx_emb)
        emb = self.y_up(x)
        final_emb = torch.cat([ctx_emb, emb], dim=1).permute(1,0,2)
        #pdb.set_trace()
        final_emb = self.pos_emb(final_emb)

        trans = self.transformer_encoder(final_emb)  # 13 * b * 128
        trans = trans[1:].permute(1,0,2)   # B * 12 * 128, drop the first one which is the z
        return self.linear(trans)

class LinearDecoder(Module):
    def __init__(self):
            super().__init__()
            self.act = F.leaky_relu
            self.layers = ModuleList([
                #nn.Linear(2, 64),
                nn.Linear(32, 64),
                nn.Linear(64, 128),
                nn.Linear(128, 256),
                nn.Linear(256, 512),
                nn.Linear(512, 256),
                nn.Linear(256, 128),
                nn.Linear(128, 12)
                #nn.Linear(2, 64),
                #nn.Linear(2, 64),
            ])
    def forward(self, code):

        out = code
        for i, layer in enumerate(self.layers):
            out = layer(out)
            if i < len(self.layers) - 1:
                out = self.act(out)
        return out
