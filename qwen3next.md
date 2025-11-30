## Qwen3HybridLinearDecoderLayer

- self.linear_attn = Qwen3GatedDeltaNet
- self.mlp = Qwen2MoeSparseMoeBlock / Qwen2MoeMLP

## Qwen3GatedDeltaNet

**获取dp attention下的tp_rank和tp_size**

```python
self.attn_tp_rank = get_attention_tp_rank()
self.attn_tp_size = get_attention_tp_size()
```

**hiddenStates->qkvz和ba的投影**

ColumnParallelLinear按照输出维度切分
TP=1,参数形状W[hidden_size,projection_size_qkvz]
在projection_size_qkvz维度切分
in_proj_ba也同理

```python
self.in_proj_qkvz = ColumnParallelLinear(
    input_size=self.hidden_size,
    output_size=projection_size_qkvz,
    bias=False,
    quant_config=quant_config,
    tp_rank=self.attn_tp_rank,
    tp_size=self.attn_tp_size,
)
self.in_proj_ba = ColumnParallelLinear(
    input_size=self.hidden_size,
    output_size=projection_size_ba,
    bias=False,
    quant_config=None,
    tp_rank=self.attn_tp_rank,
    tp_size=self.attn_tp_size,
)
```

**qkv Conv1d**

> 这里的“conv1d”本质上只是一堆 1D 卷积核的参数容器,为了复用 TP + quant + weight_loader 的基础设施,把它存在了 `ColumnParallelLinear` 里,真正的卷积计算是在 `attn_backend.forward(..., conv_weights=...)` 里完成的

- conv_kernel_size, 卷积核长度（时间维度上的 kernel 长度,4）
- conv_dim = 2 * key_dim + value_dim,卷积的通道数

每个通道 c 有一个长度为 K 的核 在时间维度上做卷积
从“单个时间步”的视角看, 卷积其实就是对长度为 K 的时间窗口做一个线性变换输出一个标量

所以对于单个时间步,所有通道的 kernel 拼在一起就是一个矩阵(K,C)
和 Linear 的权重形状 一模一样
Linear(in_features=K, out_features=C),权重形状就是W: [k, C]

ColumnParallelLinear按 输出维（列并行） 切成 `tp_size` 份
即按照W: [k, C]的C维度(通道的维度)进行TP切分

```python
self.conv_dim = self.key_dim * 2 + self.value_dim
self.conv1d = ColumnParallelLinear(
    input_size=self.conv_kernel_size,
    output_size=self.conv_dim,
    bias=False,
    quant_config=None,
    tp_rank=self.attn_tp_rank,
    tp_size=self.attn_tp_size,
)
```

**norm**

```python
self.norm = RMSNormGated(
    self.head_v_dim,
    eps=self.layer_norm_epsilon,
    group_size=None,
    norm_before_gate=True,
    device=torch.get_device_module().current_device(),
    dtype=config.torch_dtype,
)
```

**out_proj**

```python
self.out_proj = RowParallelLinear(
    self.value_dim,
    self.hidden_size,
    bias=False,
    quant_config=quant_config,
    input_is_parallel=True,
    reduce_results=False,
    tp_rank=self.attn_tp_rank,
    tp_size=self.attn_tp_size,
)
```

**forward**

开了 dp_attention 后
DP-Attention 后端一般会对 token 做 “压缩 / 重排”,只返回 有效 token 的输出,长度可能 `< z 的长度`
但是后续的 `RMSNormGated` 期望 `core_attn_out` 和 `z` 形状一致
所以这里需要进行padding

DP-Attn 的核心是把不同 sequence 的 token 按位置 / 负载重新分配到不同 DP rank 上
backend 里面会

- 选出本 DP 分片真正需要算的那一部分 token
- 丢掉 / 不返回某些位置
  比如被路由到别的 DP rank 或者是 padding / 不参与本层计算的 token

```python
def forward(...):
    # 投影,卷积等
   	# ...
    z_shape_og = z.shape
    # reshape input data into 2D tensor
    core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
    z = z.reshape(-1, z.shape[-1])

    # Add padding for DP-Attn
    if is_dp_attention_enabled():
        core_attn_out_pad = torch.zeros_like(z)
        core_attn_out_pad[: core_attn_out.shape[0], :] = core_attn_out
        core_attn_out = core_attn_out_pad

    core_attn_out = self.norm(core_attn_out, z)
    core_attn_out = core_attn_out.reshape(z_shape_og)
    core_attn_out = core_attn_out.reshape(*core_attn_out.shape[:-2], -1)

    output, _ = self.out_proj(core_attn_out)
    return output
```

## Qwen3HybridAttentionDecoderLayer

```python
class Qwen3HybridAttentionDecoderLayer(nn.Module):
    def __init__(...):
        ...
        self.attn_tp_rank = get_attention_tp_rank()
        self.attn_tp_size = get_attention_tp_size()
        ...
        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            self.total_num_heads * (1 + self.attn_output_gate),
            self.total_num_kv_heads,
            ...
            tp_rank=self.attn_tp_rank,
            tp_size=self.attn_tp_size,
        )

        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            config.hidden_size,
            ...
            tp_rank=self.attn_tp_rank,
            tp_size=self.attn_tp_size,
        )
        ...
        self.attn = RadixAttention(...)
```

- `attn_tp_rank/size` 来自 `get_attention_tp_*()`；
- QKV 和 O 的 projection 都用这个 attention-tp group 做并行

> **QKVParallelLinear**
>
> QKVParallelLinear = 一个 融合 Q/K/V 的 ColumnParallelLinear
> 按 head 维做 TP (Q heads 均匀切、KV heads 可能切也可能复制)
> 支持 MQA/GQA(`total_num_kv_heads` 小于 `total_num_heads`)
>
> ```python
> self.qkv_proj = QKVParallelLinear(
>     config.hidden_size,
>     self.head_dim,
>     self.total_num_heads * (1 + self.attn_output_gate),
>     self.total_num_kv_heads,
>     ...
> )
> ```
>

## TP+DP Attention

线性 attention 的 Q/K/V/Z 投影,conv1d,out_proj,这些权重
确实是“按 attention_tp_rank 在 attention_tp_size 个 shard 之间切分的

> 可以这样理解,tp_size=16,如果不开dp_attention,attention的参数切分成16份分到每个tp_rank上; 但如果开了dp_attention且dp_size=4,则在每个dp_attention组内,attention参数被切成4份分在dp_attention组内4个attention_tp_rank上

https://zhuanlan.zhihu.com/p/1907142274942501938

1. 每个 DP 工作单元独立地处理不同类型的批次（例如预填充,解码,空闲）
2. 在进入混合专家 (MoE) 层之前,会在所有工作单元间进行全局汇聚 (all-gathered)
3. 而在通过 MoE 层处理之后,这些数据会再次被分发回各个工作单元
