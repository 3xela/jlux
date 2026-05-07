import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Float, Array
from .rope import RoPE
from .norms import RMSNorm, QKNorm


class FluxSingleAttention(eqx.Module):
    W_qkv : eqx.nn.Linear # (3D, D)
    W_o : eqx.nn.Linear # (D, D)

    rope : eqx.Module
    dim : int
    num_heads : int
    head_dim : int
    q_norm : eqx.Module
    k_norm : eqx.Module
    
    def __init__(self, dim , num_heads , key):
        self.dim = dim
        self.num_heads = num_heads
        assert dim % num_heads == 0
        self.head_dim = dim // num_heads
        key_qkv, key_o = jax.random.split(key, 2)

        self.W_o = eqx.nn.Linear(in_features = dim, out_features = dim, key = key_o)
        self.W_qkv = eqx.nn.Linear(in_features = self.dim, out_features = 3 * self.dim, key = key_qkv)
        self.rope = RoPE()
        self.q_norm = RMSNorm(dim=self.head_dim)
        self.k_norm = RMSNorm(dim=self.head_dim)

    def _split_head(self, x):
        #extract from w_proj(x) , reshape across head him
        T, _ = x.shape
        x = jnp.reshape(x, (T, self.num_heads, self.head_dim))
        # swap axes to (H, T, d_H)
        return jnp.swapaxes(x, 0, 1)

    def _merge_heads(self, x):
        H, T, d_H = x.shape
        #undo axes swap
        x = jnp.swapaxes(x, 0, 1)
        #reshape back to (T , D)
        x = jnp.reshape(x, (T, H * d_H))
        return x

    def __call__(self, x: Float[Array , "seq_len dim"], pos_ids):
        o_proj = jax.vmap(self.W_o)

        qkv_proj = jax.vmap(self.W_qkv)

        q_proj, k_proj, v_proj = jnp.split(qkv_proj(x), [self.dim, 2 * self.dim], axis = -1)

        # (H, T, d_H)
        Q = self._split_head(q_proj) 
        K = self._split_head(k_proj)
        V = self._split_head(v_proj)

        Q = self.q_norm(Q)
        K = self.k_norm(K)

        Q = self.rope(Q, pos_ids)
        K = self.rope(K, pos_ids)

        scores = Q @ jnp.swapaxes(K, -1 , -2) / jnp.sqrt(self.head_dim)
        weights = jax.nn.softmax(scores, axis = -1)
        context = weights @ V

        context = self._merge_heads(context)
        out = o_proj(context)
        return out


#TODO THIS IS DEPRECATED DO NOT CALL
class FluxDoubleAttention(eqx.Module):
    text_W_qkv : eqx.nn.Linear # (3D, D)
    image_W_qkv : eqx.nn.Linear # (3D, D)

    text_W_o : eqx.nn.Linear # (D, D)
    image_W_o : eqx.nn.Linear # (D, D)

    rope : eqx.Module
    dim : int
    num_heads : int
    head_dim : int

    text_q_norm : eqx.Module
    text_k_norm : eqx.Module

    image_q_norm: eqx.Module
    image_k_norm: eqx.Module
    
    def __init__(self, dim , num_heads , key):
        self.dim = dim
        self.num_heads = num_heads
        assert dim % num_heads == 0
        self.head_dim = dim // num_heads
        key_text_qkv, key_text_o, key_image_qkv, key_image_o = jax.random.split(key, 4)

        self.text_W_o = eqx.nn.Linear(in_features = dim, out_features = dim, key = key_text_o)
        self.text_W_qkv = eqx.nn.Linear(in_features = self.dim, out_features = 3 * self.dim, key = key_text_qkv)
        
        self.image_W_o = eqx.nn.Linear(in_features = dim, out_features = dim, key = key_image_o)
        self.image_W_qkv = eqx.nn.Linear(in_features = self.dim, out_features = 3 * self.dim, key = key_image_qkv)
        
        self.rope = RoPE()

        self.text_q_norm = RMSNorm(dim=self.head_dim)
        self.text_k_norm = RMSNorm(dim=self.head_dim)

        self.image_q_norm = RMSNorm(dim=self.head_dim)
        self.image_k_norm = RMSNorm(dim=self.head_dim)

    def _split_head(self, x):
        #extract from w_proj(x) , reshape across head him
        T, _ = x.shape
        x = jnp.reshape(x, (T, self.num_heads, self.head_dim))
        # swap axes to (H, T, d_H)
        return jnp.swapaxes(x, 0, 1)

    def _merge_heads(self, x):
        H, T, d_H = x.shape
        #undo axes swap
        x = jnp.swapaxes(x, 0, 1)
        #reshape back to (T , D)
        x = jnp.reshape(x, (T, H * d_H))
        return x

    def __call__(self, text: Float[Array , "seq_len dim"], image: Float[Array, "seq_len dim"] , pos_ids):
        text_o_proj = jax.vmap(self.text_W_o)
        image_o_proj = jax.vmap(self.image_W_o)

        text_qkv_proj = jax.vmap(self.text_W_qkv)
        image_qkv_proj = jax.vmap(self.image_W_qkv)

        text_q_proj, text_k_proj, text_v_proj = jnp.split(text_qkv_proj(text), [self.dim, 2 * self.dim], axis = -1)
        image_q_proj, image_k_proj, image_v_proj = jnp.split(image_qkv_proj(image), [self.dim, 2 * self.dim], axis = -1)

        # (H, T, d_H)
        text_Q = self._split_head(text_q_proj) 
        text_K = self._split_head(text_k_proj)
        text_V = self._split_head(text_v_proj)

        image_Q = self._split_head(image_q_proj)
        image_K = self._split_head(image_k_proj)
        image_V = self._split_head(image_v_proj)

        text_Q = self.text_q_norm(text_Q)
        text_K = self.text_k_norm(text_K)

        image_Q = self.image_q_norm(image_Q)
        image_K = self.image_k_norm(image_K)

        s_text = text_Q.shape[1]

        joint_Q = jnp.concat([text_Q, image_Q], axis = 1)
        joint_K = jnp.concat([text_K, image_K], axis = 1)
        joint_V = jnp.concat([text_V, image_V], axis = 1)

        joint_Q = self.rope(joint_Q, pos_ids)
        joint_K = self.rope(joint_K, pos_ids)

        scores = joint_Q @ jnp.swapaxes(joint_K, -1 , -2) / jnp.sqrt(self.head_dim)
        weights = jax.nn.softmax(scores, axis = -1)
        context = weights @ joint_V

        text_context, image_context = jnp.split(context, [s_text], axis = 1)

        text_context = self._merge_heads(text_context)
        image_context = self._merge_heads(image_context)

        text_out = text_o_proj(text_context)
        image_out = image_o_proj(image_context)
        return text_out, image_out


class FluxSelfAttention(eqx.Module):
    qkv : eqx.nn.Linear # (3D, D)
    o: eqx.nn.Linear
    norm: QKNorm

    num_heads: int
    dim: int
    head_dim: int
    
    def __init__(self, dim , num_heads , key):
        self.dim = dim
        self.num_heads = num_heads
        assert dim % num_heads == 0
        self.head_dim = dim // num_heads
        key_qkv, key_o = jax.random.split(key, 2)

        self.o = eqx.nn.Linear(in_features = dim, out_features = dim, key = key_o)
        self.qkv = eqx.nn.Linear(in_features = self.dim, out_features = 3 * self.dim, key = key_qkv)
        self.norm = QKNorm(self.head_dim)


    def qkv_proj(self, x):
        QKV = jax.vmap(self.qkv)(x)
        q_proj, k_proj, v_proj = jnp.split(QKV, [self.dim, 2 * self.dim], axis = -1)
        Q = self._split_head(q_proj) 
        K = self._split_head(k_proj)
        V = self._split_head(v_proj)
        return self.norm(Q,K,V)
        
    def out_proj(self, context):
        x = self._merge_heads(context)
        return jax.vmap(self.o)(x)

    def _split_head(self, x):
        #extract from w_proj(x) , reshape across head him
        T, _ = x.shape
        x = jnp.reshape(x, (T, self.num_heads, self.head_dim))
        # swap axes to (H, T, d_H)
        return jnp.swapaxes(x, 0, 1)

    def _merge_heads(self, x):
        H, T, d_H = x.shape
        #undo axes swap
        x = jnp.swapaxes(x, 0, 1)
        #reshape back to (T , D)
        x = jnp.reshape(x, (T, H * d_H))
        return x