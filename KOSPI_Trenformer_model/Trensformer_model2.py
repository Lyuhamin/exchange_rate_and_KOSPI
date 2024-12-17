import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LayerNormalization, MultiHeadAttention, Dropout
from tensorflow.keras.models import Model

# Transformer 블록 정의
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0.1):
    # Multi-Head Attention
    x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(inputs, inputs)
    x = Dropout(dropout)(x)
    x = LayerNormalization(epsilon=1e-6)(x + inputs)

    # Feed Forward Network (FFN)
    ffn_output = Dense(ff_dim, activation="relu")(x)
    ffn_output = Dense(inputs.shape[-1])(ffn_output)
    x = Dropout(dropout)(ffn_output)
    return LayerNormalization(epsilon=1e-6)(x + ffn_output)

# Transformer 모델 정의
def build_transformer(input_shape, head_size=512, num_heads=8, ff_dim=2048, num_blocks=4, dropout=0.1):
    inputs = Input(shape=input_shape)
    x = inputs

    # 여러 개의 Transformer Encoder Block
    for _ in range(num_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = Dense(1, activation='linear')(x[:, -1])  # 마지막 타임스텝만 예측
    return Model(inputs, x)

# 하이퍼파라미터 설정
input_shape = (3, 300, 3)  # 3개데이터셋, 303일 룩북데이터 시퀀스, 3개의 적용되는 피처
learning_rate = 0.0001

# 모델 생성
model = build_transformer(input_shape)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mse')
model.summary()
