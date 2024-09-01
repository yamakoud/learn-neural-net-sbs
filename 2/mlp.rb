require 'matrix'

class Matrix
  # 行列の同じ場所の要素同士の積を計算する関数
  # アダマール積とも呼ばれる
  def hadamard(other)
    Matrix.build(row_count, column_count) do |i, j|
      self[i, j] * other[i, j]
    end
  end
end

def sigmoid(x)
  1 / (1 + Math.exp(-x))
end

def sigmoid_derivative(x)
  x * (1 - x)
end

class BackpropMLP
  def initialize(input_size, hidden_size, output_size, learning_rate)
    @w1 = Matrix.build(input_size, hidden_size) { rand(-1.0..1.0) }
    @w2 = Matrix.build(hidden_size, output_size) { rand(-1.0..1.0) }
    @learning_rate = learning_rate
  end

  def forward(x)
    @x = Matrix[x]
    @z1 = @x * @w1
    @a1 = @z1.map { |e| sigmoid(e) }
    @z2 = @a1 * @w2
    @a2 = @z2.map { |e| sigmoid(e) }
    @a2.to_a.flatten
  end

  def backward(y)
    y = Matrix[y]
    
    # 出力層の誤差
    d_output = (y - @a2).hadamard(@a2.map { |e| sigmoid_derivative(e) })
    
    # 隠れ層の誤差
    d_hidden = (d_output * @w2.transpose).hadamard(@a1.map { |e| sigmoid_derivative(e) })
    
    # 重みの更新
    @w2 += @a1.transpose * d_output * @learning_rate
    @w1 += @x.transpose * d_hidden * @learning_rate
  end

  def train(x, y, epochs)
    epochs.times do
      x.each_with_index do |input, i|
        output = forward(input)
        backward(y[i])
      end
    end
  end
end

# XOR問題のデータ
x = [
  [0, 0],
  [0, 1],
  [1, 0],
  [1, 1]
]

y = [
  [0],
  [1],
  [1],
  [0]
]

# モデルの初期化と学習
mlp = BackpropMLP.new(2, 2, 1, 0.1)
mlp.train(x, y, 10000)

# テスト
x.each_with_index do |input, i|
  prediction = mlp.forward(input)
  puts "入力: #{input}, 正解: #{y[i]}, 予測: #{prediction}"
end