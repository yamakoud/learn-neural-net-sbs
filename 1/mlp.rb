require 'matrix'

def sigmoid(x)
  1 / (1 + Math.exp(-x))
end

class ForwardOnlyMLP
  def initialize(input_size, hidden_size, output_size)
    @w1 = Matrix.build(input_size, hidden_size) { rand(-1.0..1.0) }
    @w2 = Matrix.build(hidden_size, output_size) { rand(-1.0..1.0) }
  end

  def forward(x)
    x = Matrix[x]
    z1 = x * @w1
    a1 = z1.map { |e| sigmoid(e) }
    z2 = a1 * @w2
    a2 = z2.map { |e| sigmoid(e) }
    a2.to_a.flatten
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

# モデルの初期化
mlp = ForwardOnlyMLP.new(2, 2, 1)

# テスト
x.each_with_index do |input, i|
  prediction = mlp.forward(input)
  puts "入力: #{input}, 正解: #{y[i]}, 予測: #{prediction}"
end
