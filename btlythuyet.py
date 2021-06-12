# Tìm hiểu method flat_map() trong tf.data.Dataset. Code ít nhất 1 ví dụ cho thấy sự khác biệt với với method map().
# Trả lời:
# Method flat_map() trong tf.data.Dataset là phương thức làm phẳng tập dữ liệu và ánh xạ hàm đã cho trong tham số truyền vào của phương thức với tập dữ liệu (dataset). Hàm được cung cấp trong tham số của phương thức phải trả về một dataset object.

""" Ví dụ: """

# Đầu tiên ta tạo một dataset với
# tf.data.Dataset.from_tensor_slices

import tensorflow as tf

dataset = tf.data.Dataset.from_tensor_slices([[[1,2, 3], [3,4,5]]])
for i in dataset:
  print(i)
  print(i.shape)
  
# Ta được output:
# tf.Tensor(
# [[1 2 3]
#  [3 4 5]], shape=(2, 3), dtype=int32)
# (2, 3)


# Áp dụng flat_map vào tập dữ liệu (dataset)

dataset = dataset.flat_map(lambda x : tf.data.Dataset.from_tensor_slices(x**2))
for i in dataset:
  print(i)
  
# Ta được output:
# tf.Tensor([1 4 9], shape=(3,), dtype=int32)
# tf.Tensor([ 9 16 25], shape=(3,), dtype=int32)
# Mảng trên đã được làm phẳng thành mảng 1 chiều

# Còn đối với method map(), nó chỉ xử lý trực tiếp trên mảng 2 chiều trên mà không làm phẳng nó thành mảng 1 chiều

dataset = tf.data.Dataset.from_tensor_slices([[[1,2, 3], [3,4,5]]])
dataset = dataset.map(lambda x: x*2)
print(list(dataset.as_numpy_iterator()))

# Ta được output:
# [array([[ 2,  4,  6],
#         [ 6,  8, 10]], dtype=int32)]


# Reference: https://www.gcptutorials.com/article/how-to-use-flat_map-method-in-tf.data.dataset
