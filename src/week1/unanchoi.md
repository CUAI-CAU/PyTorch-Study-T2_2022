## Tensor(텐서)

- 배열(array), 행렬(matrix)와 유사한 자료구조
- pytorch에서는 입출력 + 모델의 매개변수를 encoding한다.

### Tensor 생성

```python
torch.manual_seed(1729) # random seed fix

torch.empty(2,3) # initialized 된 data 
torch.rand(2,3) # 0~1 부터 random한 data
torch.arange(start, end, step)  # [start, end) 에서 array 생성

torch.FloatTensor()
torch.LongTensor()

-----------
torch.rand(2,3)
torch.randn(2,3)
torch.randint()
torch.randperm() # 정수 난수 순열

--------------------
# numpy array를 tensor로 생성 
data = [[1,2], [3,4]]
np_array = n.array(data)
tensor = torch.tensor(np_array)  # Shallow Copy
# 또는
tensor = torch.as_tensor(np_array) # Deep Copy
# 또는
tensor = torch.from_numpy(np_array) # Deep Copy

# 다시 numpy array 로 변환하고 싶을 때,
np_again = tensor.numpy()

----------------------------------------------------
# _like : 같은 shape 와 dtype으로 tensor를 만들고 싶을 때

arr = [[1,2,3],[2,3,4]]
t1 = torch.as_tensor(arr)

tensor = torch.zeros_like(t1)
tensor = torch.ones_like(t1)
tensor = torch.full_like(t1,3)
tensor = torch.empty_like(t1)
tensor = torch.randint_like(t1)
print(tensor)

---------------------------------
torch.eye(n) # diangonal entry = 1인 n x n diagonal array 생성
```

```python
torch.manual_seed(1729) # random seed fix

torch.empty(2,3) # initialized 된 data 
torch.rand(2,3) # 0~1 부터 random한 data
torch.arange(start, end, step)  # [start, end) 에서 array 생성

torch.FloatTensor()
torch.LongTensor()

-----------
torch.rand(2,3)
torch.randn(2,3)
torch.randint()
torch.randperm() # 정수 난수 순열

--------------------
# numpy array를 tensor로 생성 
data = [[1,2], [3,4]]
np_array = n.array(data)
tensor = torch.tensor(np_array)  # Shallow Copy
# 또는
tensor = torch.as_tensor(np_array) # Deep Copy
# 또는
tensor = torch.from_numpy(np_array) # Deep Copy

# 다시 numpy array 로 변환하고 싶을 때,
np_again = tensor.numpy()

----------------------------------------------------
# _like : 같은 shape 와 dtype으로 tensor를 만들고 싶을 때

arr = [[1,2,3],[2,3,4]]
t1 = torch.as_tensor(arr)

tensor = torch.zeros_like(t1)
tensor = torch.ones_like(t1)
tensor = torch.full_like(t1,3)
tensor = torch.empty_like(t1)
tensor = torch.randint_like(t1)
print(tensor)

---------------------------------
torch.eye(n) # diangonal entry = 1인 n x n diagonal array 생성
```

### Shape

shape는 텐서의 차원을 나타내는 튜플

```python
shape = (2,3,)

```

## Attribute

- 텐서의 속성

```python
tensor = torch.rand(3,4)

tensor.shape # 텐서의 차원
tensor.dtype # 텐서의 자료형
tensor.device # 텐서가 어떤 장치에 저장되었는지

tensor.int() # 정수형으로 데이터 바꾸기

```

## Operation

- GPU로의 device 전환

```python
cpu = 'cpu'
gpu = 'cuda'

if torch.cuda.is_available():
  tensor = tensor.to(gpu)
```

- 텐서 합치기

```python
tensor = torch.rand(3,3)
tensor2 = torch.cat([tensor, tensor, tensor)], dim = 1)

```

- Transpose

```python
x = torch.arange(1,13).reshape(3,4)
y = x.transpose(4,3)
```

- Permute → 차원의 순서를 재배치

```python
x = torch.rand(4,5,3)
z = x.permute(2,1,0) # 0 -> 2 , 1 -> 1 , 2->0 으로 재배치
```

- Indexing , Slicing

```python
x = torch.arange(1,13).reshape(3,4)

print(x)
print(x[1])
print(x[0][-1])
print(x[1:-1])
print(x[:2,2:])

'''
tensor([[ 1,  2,  3,  4],
        [ 5,  6,  7,  8],
        [ 9, 10, 11, 12]])
tensor([5, 6, 7, 8])
tensor(4)
tensor([[5, 6, 7, 8]])
tensor([[3, 4],
        [7, 8]])
'''

```

```python
print(x + y) # 성분 별 덧셈
print(x - y) # 성분 별 뺄셈
print(x * y) # 성분 별 곱셈
print(x / y) # 성분 별 나눗셈
print(x @ y) # 행렬 곱

print(torch.add(x, y)) # 성분 별 덧셈
print(torch.subtract(x, y)) # 성분 별 뺄셈
print(torch.multiply(x, y)) # 성분 별 곱셈
print(torch.divide(x, y)) # 성분 별 나눗셈

print(torch.matmul(x, y)), print(torch.mm(x,y)) # 행렬 곱

y = x ** 2 # 거듭제곱

```
