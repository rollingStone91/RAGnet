# Groth16零知识证明系统API文档

## 概述

这是一个基于Groth16零知识证明系统的API服务，支持多种证明类型和动态向量长度切换。系统提供以下功能：

**要添加数据块请将jsonl文件放在./asset/data目录下，发起请求时client_id设定为与文件名一致（不包括后缀）**

- **Pedersen承诺 + Merkle树证明**: 基于Pedersen承诺和Merkle树的证明系统
- **Groth16零知识证明 + Merkle树证明**: 基于Groth16 ZKP和Merkle树的证明系统
- **Proof of Generation (PoG)**: 生成证明
- **动态向量长度**: 支持运行时切换Groth16电路的向量长度

## 功能特性

- **多种证明类型**: 支持Pedersen承诺、Groth16 ZKP、Merkle树证明
- **动态向量长度**: 支持128、256、384、512、640、768、896、1024、2048等向量长度
- **预生成电路**: 为每种向量长度预生成电路和密钥
- **RESTful API**: 提供完整的RESTful API接口
- **线程安全**: 使用读写锁保证并发安全
- **性能监控**: 提供时间和空间成本统计（时间单位为纳秒）

## 快速开始

### 1. 启动服务

```bash
go run .
```

服务将在 `http://localhost:8080` 启动。

## API接口

### 1. 动态向量长度管理

#### 1.1 获取当前向量长度
```bash
GET /vector_length
```

响应示例：
```json
{
  "code": 200,
  "data": {
    "current_length": 256,
    "supported_lengths": [128, 256, 384, 512, 640, 768, 896, 1024, 2048]
  }
}
```

#### 1.2 设置当前向量长度
```bash
POST /vector_length?length={length}
```

参数：
- `length`: 要设置的向量长度

示例：
```bash
curl -X POST "http://localhost:8080/vector_length?length=128"
```

### 2. Pedersen承诺 + Merkle树证明

#### 2.1 生成Pedersen承诺和Merkle树证明
```bash
POST /gen_pedersen_merkle_proof
```

请求体：
```json
{
  "client_id": "client1",
  "K": [1.0, 2.0, 3.0, ...],
  "Q": [4.0, 5.0, 6.0, ...],
  "data": "要证明的数据"
}
```

响应示例：
```json
{
  "code": 200,
  "proof_id": 12345,
  "time_cost": 150000000,
  "space_cost": 2048
}
```

#### 2.2 验证Pedersen承诺和Merkle树证明
```bash
POST /verify_pedersen_merkle_proof
```

请求体：
```json
{
  "proof_id": 12345
}
```

响应示例：
```json
{
  "code": 200,
  "msg": "ok",
  "time_cost": 25000000
}
```

### 3. Groth16零知识证明 + Merkle树证明

#### 3.1 生成Groth16证明和Merkle树证明
```bash
POST /gen_groth_merkle_proof
```

请求体：
```json
{
  "client_id": "client1",
  "K": [1.0, 2.0, 3.0, ...],
  "Q": [4.0, 5.0, 6.0, ...],
  "data": "要证明的数据"
}
```

响应示例：
```json
{
  "code": 200,
  "proof_id": 12346,
  "time_cost": 300000000,
  "space_cost": 4096
}
```

#### 3.2 验证Groth16证明和Merkle树证明
```bash
POST /verify_groth_merkle_proof
```

请求体：
```json
{
  "proof_id": 12346
}
```

响应示例：
```json
{
  "code": 200,
  "msg": "ok",
  "time_cost": 50000000
}
```

### 4. Proof of Generation (PoG)

#### 4.1 生成PoG证明
```bash
POST /gen_PoG
```

请求体：
```json
{
  "Q": [1.0, 2.0, 3.0, ...],
  "K": [4.0, 5.0, 6.0, ...],
  "data": "要证明的数据"
}
```

响应示例：
```json
{
  "code": 200,
  "proof_id": 12347,
  "time_cost": 100000,
  "space_cost": 40
}
```

#### 4.2 验证PoG证明
```bash
POST /verify_PoG
```

请求体：
```json
{
  "por_id": 12346,
  "pog_id": 12347
}
```

响应示例：
```json
{
  "code": 200,
  "msg": "ok",
  "time_cost": 100000
}
```

## 编程接口

### 动态向量长度管理

#### 设置向量长度
```go
import "gnarktest/config"

// 设置向量长度为128
success := config.GlobalVectorManager.SetVectorLength(128)
if !success {
    // 处理错误
}
```

#### 获取当前长度
```go
currentLength := config.GlobalVectorManager.GetCurrentLength()
```

#### 生成指定长度的证明
```go
import "gnarktest/pkg"

// 生成128长度的证明
q, k, l := pkg.GenRandVector(128)
proof, cost, err := pkg.GenGrothProofWithLength(q, k, 128)
```

#### 验证指定长度的证明
```go
valid, cost, err := pkg.VerifyGrothProofWithLength(proof, q, l, 128)
```

### 原有功能接口

#### 生成Groth16证明
```go
q, k, l := pkg.GenRandVector(256)
proof, cost, err := pkg.GenGrothProof(q, k)
```

#### 验证Groth16证明
```go
valid, cost, err := pkg.VerifyGrothProof(proof, q, l)
```

#### 生成Pedersen承诺
```go
commits, r, cost, err := pkg.GenCommitment(k)
```

#### 验证Pedersen承诺
```go
valid, cost, err := pkg.VerifyCommitment(commits, q, l, r)
```

#### 生成Merkle树证明
```go
proof, index, root, cost, err := pkg.GenMerkleProof(clientId, data)
```

#### 验证Merkle树证明
```go
valid, cost := pkg.VerifyMerkleProofByString(data, proof, index, root, sha256.New)
```

## 文件结构

```
asset/
├── data/                    # Merkle树数据文件
│   ├── client1.jsonl
│   └── client2.jsonl
├── groth/                   # Groth16电路和密钥
│   ├── length_128/
│   │   ├── cs
│   │   ├── proving.key
│   │   └── verifying.key
│   ├── length_256/
│   │   ├── cs
│   │   ├── proving.key
│   │   └── verifying.key
│   └── ...
├── pedersen/               # Pedersen承诺参数
│   └── g
└── proofs/                 # 生成的证明文件
    ├── 12345pedersen.json
    ├── 12346groth.json
    └── 12347pog.json
```

## 数据格式

### 请求数据格式

#### GenPoRReq (生成证明请求)
```json
{
  "client_id": "string",     // 客户端ID，对应数据文件名
  "K": [1.0, 2.0, ...],     // 密钥向量
  "Q": [3.0, 4.0, ...],     // 查询向量
  "data": "string"           // 要证明的数据
}
```

#### GenPoGReq (生成PoG请求)
```json
{
  "Q": [1.0, 2.0, ...],     // 查询向量
  "K": [3.0, 4.0, ...],     // 密钥向量
  "data": "string"           // 要证明的数据
}
```

#### VerifyPoRReq (验证证明请求)
```json
{
  "proof_id": 12345          // 证明ID
}
```

#### VerifyPoGReq (验证PoG请求)
```json
{
  "por_id": 12346,           // PoR证明ID
  "pog_id": 12347            // PoG证明ID
}
```

### 响应数据格式

#### 成功响应
```json
{
  "code": 200,
  "msg": "success message",
  "data": {...}              // 可选的数据字段
}
```

**注意**: 
- `time_cost` 字段返回纳秒为单位的整数值
- `space_cost` 字段返回字节为单位的整数值
- 在动态向量长度测试接口中，`generation_cost.time` 和 `verification_cost.time` 也返回纳秒为单位的整数值

#### 错误响应
```json
{
  "code": 400,
  "msg": "error message"
}
```

## 注意事项

1. **首次使用**: 首次使用前需要生成所有长度的电路和密钥
2. **内存使用**: 每种长度的电路和密钥都会加载到内存中
3. **向量长度匹配**: 确保输入向量的长度与当前设置的向量长度一致
4. **性能考虑**: 不同长度的电路性能差异较大，建议根据实际需求选择合适的长度
5. **数据文件**: 确保Merkle树数据文件存在且格式正确
6. **证明存储**: 生成的证明会保存到本地文件系统

## 测试

运行测试来验证功能：

```bash
# 测试动态向量长度功能
go test -v -run TestDynamicVectorLength

# 测试电路生成
go test -v -run TestGenerateCircuits

# 测试原有功能
go test -v -run TestHandler
```

## 错误处理

常见错误及解决方案：

1. **"circuit and keys for length X not found"**
   - 解决方案：运行 `POST /generate_circuit?length=X` 生成电路和密钥

2. **"vector length mismatch"**
   - 解决方案：确保输入向量的长度与当前设置的向量长度一致

3. **"unsupported vector length"**
   - 解决方案：检查是否使用了支持的长度

4. **"merkle proof verify failed"**
   - 解决方案：检查数据文件是否存在，数据是否正确

5. **"commitment verify failed"** 或 **"groth proof verify failed"**
   - 解决方案：检查输入数据是否正确，向量长度是否匹配